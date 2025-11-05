"""Redis-based distributed cache for fast intermediate storage.

Caches architecture evaluations, optimizer state, and temporary results.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import pickle
from typing import Any, Dict, List, Optional

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from morphml.exceptions import DistributedError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class DistributedCache:
    """
    Redis-based distributed cache.

    Provides fast caching for:
    - Architecture evaluation results
    - Optimizer state
    - Temporary computation results
    - Worker metadata

    Args:
        redis_url: Redis connection URL (default: redis://localhost:6379)
        prefix: Key prefix for namespacing (default: 'morphml')
        default_ttl: Default time-to-live in seconds (default: None = no expiry)

    Example:
        >>> cache = DistributedCache('redis://localhost:6379')
        >>> cache.set('key', {'value': 42}, ttl=3600)
        >>> result = cache.get('key')
        >>> cache.cache_architecture_result('abc123', {'fitness': 0.95})
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "morphml",
        default_ttl: Optional[int] = None,
    ):
        """Initialize distributed cache."""
        if not REDIS_AVAILABLE:
            raise DistributedError("Redis not available. Install with: pip install redis")

        self.redis_url = redis_url
        self.prefix = prefix
        self.default_ttl = default_ttl

        try:
            self.client = redis.from_url(redis_url, decode_responses=False)
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis: {redis_url}")
        except redis.ConnectionError as e:
            raise DistributedError(f"Failed to connect to Redis: {e}")

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}:{key}"

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be pickled)
            ttl: Time-to-live in seconds (None = use default)
        """
        full_key = self._make_key(key)
        serialized = pickle.dumps(value)

        ttl = ttl or self.default_ttl

        if ttl:
            self.client.setex(full_key, ttl, serialized)
        else:
            self.client.set(full_key, serialized)

        logger.debug(f"Cached {key} (ttl={ttl})")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        full_key = self._make_key(key)
        value = self.client.get(full_key)

        if value is None:
            logger.debug(f"Cache miss: {key}")
            return None

        logger.debug(f"Cache hit: {key}")
        return pickle.loads(value)

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        full_key = self._make_key(key)
        result = self.client.delete(full_key)
        return result > 0

    def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        full_key = self._make_key(key)
        return self.client.exists(full_key) > 0

    def cache_architecture_result(
        self,
        arch_hash: str,
        result: Dict[str, Any],
        ttl: int = 86400,  # 24 hours
    ) -> None:
        """
        Cache architecture evaluation result.

        Args:
            arch_hash: Architecture hash
            result: Evaluation result
            ttl: Time-to-live (seconds)
        """
        key = f"arch:{arch_hash}"
        self.set(key, result, ttl=ttl)

    def get_architecture_result(self, arch_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached architecture result.

        Args:
            arch_hash: Architecture hash

        Returns:
            Cached result or None
        """
        key = f"arch:{arch_hash}"
        return self.get(key)

    def cache_optimizer_state(
        self,
        experiment_id: str,
        generation: int,
        state: Dict[str, Any],
        ttl: int = 3600,  # 1 hour
    ) -> None:
        """
        Cache optimizer state for quick recovery.

        Args:
            experiment_id: Experiment ID
            generation: Generation number
            state: Optimizer state
            ttl: Time-to-live (seconds)
        """
        key = f"optimizer:{experiment_id}:gen{generation}"
        self.set(key, state, ttl=ttl)

    def get_optimizer_state(self, experiment_id: str, generation: int) -> Optional[Dict[str, Any]]:
        """
        Get cached optimizer state.

        Args:
            experiment_id: Experiment ID
            generation: Generation number

        Returns:
            Cached state or None
        """
        key = f"optimizer:{experiment_id}:gen{generation}"
        return self.get(key)

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Key pattern (supports * wildcard)

        Returns:
            Number of keys deleted
        """
        full_pattern = self._make_key(pattern)
        keys = list(self.client.scan_iter(match=full_pattern))

        if keys:
            deleted = self.client.delete(*keys)
            logger.info(f"Invalidated {deleted} keys matching {pattern}")
            return deleted

        return 0

    def invalidate_experiment(self, experiment_id: str) -> int:
        """
        Invalidate all cache entries for experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Number of keys deleted
        """
        return self.invalidate_pattern(f"*:{experiment_id}:*")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Statistics dictionary
        """
        info = self.client.info("stats")

        return {
            "total_connections": info.get("total_connections_received", 0),
            "total_commands": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": (
                info.get("keyspace_hits", 0)
                / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0),
                    1,
                )
                * 100
            ),
        }

    def clear_all(self) -> None:
        """
        Clear all cache entries with prefix.

        Warning: This deletes all keys with the configured prefix!
        """
        pattern = f"{self.prefix}:*"
        keys = list(self.client.scan_iter(match=pattern))

        if keys:
            self.client.delete(*keys)
            logger.warning(f"Cleared {len(keys)} cache entries")

    def set_multiple(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Set multiple key-value pairs.

        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time-to-live for all keys
        """
        pipe = self.client.pipeline()

        for key, value in mapping.items():
            full_key = self._make_key(key)
            serialized = pickle.dumps(value)

            if ttl:
                pipe.setex(full_key, ttl, serialized)
            else:
                pipe.set(full_key, serialized)

        pipe.execute()
        logger.debug(f"Cached {len(mapping)} keys")

    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of found key-value pairs
        """
        full_keys = [self._make_key(k) for k in keys]
        values = self.client.mget(full_keys)

        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = pickle.loads(value)

        return result

    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment counter.

        Args:
            key: Counter key
            amount: Increment amount

        Returns:
            New counter value
        """
        full_key = self._make_key(key)
        return self.client.incr(full_key, amount)

    def decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement counter.

        Args:
            key: Counter key
            amount: Decrement amount

        Returns:
            New counter value
        """
        full_key = self._make_key(key)
        return self.client.decr(full_key, amount)

    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, or None if no TTL set
        """
        full_key = self._make_key(key)
        ttl = self.client.ttl(full_key)

        if ttl == -1:  # No TTL
            return None
        elif ttl == -2:  # Key doesn't exist
            return None
        else:
            return ttl

    def close(self) -> None:
        """Close Redis connection."""
        self.client.close()
        logger.info("Closed Redis connection")

"""Rate limiting middleware for MorphML API.

Prevents API abuse by limiting request rates per client.

Example:
    >>> from morphml.api.rate_limit import RateLimitMiddleware
    >>> app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
"""

from typing import Dict, Tuple
from datetime import datetime, timedelta
import time

try:
    from fastapi import Request, HTTPException, status
    from starlette.middleware.base import BaseHTTPMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    Request = None
    HTTPException = None
    BaseHTTPMiddleware = None

from morphml.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    Tracks request counts per client IP address.
    
    Attributes:
        requests_per_minute: Maximum requests per minute
        requests: Dictionary tracking requests per IP
    """
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
        self.cleanup_interval = 60  # Cleanup old entries every 60 seconds
        self.last_cleanup = time.time()
    
    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if request is allowed for client.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Cleanup old entries periodically
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup()
        
        # Get or create request list for this IP
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Remove requests older than 1 minute
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check if limit exceeded
        request_count = len(self.requests[client_ip])
        
        if request_count >= self.requests_per_minute:
            return False, 0
        
        # Add current request
        self.requests[client_ip].append(now)
        
        remaining = self.requests_per_minute - request_count - 1
        
        return True, remaining
    
    def _cleanup(self):
        """Remove old entries to prevent memory leak."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove IPs with no recent requests
        ips_to_remove = []
        for ip, req_times in self.requests.items():
            recent_requests = [t for t in req_times if t > minute_ago]
            if not recent_requests:
                ips_to_remove.append(ip)
            else:
                self.requests[ip] = recent_requests
        
        for ip in ips_to_remove:
            del self.requests[ip]
        
        self.last_cleanup = time.time()
        
        if ips_to_remove:
            logger.debug(f"Cleaned up {len(ips_to_remove)} inactive IPs")


class RateLimitMiddleware(BaseHTTPMiddleware if FASTAPI_AVAILABLE else object):
    """
    FastAPI middleware for rate limiting.
    
    Automatically limits requests per client IP.
    
    Example:
        >>> app.add_middleware(
        ...     RateLimitMiddleware,
        ...     requests_per_minute=60
        ... )
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI required for rate limiting")
        
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute)
        logger.info(f"Rate limiting enabled: {requests_per_minute} requests/minute")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        # Get client IP
        client_ip = request.client.host
        
        # Check rate limit
        allowed, remaining = self.limiter.is_allowed(client_ip)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response


def create_rate_limiter(requests_per_minute: int = 60) -> RateLimiter:
    """
    Create a rate limiter instance.
    
    Args:
        requests_per_minute: Maximum requests per minute
        
    Returns:
        RateLimiter instance
        
    Example:
        >>> limiter = create_rate_limiter(100)
        >>> allowed, remaining = limiter.is_allowed("192.168.1.1")
    """
    return RateLimiter(requests_per_minute)

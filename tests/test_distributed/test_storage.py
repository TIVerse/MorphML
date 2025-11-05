"""Tests for distributed storage backends.

Note: These tests require external services (PostgreSQL, Redis, S3/MinIO)
to be running. They are skipped if services are not available.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from morphml.core.dsl import Layer, SearchSpace


# Test database backend
class TestDatabaseManager:
    """Test PostgreSQL database manager."""
    
    @pytest.fixture
    def db_connection_string(self):
        """Get database connection string from environment."""
        return os.getenv(
            "MORPHML_TEST_DB",
            "postgresql://morphml:morphml@localhost:5432/morphml_test",
        )
    
    @pytest.fixture
    def db_available(self, db_connection_string):
        """Check if database is available."""
        try:
            from morphml.distributed.storage import DatabaseManager
            
            db = DatabaseManager(db_connection_string)
            db.get_session().close()
            return True
        except Exception:
            return False
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_DB"), reason="Database not configured"
    )
    def test_database_initialization(self, db_connection_string):
        """Test database initialization."""
        from morphml.distributed.storage import DatabaseManager
        
        db = DatabaseManager(db_connection_string)
        
        assert db.connection_string == db_connection_string
        assert db.engine is not None
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_DB"), reason="Database not configured"
    )
    def test_create_experiment(self, db_connection_string):
        """Test creating experiment."""
        from morphml.distributed.storage import DatabaseManager
        
        db = DatabaseManager(db_connection_string)
        
        exp_id = db.create_experiment(
            name=f"test_exp_{os.getpid()}",
            config={"optimizer": "genetic", "population_size": 50},
        )
        
        assert exp_id > 0
        
        # Verify experiment
        exp = db.get_experiment(exp_id)
        assert exp is not None
        assert exp.status == "running"
        
        # Cleanup
        db.delete_experiment(exp_id)
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_DB"), reason="Database not configured"
    )
    def test_save_and_retrieve_architecture(self, db_connection_string):
        """Test saving and retrieving architecture."""
        from morphml.distributed.storage import DatabaseManager
        
        db = DatabaseManager(db_connection_string)
        
        # Create experiment
        exp_id = db.create_experiment(
            name=f"test_arch_{os.getpid()}", config={}
        )
        
        # Create architecture
        space = SearchSpace("test")
        space.add_layers(
            Layer.input(shape=(3, 32, 32)),
            Layer.conv2d(filters=64),
            Layer.output(units=10),
        )
        graph = space.sample()
        
        # Save
        arch_id = db.save_architecture(
            exp_id,
            graph,
            fitness=0.95,
            metrics={"accuracy": 0.95, "params": 100000},
            generation=1,
            worker_id="test-worker",
        )
        
        assert arch_id > 0
        
        # Retrieve best
        best = db.get_best_architectures(exp_id, top_k=1)
        assert len(best) == 1
        assert best[0].fitness == 0.95
        
        # Cleanup
        db.delete_experiment(exp_id)


# Test cache backend
class TestDistributedCache:
    """Test Redis cache."""
    
    @pytest.fixture
    def redis_url(self):
        """Get Redis URL from environment."""
        return os.getenv("MORPHML_TEST_REDIS", "redis://localhost:6379/15")
    
    @pytest.fixture
    def cache_available(self, redis_url):
        """Check if Redis is available."""
        try:
            from morphml.distributed.storage import DistributedCache
            
            cache = DistributedCache(redis_url, prefix="test")
            cache.client.ping()
            cache.close()
            return True
        except Exception:
            return False
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_REDIS"), reason="Redis not configured"
    )
    def test_cache_initialization(self, redis_url):
        """Test cache initialization."""
        from morphml.distributed.storage import DistributedCache
        
        cache = DistributedCache(redis_url, prefix="test")
        
        assert cache.client is not None
        
        cache.close()
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_REDIS"), reason="Redis not configured"
    )
    def test_set_and_get(self, redis_url):
        """Test set and get operations."""
        from morphml.distributed.storage import DistributedCache
        
        cache = DistributedCache(redis_url, prefix="test")
        
        # Set
        cache.set("test_key", {"value": 42}, ttl=60)
        
        # Get
        result = cache.get("test_key")
        assert result == {"value": 42}
        
        # Cleanup
        cache.delete("test_key")
        cache.close()
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_REDIS"), reason="Redis not configured"
    )
    def test_cache_architecture_result(self, redis_url):
        """Test caching architecture result."""
        from morphml.distributed.storage import DistributedCache
        
        cache = DistributedCache(redis_url, prefix="test")
        
        # Cache result
        cache.cache_architecture_result(
            "abc123", {"fitness": 0.95, "accuracy": 0.95}, ttl=60
        )
        
        # Retrieve
        result = cache.get_architecture_result("abc123")
        assert result is not None
        assert result["fitness"] == 0.95
        
        # Cleanup
        cache.close()


# Test artifact store
class TestArtifactStore:
    """Test S3/MinIO artifact storage."""
    
    @pytest.fixture
    def s3_config(self):
        """Get S3 config from environment."""
        return {
            "bucket": os.getenv("MORPHML_TEST_BUCKET", "morphml-test"),
            "endpoint_url": os.getenv("MORPHML_TEST_S3_ENDPOINT"),
            "aws_access_key": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }
    
    @pytest.fixture
    def store_available(self, s3_config):
        """Check if S3/MinIO is available."""
        try:
            from morphml.distributed.storage import ArtifactStore
            
            store = ArtifactStore(**s3_config)
            store.list_keys()
            return True
        except Exception:
            return False
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_BUCKET"), reason="S3 not configured"
    )
    def test_store_initialization(self, s3_config):
        """Test artifact store initialization."""
        from morphml.distributed.storage import ArtifactStore
        
        store = ArtifactStore(**s3_config)
        
        assert store.bucket == s3_config["bucket"]
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_BUCKET"), reason="S3 not configured"
    )
    def test_upload_and_download_file(self, s3_config):
        """Test file upload and download."""
        from morphml.distributed.storage import ArtifactStore
        
        store = ArtifactStore(**s3_config)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            # Upload
            s3_key = f"test/{os.getpid()}/test.txt"
            store.upload_file(temp_path, s3_key)
            
            # Verify exists
            assert store.exists(s3_key)
            
            # Download
            download_path = temp_path + ".downloaded"
            store.download_file(s3_key, download_path)
            
            # Verify content
            with open(download_path, "r") as f:
                content = f.read()
            
            assert content == "test content"
            
            # Cleanup
            store.delete(s3_key)
            os.unlink(download_path)
        
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_BUCKET"), reason="S3 not configured"
    )
    def test_upload_and_download_bytes(self, s3_config):
        """Test bytes upload and download."""
        from morphml.distributed.storage import ArtifactStore
        
        store = ArtifactStore(**s3_config)
        
        # Upload bytes
        data = b"test binary data"
        s3_key = f"test/{os.getpid()}/test.bin"
        store.upload_bytes(data, s3_key)
        
        # Download bytes
        downloaded = store.download_bytes(s3_key)
        
        assert downloaded == data
        
        # Cleanup
        store.delete(s3_key)


# Test checkpoint manager
class TestCheckpointManager:
    """Test checkpoint manager."""
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_BUCKET"), reason="S3 not configured"
    )
    def test_checkpoint_manager_initialization(self):
        """Test checkpoint manager initialization."""
        from morphml.distributed.storage import ArtifactStore, CheckpointManager
        
        store = ArtifactStore(bucket=os.getenv("MORPHML_TEST_BUCKET", "morphml-test"))
        manager = CheckpointManager(store, checkpoint_interval=10)
        
        assert manager.checkpoint_interval == 10
    
    @pytest.mark.skipif(
        not os.getenv("MORPHML_TEST_BUCKET"), reason="S3 not configured"
    )
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoint."""
        from morphml.distributed.storage import ArtifactStore, CheckpointManager
        
        store = ArtifactStore(bucket=os.getenv("MORPHML_TEST_BUCKET", "morphml-test"))
        manager = CheckpointManager(store)
        
        # Save checkpoint
        exp_id = f"test_exp_{os.getpid()}"
        manager.save_checkpoint(
            experiment_id=exp_id,
            generation=10,
            optimizer_state={"population_size": 50},
            population=None,
        )
        
        # Load checkpoint
        checkpoint = manager.load_checkpoint(exp_id)
        
        assert checkpoint is not None
        assert checkpoint["generation"] == 10
        assert checkpoint["optimizer_state"]["population_size"] == 50
        
        # Cleanup
        manager.delete_all_checkpoints(exp_id)
    
    def test_should_checkpoint(self):
        """Test checkpoint interval logic."""
        from morphml.distributed.storage import CheckpointManager
        
        # Mock store since we don't need it
        store = Mock()
        manager = CheckpointManager(store, checkpoint_interval=10)
        
        assert manager.should_checkpoint(0)
        assert not manager.should_checkpoint(5)
        assert manager.should_checkpoint(10)
        assert manager.should_checkpoint(20)
        assert not manager.should_checkpoint(15)


def test_storage_imports():
    """Test that storage classes can be imported."""
    from morphml.distributed.storage import (
        Architecture,
        ArtifactStore,
        CheckpointManager,
        DatabaseManager,
        DistributedCache,
        Experiment,
    )
    
    # Verify classes exist
    assert DatabaseManager is not None
    assert Experiment is not None
    assert Architecture is not None
    assert DistributedCache is not None
    assert ArtifactStore is not None
    assert CheckpointManager is not None

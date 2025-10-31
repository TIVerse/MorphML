# Component 3: Distributed Storage

**Duration:** Week 5  
**LOC Target:** ~4,000  
**Dependencies:** Components 1-2

---

## 🎯 Objective

Implement distributed storage for:
1. **Results Database** - PostgreSQL for structured experiment data
2. **Shared Cache** - Redis for fast intermediate results
3. **Artifact Storage** - S3/MinIO for models and checkpoints
4. **Synchronization** - Consistent state across nodes

---

## 📋 Files to Create

### 1. `distributed/storage/database.py` (~1,500 LOC)

**PostgreSQL integration:**

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class Experiment(Base):
    """Experiment table."""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)
    search_space = Column(JSON)
    optimizer = Column(String(100))
    status = Column(String(50))  # 'running', 'completed', 'failed'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.datetime.utcnow)


class Architecture(Base):
    """Architecture evaluation table."""
    __tablename__ = 'architectures'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer)
    architecture_hash = Column(String(64), unique=True)
    architecture_json = Column(JSON)
    fitness = Column(Float)
    metrics = Column(JSON)  # accuracy, latency, params, etc.
    generation = Column(Integer)
    worker_id = Column(String(100))
    evaluated_at = Column(DateTime, default=datetime.datetime.utcnow)


class DatabaseManager:
    """
    Manage PostgreSQL database for experiments.
    
    Usage:
        db = DatabaseManager('postgresql://user:pass@localhost/morphml')
        db.save_architecture(arch, fitness, metrics)
        best = db.get_best_architectures(experiment_id, top_k=10)
    """
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string, pool_size=20)
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def create_experiment(self, name: str, config: Dict[str, Any]) -> int:
        """Create new experiment."""
        exp = Experiment(
            name=name,
            search_space=config.get('search_space'),
            optimizer=config.get('optimizer'),
            status='running'
        )
        
        self.session.add(exp)
        self.session.commit()
        
        return exp.id
    
    def save_architecture(
        self,
        experiment_id: int,
        architecture: ModelGraph,
        fitness: float,
        metrics: Dict[str, float],
        generation: int,
        worker_id: str
    ):
        """Save architecture evaluation."""
        arch_hash = architecture.hash()
        
        # Check if already evaluated
        existing = self.session.query(Architecture).filter_by(
            architecture_hash=arch_hash
        ).first()
        
        if existing:
            logger.debug(f"Architecture {arch_hash} already evaluated")
            return existing.id
        
        arch = Architecture(
            experiment_id=experiment_id,
            architecture_hash=arch_hash,
            architecture_json=architecture.to_dict(),
            fitness=fitness,
            metrics=metrics,
            generation=generation,
            worker_id=worker_id
        )
        
        self.session.add(arch)
        self.session.commit()
        
        return arch.id
    
    def get_best_architectures(
        self,
        experiment_id: int,
        top_k: int = 10
    ) -> List[Architecture]:
        """Get top-k architectures by fitness."""
        return self.session.query(Architecture).filter_by(
            experiment_id=experiment_id
        ).order_by(Architecture.fitness.desc()).limit(top_k).all()
    
    def get_architecture_by_hash(self, arch_hash: str) -> Optional[Architecture]:
        """Retrieve architecture by hash (avoid re-evaluation)."""
        return self.session.query(Architecture).filter_by(
            architecture_hash=arch_hash
        ).first()
```

---

### 2. `distributed/storage/cache.py` (~1,000 LOC)

**Redis cache:**

```python
import redis
import pickle
from typing import Optional

class DistributedCache:
    """
    Redis-based distributed cache.
    
    Caches:
    - Architecture evaluations
    - Optimizer state
    - Temporary results
    
    Usage:
        cache = DistributedCache('redis://localhost:6379')
        cache.set('arch:abc123', {'fitness': 0.95})
        result = cache.get('arch:abc123')
    """
    
    def __init__(self, redis_url: str = 'redis://localhost:6379'):
        self.client = redis.from_url(redis_url, decode_responses=False)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (will be pickled)
            ttl: Time-to-live in seconds
        """
        serialized = pickle.dumps(value)
        
        if ttl:
            self.client.setex(key, ttl, serialized)
        else:
            self.client.set(key, serialized)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = self.client.get(key)
        
        if value is None:
            return None
        
        return pickle.loads(value)
    
    def get_architecture_result(self, arch_hash: str) -> Optional[Dict]:
        """Get cached architecture evaluation."""
        return self.get(f'arch:{arch_hash}')
    
    def cache_architecture_result(
        self,
        arch_hash: str,
        result: Dict,
        ttl: int = 86400  # 24 hours
    ):
        """Cache architecture result."""
        self.set(f'arch:{arch_hash}', result, ttl=ttl)
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern."""
        for key in self.client.scan_iter(pattern):
            self.client.delete(key)
```

---

### 3. `distributed/storage/artifacts.py` (~1,000 LOC)

**S3/MinIO artifact storage:**

```python
import boto3
from botocore.exceptions import ClientError

class ArtifactStore:
    """
    S3-compatible artifact storage.
    
    Stores:
    - Trained models (.pt, .h5)
    - Checkpoints (.ckpt)
    - Plots and visualizations (.png, .html)
    - Logs (.txt, .json)
    
    Usage:
        store = ArtifactStore(bucket='morphml-artifacts')
        store.upload_model(model, 'experiments/exp1/best_model.pt')
        model = store.download_model('experiments/exp1/best_model.pt')
    """
    
    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,  # For MinIO
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None
    ):
        self.bucket = bucket
        
        self.s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        # Create bucket if not exists
        try:
            self.s3.head_bucket(Bucket=bucket)
        except ClientError:
            self.s3.create_bucket(Bucket=bucket)
    
    def upload_file(self, local_path: str, s3_key: str):
        """Upload file to S3."""
        self.s3.upload_file(local_path, self.bucket, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{s3_key}")
    
    def download_file(self, s3_key: str, local_path: str):
        """Download file from S3."""
        self.s3.download_file(self.bucket, s3_key, local_path)
        logger.info(f"Downloaded s3://{self.bucket}/{s3_key} to {local_path}")
    
    def upload_model(self, model: torch.nn.Module, s3_key: str):
        """Upload PyTorch model."""
        local_path = f'/tmp/{s3_key.split("/")[-1]}'
        torch.save(model.state_dict(), local_path)
        self.upload_file(local_path, s3_key)
    
    def download_model(self, s3_key: str) -> torch.nn.Module:
        """Download PyTorch model."""
        local_path = f'/tmp/{s3_key.split("/")[-1]}'
        self.download_file(s3_key, local_path)
        
        # Load model (requires architecture definition)
        state_dict = torch.load(local_path)
        return state_dict
    
    def list_artifacts(self, prefix: str) -> List[str]:
        """List all artifacts with prefix."""
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            return []
        
        return [obj['Key'] for obj in response['Contents']]
```

---

### 4. `distributed/storage/checkpointing.py` (~500 LOC)

**Distributed checkpointing:**

```python
class CheckpointManager:
    """
    Manage experiment checkpoints.
    
    Allows resuming interrupted experiments.
    """
    
    def __init__(self, artifact_store: ArtifactStore, cache: DistributedCache):
        self.store = artifact_store
        self.cache = cache
    
    def save_checkpoint(
        self,
        experiment_id: str,
        generation: int,
        optimizer_state: Dict,
        population: List[Individual]
    ):
        """Save experiment checkpoint."""
        checkpoint = {
            'experiment_id': experiment_id,
            'generation': generation,
            'optimizer_state': optimizer_state,
            'population': [ind.to_dict() for ind in population],
            'timestamp': time.time()
        }
        
        # Save to cache (fast recovery)
        self.cache.set(f'checkpoint:{experiment_id}', checkpoint, ttl=3600)
        
        # Save to S3 (persistent)
        local_path = f'/tmp/checkpoint_{experiment_id}_{generation}.pkl'
        with open(local_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        s3_key = f'checkpoints/{experiment_id}/gen_{generation}.pkl'
        self.store.upload_file(local_path, s3_key)
        
        logger.info(f"Saved checkpoint for generation {generation}")
    
    def load_checkpoint(self, experiment_id: str) -> Optional[Dict]:
        """Load latest checkpoint."""
        # Try cache first
        checkpoint = self.cache.get(f'checkpoint:{experiment_id}')
        
        if checkpoint:
            return checkpoint
        
        # Load from S3
        artifacts = self.store.list_artifacts(f'checkpoints/{experiment_id}/')
        
        if not artifacts:
            return None
        
        # Get latest
        latest = sorted(artifacts)[-1]
        local_path = f'/tmp/{latest.split("/")[-1]}'
        self.store.download_file(latest, local_path)
        
        with open(local_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return checkpoint
```

---

## 🧪 Tests

```python
def test_database_save():
    """Test saving architecture to database."""
    db = DatabaseManager('postgresql://localhost/test')
    
    exp_id = db.create_experiment('test_exp', {})
    
    arch = ModelGraph(...)
    db.save_architecture(exp_id, arch, 0.95, {'accuracy': 0.95}, 1, 'worker1')
    
    best = db.get_best_architectures(exp_id, top_k=1)
    assert len(best) == 1
    assert best[0].fitness == 0.95
```

---

## ✅ Deliverables

- [ ] PostgreSQL database schema and manager
- [ ] Redis distributed cache
- [ ] S3/MinIO artifact storage
- [ ] Checkpoint save/load system
- [ ] Architecture deduplication via hashing

---

**Next:** `04_fault_tolerance.md`

"""S3/MinIO artifact storage for models and checkpoints.

Provides object storage for large binary artifacts.

Author: Eshan Roy <eshanized@proton.me>
Organization: TONMOY INFRASTRUCTURE & VISION
"""

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from morphml.exceptions import DistributedError
from morphml.logging_config import get_logger

logger = get_logger(__name__)


class ArtifactStore:
    """
    S3-compatible artifact storage.
    
    Stores large binary artifacts:
    - Trained model weights (.pt, .h5, .ckpt)
    - Architecture checkpoints
    - Visualization plots (.png, .html)
    - Training logs (.txt, .json)
    
    Compatible with both AWS S3 and MinIO.
    
    Args:
        bucket: S3 bucket name
        endpoint_url: Custom endpoint for MinIO (None for AWS S3)
        aws_access_key: AWS access key ID (or from environment)
        aws_secret_key: AWS secret access key (or from environment)
        region: AWS region (default: us-east-1)
    
    Example:
        >>> # AWS S3
        >>> store = ArtifactStore(bucket='morphml-artifacts')
        >>> 
        >>> # MinIO
        >>> store = ArtifactStore(
        ...     bucket='morphml',
        ...     endpoint_url='http://localhost:9000',
        ...     aws_access_key='minioadmin',
        ...     aws_secret_key='minioadmin'
        ... )
        >>> 
        >>> # Upload
        >>> store.upload_file('model.pt', 'experiments/exp1/model.pt')
        >>> 
        >>> # Download
        >>> store.download_file('experiments/exp1/model.pt', 'model.pt')
    """
    
    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region: str = "us-east-1",
    ):
        """Initialize artifact store."""
        if not BOTO3_AVAILABLE:
            raise DistributedError(
                "boto3 not available. Install with: pip install boto3"
            )
        
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        
        # Create S3 client
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_key
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
        )
        
        # Create bucket if not exists
        self._ensure_bucket_exists()
        
        logger.info(f"Initialized artifact store: {bucket} (endpoint: {endpoint_url or 'AWS S3'})")
    
    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            logger.debug(f"Bucket {self.bucket} exists")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code == "404":
                # Bucket doesn't exist, create it
                try:
                    self.s3.create_bucket(Bucket=self.bucket)
                    logger.info(f"Created bucket: {self.bucket}")
                except ClientError as create_error:
                    raise DistributedError(
                        f"Failed to create bucket {self.bucket}: {create_error}"
                    )
            else:
                raise DistributedError(f"Failed to access bucket {self.bucket}: {e}")
    
    def upload_file(
        self, local_path: str, s3_key: str, metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Upload file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key (path in bucket)
            metadata: Optional metadata tags
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata
            
            self.s3.upload_file(local_path, self.bucket, s3_key, ExtraArgs=extra_args)
            
            logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{s3_key}")
        
        except ClientError as e:
            raise DistributedError(f"Failed to upload {local_path}: {e}")
    
    def download_file(self, s3_key: str, local_path: str) -> None:
        """
        Download file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
        """
        try:
            # Create parent directories
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.s3.download_file(self.bucket, s3_key, local_path)
            
            logger.info(f"Downloaded s3://{self.bucket}/{s3_key} to {local_path}")
        
        except ClientError as e:
            raise DistributedError(f"Failed to download {s3_key}: {e}")
    
    def upload_bytes(
        self, data: bytes, s3_key: str, metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Upload bytes directly to S3.
        
        Args:
            data: Bytes to upload
            s3_key: S3 object key
            metadata: Optional metadata
        """
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata
            
            self.s3.put_object(
                Bucket=self.bucket, Key=s3_key, Body=data, **extra_args
            )
            
            logger.info(f"Uploaded {len(data)} bytes to s3://{self.bucket}/{s3_key}")
        
        except ClientError as e:
            raise DistributedError(f"Failed to upload bytes to {s3_key}: {e}")
    
    def download_bytes(self, s3_key: str) -> bytes:
        """
        Download bytes from S3.
        
        Args:
            s3_key: S3 object key
        
        Returns:
            Downloaded bytes
        """
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            data = response["Body"].read()
            
            logger.info(f"Downloaded {len(data)} bytes from s3://{self.bucket}/{s3_key}")
            
            return data
        
        except ClientError as e:
            raise DistributedError(f"Failed to download {s3_key}: {e}")
    
    def exists(self, s3_key: str) -> bool:
        """
        Check if object exists.
        
        Args:
            s3_key: S3 object key
        
        Returns:
            True if object exists
        """
        try:
            self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False
    
    def delete(self, s3_key: str) -> None:
        """
        Delete object.
        
        Args:
            s3_key: S3 object key
        """
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket}/{s3_key}")
        except ClientError as e:
            raise DistributedError(f"Failed to delete {s3_key}: {e}")
    
    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects with prefix.
        
        Args:
            prefix: Key prefix to filter
            max_keys: Maximum number of keys to return
        
        Returns:
            List of object metadata dictionaries
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix, MaxKeys=max_keys
            )
            
            if "Contents" not in response:
                return []
            
            return [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                }
                for obj in response["Contents"]
            ]
        
        except ClientError as e:
            raise DistributedError(f"Failed to list objects: {e}")
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List object keys with prefix.
        
        Args:
            prefix: Key prefix to filter
        
        Returns:
            List of object keys
        """
        objects = self.list_objects(prefix)
        return [obj["key"] for obj in objects]
    
    def get_metadata(self, s3_key: str) -> Dict[str, Any]:
        """
        Get object metadata.
        
        Args:
            s3_key: S3 object key
        
        Returns:
            Metadata dictionary
        """
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            
            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "content_type": response.get("ContentType"),
                "metadata": response.get("Metadata", {}),
            }
        
        except ClientError as e:
            raise DistributedError(f"Failed to get metadata for {s3_key}: {e}")
    
    def get_presigned_url(
        self, s3_key: str, expiration: int = 3600, method: str = "get_object"
    ) -> str:
        """
        Generate presigned URL for temporary access.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds
            method: S3 method ('get_object' or 'put_object')
        
        Returns:
            Presigned URL
        """
        try:
            url = self.s3.generate_presigned_url(
                method,
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
            
            return url
        
        except ClientError as e:
            raise DistributedError(f"Failed to generate presigned URL: {e}")
    
    def copy(self, source_key: str, dest_key: str) -> None:
        """
        Copy object within bucket.
        
        Args:
            source_key: Source object key
            dest_key: Destination object key
        """
        try:
            copy_source = {"Bucket": self.bucket, "Key": source_key}
            self.s3.copy_object(
                CopySource=copy_source, Bucket=self.bucket, Key=dest_key
            )
            
            logger.info(f"Copied {source_key} to {dest_key}")
        
        except ClientError as e:
            raise DistributedError(f"Failed to copy {source_key}: {e}")
    
    def delete_prefix(self, prefix: str) -> int:
        """
        Delete all objects with prefix.
        
        Args:
            prefix: Key prefix
        
        Returns:
            Number of objects deleted
        """
        keys = self.list_keys(prefix)
        
        if not keys:
            return 0
        
        # Delete in batches of 1000 (S3 limit)
        deleted = 0
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            
            try:
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": key} for key in batch]},
                )
                
                deleted += len(batch)
            
            except ClientError as e:
                logger.error(f"Failed to delete batch: {e}")
        
        logger.info(f"Deleted {deleted} objects with prefix {prefix}")
        
        return deleted
    
    def get_total_size(self, prefix: str = "") -> int:
        """
        Get total size of objects with prefix.
        
        Args:
            prefix: Key prefix
        
        Returns:
            Total size in bytes
        """
        objects = self.list_objects(prefix)
        return sum(obj["size"] for obj in objects)

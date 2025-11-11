"""Authentication utilities for MorphML API.

Provides JWT-based authentication for the REST API.

Example:
    >>> from morphml.api.auth import create_access_token, verify_token
    >>> token = create_access_token({"sub": "user@example.com"})
    >>> payload = verify_token(token)
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext

    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False
    jwt = None
    CryptContext = None

from morphml.logging_config import get_logger

logger = get_logger(__name__)

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
if JOSE_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
else:
    pwd_context = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        True if password matches
    """
    if not JOSE_AVAILABLE:
        raise ImportError("python-jose and passlib required for authentication")

    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    if not JOSE_AVAILABLE:
        raise ImportError("python-jose and passlib required for authentication")

    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Optional expiration time

    Returns:
        JWT token string

    Example:
        >>> token = create_access_token({"sub": "user@example.com"})
    """
    if not JOSE_AVAILABLE:
        raise ImportError("python-jose required for authentication")

    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded payload or None if invalid

    Example:
        >>> payload = verify_token(token)
        >>> if payload:
        ...     user_email = payload.get("sub")
    """
    if not JOSE_AVAILABLE:
        raise ImportError("python-jose required for authentication")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        return None


# Simple in-memory user store (replace with database in production)
fake_users_db = {
    "admin@morphml.com": {
        "username": "admin@morphml.com",
        "full_name": "Admin User",
        "email": "admin@morphml.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
    }
}


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate user with username and password.

    Args:
        username: Username (email)
        password: Plain text password

    Returns:
        User dict if authenticated, None otherwise
    """
    user = fake_users_db.get(username)

    if not user:
        return None

    if not verify_password(password, user["hashed_password"]):
        return None

    return user


def create_user(username: str, password: str, full_name: str = "") -> Dict[str, Any]:
    """
    Create a new user.

    Args:
        username: Username (email)
        password: Plain text password
        full_name: Full name

    Returns:
        Created user dict
    """
    if username in fake_users_db:
        raise ValueError(f"User {username} already exists")

    hashed_password = get_password_hash(password)

    user = {
        "username": username,
        "full_name": full_name,
        "email": username,
        "hashed_password": hashed_password,
        "disabled": False,
    }

    fake_users_db[username] = user

    logger.info(f"Created user: {username}")

    return user

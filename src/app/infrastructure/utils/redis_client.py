"""Redis Client Utility

Provides a singleton Redis client instance configured via environment variables or config files.
"""
import logging
from typing import Optional, Any
import redis

from app.infrastructure.config.config_loader import get_config, get_config_int, get_config_bool

logger = logging.getLogger(__name__)

# Singleton Redis client instance
_redis_client: Optional[redis.Redis] = None
_initialized: bool = False

def get_redis_client() -> Optional[redis.Redis]:
    """Get or create the shared Redis client instance.
    
    Returns:
        Redis client instance if configured and enabled, None otherwise.
    """
    global _redis_client, _initialized
    
    if _initialized:
        return _redis_client
        
    # Check if Redis is enabled
    use_redis = get_config_bool("use_redis", False, section="redis")
    if not use_redis:
        logger.info("Redis is disabled (use_redis=False). CircuitBreakers will use local memory.")
        _initialized = True
        return None
        
    try:
        host = get_config("host", "localhost", section="redis")
        port = get_config_int("port", 6379, section="redis")
        db = get_config_int("db", 0, section="redis")
        max_connections = get_config_int("max_connections", 50, section="redis")
        
        # Create connection pool
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            max_connections=max_connections,
            decode_responses=True # Ensure we get strings, not bytes
        )
        
        # Create client
        client = redis.Redis(connection_pool=pool)
        
        # Test connection
        client.ping()
        
        _redis_client = client
        logger.info(f"Redis connected: {host}:{port}/{db}")
        
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}. CircuitBreakers will use local memory.")
        _redis_client = None
        
    _initialized = True
    return _redis_client

def reset_redis_client():
    """Reset the Redis client (useful for testing)."""
    global _redis_client, _initialized
    if _redis_client:
        try:
            _redis_client.close()
        except:
            pass
    _redis_client = None
    _initialized = False

# config_loader.py
"""
Configuration loader with environment variable substitution support
"""

import os
import yaml
import re
from typing import Any, Dict
from dotenv import load_dotenv

class ConfigLoader:
    """Load and process configuration with environment variable substitution"""
    
    def __init__(self, config_file: str = 'config.yaml'):
        """Initialize config loader
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        load_dotenv()  # Load .env file
        
    def load(self) -> Dict[str, Any]:
        """Load configuration with environment variable substitution
        
        Returns:
            Dictionary containing processed configuration
        """
        with open(self.config_file, 'r') as f:
            config_text = f.read()
            
        # Substitute environment variables
        config_text = self._substitute_env_vars(config_text)
        
        # Parse YAML
        config = yaml.safe_load(config_text)
        
        return config
    
    def _substitute_env_vars(self, text: str) -> str:
        """Substitute environment variables in configuration text
        
        Supports patterns like:
        - ${VAR_NAME} - required variable
        - ${VAR_NAME:-default_value} - optional with default
        
        Args:
            text: Raw configuration text
            
        Returns:
            Text with environment variables substituted
        """
        def replace_var(match):
            full_match = match.group(1)
            
            if ':-' in full_match:
                # Variable with default value
                var_name, default_value = full_match.split(':-', 1)
                value = os.getenv(var_name, default_value)
            else:
                # Required variable
                var_name = full_match
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(f"Required environment variable '{var_name}' not found")
            
            # Convert boolean strings
            if isinstance(value, str):
                if value.lower() in ('true', 'yes', '1', 'on'):
                    return 'true'
                elif value.lower() in ('false', 'no', '0', 'off'):
                    return 'false'
                    
            return str(value)
        
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, text)
    
    def get_database_url(self) -> str:
        """Get database URL, constructing from components if needed"""
        # First try direct DATABASE_URL
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return database_url
            
        # Construct from components
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB', 'breakout_db')
        
        if not all([user, password]):
            raise ValueError("Database credentials not found in environment variables")
            
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    def get_redis_url(self) -> str:
        """Get Redis URL, constructing from components if needed"""
        # First try direct REDIS_URL
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            return redis_url
            
        # Construct from components
        password = os.getenv('REDIS_PASSWORD', '')
        host = os.getenv('REDIS_HOST', 'localhost')
        port = os.getenv('REDIS_PORT', '6379')
        db = os.getenv('REDIS_DB', '0')
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"

# Global config instance
config_loader = ConfigLoader()

def load_config() -> Dict[str, Any]:
    """Load application configuration"""
    return config_loader.load()

def get_database_url() -> str:
    """Get database URL"""
    return config_loader.get_database_url()

def get_redis_url() -> str:
    """Get Redis URL"""
    return config_loader.get_redis_url()

def validate_required_env_vars():
    """Validate that all required environment variables are set"""
    required_vars = [
        'SECRET_KEY',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'TWELVEDATA_API_KEY',
        'ALPHAVANTAGE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please check your .env file or environment configuration."
        )

if __name__ == "__main__":
    # Test configuration loading
    try:
        validate_required_env_vars()
        config = load_config()
        print("✅ Configuration loaded successfully")
        print(f"Database URL: {get_database_url()}")
        print(f"Redis URL: {get_redis_url()}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
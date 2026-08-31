"""Global Application Configuration and Security Utilities."""
import os
import base64
import hashlib
from pathlib import Path
from typing import Optional
from loguru import logger
from cryptography.fernet import Fernet
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
PAIRS_CONFIG_DIR = CONFIG_DIR / "pairs"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "pmm_engine.db"

# Ensure runtime directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class AppSettings(BaseSettings):
    """Environment configuration settings."""
    app_env: str = "production"
    log_level: str = "INFO"
    web_host: str = "0.0.0.0"
    web_port: int = 8502
    secret_key: str = "pmm-super-secret-master-key-hedge-v3-change-in-prod"
    default_exchange: str = "binance"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = AppSettings()

# Configure Loguru logger
logger.add(
    LOGS_DIR / "pmm_engine_{time:YYYY-MM-DD}.log",
    rotation="50 MB",
    retention="10 days",
    level=settings.log_level,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
)


def _get_fernet() -> Fernet:
    """Generate deterministic Fernet key from master secret."""
    key = hashlib.sha256(settings.secret_key.encode("utf-8")).digest()
    fernet_key = base64.urlsafe_b64encode(key)
    return Fernet(fernet_key)


def encrypt_secret(plain_text: str) -> str:
    """Encrypt sensitive string (API Key / Secret)."""
    if not plain_text:
        return ""
    fernet = _get_fernet()
    return fernet.encrypt(plain_text.encode("utf-8")).decode("utf-8")


def decrypt_secret(cipher_text: str) -> str:
    """Decrypt sensitive string."""
    if not cipher_text:
        return ""
    try:
        fernet = _get_fernet()
        return fernet.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to decrypt secret: {e}")
        return ""

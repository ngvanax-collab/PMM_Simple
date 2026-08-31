"""Configuration and Credential Storage Management."""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import orjson
from loguru import logger

from app.config import CONFIG_DIR, PAIRS_CONFIG_DIR, decrypt_secret, encrypt_secret
from app.models.config import ExchangeCredentials, GlobalConfig, PairConfig
from app.persistence.db import db


def sanitize_symbol_filename(symbol: str) -> str:
    """Convert symbol like SOL/USDT:USDT to SOL_USDT_USDT.json filename."""
    return symbol.replace("/", "_").replace(":", "_") + ".json"


class ConfigStore:
    """Manages file-based and database-backed configuration store."""

    def __init__(self):
        self.global_config_path = CONFIG_DIR / "global.json"
        self.pairs_dir = PAIRS_CONFIG_DIR
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        """Create default configs if they do not exist."""
        if not self.global_config_path.exists():
            default_global = GlobalConfig()
            self.save_global_config(default_global)

    def load_global_config(self) -> GlobalConfig:
        """Load global configuration from disk."""
        if self.global_config_path.exists():
            try:
                with open(self.global_config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return GlobalConfig(**data)
            except Exception as e:
                logger.error(f"Error loading global config, fallback to default: {e}")
        default_cfg = GlobalConfig()
        self.save_global_config(default_cfg)
        return default_cfg

    def save_global_config(self, config: GlobalConfig) -> None:
        """Save global configuration to disk."""
        with open(self.global_config_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2)
        logger.info("Global configuration saved")

    def load_all_pair_configs(self) -> Dict[str, PairConfig]:
        """Load all pair configurations."""
        configs: Dict[str, PairConfig] = {}
        for file_path in self.pairs_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pair = PairConfig(**data)
                configs[pair.symbol] = pair
            except Exception as e:
                logger.error(f"Error loading pair config from {file_path}: {e}")
        return configs

    def load_pair_config(self, symbol: str) -> Optional[PairConfig]:
        """Load specific pair configuration."""
        file_path = self.pairs_dir / sanitize_symbol_filename(symbol)
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return PairConfig(**data)
            except Exception as e:
                logger.error(f"Error loading pair config for {symbol}: {e}")
        return None

    def save_pair_config(self, config: PairConfig) -> None:
        """Save specific pair configuration."""
        file_path = self.pairs_dir / sanitize_symbol_filename(config.symbol)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2)
        logger.info(f"Saved pair configuration for {config.symbol}")

    def delete_pair_config(self, symbol: str) -> bool:
        """Delete pair configuration file."""
        file_path = self.pairs_dir / sanitize_symbol_filename(symbol)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted pair configuration for {symbol}")
            return True
        return False


class CredentialStore:
    """Stores and retrieves encrypted API credentials."""

    @staticmethod
    async def save_credentials(creds: ExchangeCredentials) -> None:
        """Encrypt and store credentials in DB."""
        assert db._conn is not None
        enc_key = encrypt_secret(creds.api_key)
        enc_secret = encrypt_secret(creds.api_secret)
        enc_passphrase = encrypt_secret(creds.passphrase) if creds.passphrase else None

        async with db._lock:
            await db._conn.execute(
                """
                INSERT OR REPLACE INTO credentials (
                    exchange, api_key_encrypted, api_secret_encrypted, passphrase_encrypted, testnet, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    creds.exchange.lower(),
                    enc_key,
                    enc_secret,
                    enc_passphrase,
                    1 if creds.testnet else 0,
                    time.time(),
                )
            )
            await db._conn.commit()
        logger.info(f"Encrypted credentials saved for exchange: {creds.exchange}")

    @staticmethod
    async def load_credentials(exchange: str = "binance") -> Optional[ExchangeCredentials]:
        """Load and decrypt credentials from DB."""
        if db._conn is None:
            await db.connect()
        assert db._conn is not None

        async with db._conn.execute(
            "SELECT * FROM credentials WHERE exchange = ?", (exchange.lower(),)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None

            raw_key = decrypt_secret(row["api_key_encrypted"])
            raw_secret = decrypt_secret(row["api_secret_encrypted"])
            raw_pass = decrypt_secret(row["passphrase_encrypted"]) if row["passphrase_encrypted"] else None

            return ExchangeCredentials(
                exchange=row["exchange"],
                api_key=raw_key,
                api_secret=raw_secret,
                passphrase=raw_pass,
                testnet=bool(row["testnet"]),
            )


# Global instances
config_store = ConfigStore()
credential_store = CredentialStore()

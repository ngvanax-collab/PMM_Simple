"""Async SQLite Database Layer with WAL Mode."""
import asyncio
from typing import Any, Dict, List, Optional
import aiosqlite
import orjson
from loguru import logger
from app.config import DB_PATH
from app.models.state import FillRecord, OrderRecord, PnLRecord, PositionSide, OrderSide, OrderStatus, OrderType, OrderPurpose


class Database:
    """Async SQLite Database connection manager."""

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to SQLite and initialize schema with WAL mode."""
        if self._conn is not None:
            return

        self._conn = await aiosqlite.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = aiosqlite.Row

        # Enable Write-Ahead Logging (WAL) for high performance and concurrent reads
        await self._conn.execute("PRAGMA journal_mode = WAL;")
        await self._conn.execute("PRAGMA synchronous = NORMAL;")
        await self._conn.execute("PRAGMA foreign_keys = ON;")
        await self._conn.execute("PRAGMA busy_timeout = 30000;")
        await self._conn.commit()

        await self._create_tables()
        logger.info(f"Database initialized at {self.db_path} (WAL mode enabled)")

    async def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    async def _create_tables(self) -> None:
        """Create necessary database tables if they do not exist."""
        assert self._conn is not None

        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS credentials (
                exchange TEXT PRIMARY KEY,
                api_key_encrypted TEXT NOT NULL,
                api_secret_encrypted TEXT NOT NULL,
                passphrase_encrypted TEXT,
                testnet INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                client_order_id TEXT NOT NULL,
                exchange_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                position_side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                price REAL NOT NULL,
                stop_price REAL,
                amount REAL NOT NULL,
                filled_amount REAL NOT NULL DEFAULT 0.0,
                remaining_amount REAL NOT NULL,
                status TEXT NOT NULL,
                purpose TEXT NOT NULL,
                level INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                raw_response TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_orders_client_id ON orders(client_order_id);

            CREATE TABLE IF NOT EXISTS fills (
                id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                client_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                position_side TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                quote_amount REAL NOT NULL,
                fee REAL NOT NULL DEFAULT 0.0,
                fee_currency TEXT NOT NULL DEFAULT 'USDT',
                is_maker INTEGER NOT NULL DEFAULT 1,
                timestamp REAL NOT NULL,
                realized_pnl REAL NOT NULL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
            CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills(timestamp);

            CREATE TABLE IF NOT EXISTS pnl_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                position_side TEXT NOT NULL,
                realized_pnl REAL NOT NULL,
                fee REAL NOT NULL DEFAULT 0.0,
                net_pnl REAL NOT NULL,
                timestamp REAL NOT NULL,
                note TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_pnl_timestamp ON pnl_records(timestamp);

            CREATE TABLE IF NOT EXISTS config_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                change_type TEXT NOT NULL,
                config_json TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
        """)
        await self._conn.commit()

    # ── Orders CRUD ──
    async def save_order(self, order: OrderRecord) -> None:
        """Insert or replace an order record with lock retry."""
        if self._conn is None:
            await self.connect()
        if self._conn is None:
            logger.warning("Database not connected, cannot save order.")
            return

        raw_json = orjson.dumps(order.raw_response).decode() if order.raw_response else None
        for attempt in range(5):
            try:
                async with self._lock:
                    await self._conn.execute(
                        """
                        INSERT OR REPLACE INTO orders (
                            id, client_order_id, exchange_order_id, symbol, side, position_side,
                            order_type, price, stop_price, amount, filled_amount, remaining_amount,
                            status, purpose, level, created_at, updated_at, raw_response
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            order.id,
                            order.client_order_id,
                            order.exchange_order_id,
                            order.symbol,
                            order.side.value if hasattr(order.side, 'value') else str(order.side),
                            order.position_side.value if hasattr(order.position_side, 'value') else str(order.position_side),
                            order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                            order.price,
                            order.stop_price,
                            order.amount,
                            order.filled_amount,
                            order.remaining_amount,
                            order.status.value if hasattr(order.status, 'value') else str(order.status),
                            order.purpose.value if hasattr(order.purpose, 'value') else str(order.purpose),
                            order.level,
                            order.created_at,
                            order.updated_at,
                            raw_json,
                        )
                    )
                    await self._conn.commit()
                return
            except Exception as e:
                if "locked" in str(e).lower() and attempt < 4:
                    await asyncio.sleep(0.05 * (attempt + 1))
                    continue
                logger.warning(f"Failed to save order {order.id}: {e}")
                return

    async def get_active_orders(self, symbol: Optional[str] = None) -> List[OrderRecord]:
        """Fetch active orders (NEW or PARTIALLY_FILLED)."""
        if self._conn is None:
            return []

        query = "SELECT * FROM orders WHERE status IN ('NEW', 'PARTIALLY_FILLED')"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_order(row) for row in rows]

    # ── Fills CRUD ──
    async def save_fill(self, fill: FillRecord) -> None:
        """Insert or replace a fill record with lock retry."""
        if self._conn is None:
            await self.connect()
        if self._conn is None:
            logger.warning("Database not connected, cannot save fill.")
            return

        for attempt in range(5):
            try:
                async with self._lock:
                    await self._conn.execute(
                        """
                        INSERT OR REPLACE INTO fills (
                            id, order_id, client_order_id, symbol, side, position_side,
                            price, amount, quote_amount, fee, fee_currency, is_maker,
                            timestamp, realized_pnl
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fill.id,
                            fill.order_id,
                            fill.client_order_id,
                            fill.symbol,
                            fill.side.value if hasattr(fill.side, 'value') else str(fill.side),
                            fill.position_side.value if hasattr(fill.position_side, 'value') else str(fill.position_side),
                            fill.price,
                            fill.amount,
                            fill.quote_amount,
                            fill.fee,
                            fill.fee_currency,
                            1 if fill.is_maker else 0,
                            fill.timestamp,
                            fill.realized_pnl,
                        )
                    )
                    await self._conn.commit()
                return
            except Exception as e:
                if "locked" in str(e).lower() and attempt < 4:
                    await asyncio.sleep(0.05 * (attempt + 1))
                    continue
                logger.warning(f"Failed to save fill {fill.id}: {e}")
                return

    async def get_recent_fills(self, limit: int = 100, symbol: Optional[str] = None) -> List[FillRecord]:
        """Fetch recent fills."""
        if self._conn is None:
            return []

        query = "SELECT * FROM fills"
        params = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_fill(row) for row in rows]

    # ── PnL Records CRUD ──
    async def record_pnl(self, record: PnLRecord) -> int:
        """Insert a PnL realization record with lock retry."""
        if self._conn is None:
            await self.connect()
        if self._conn is None:
            logger.warning("Database not connected, cannot record PnL.")
            return 0

        for attempt in range(5):
            try:
                async with self._lock:
                    cursor = await self._conn.execute(
                        """
                        INSERT INTO pnl_records (symbol, position_side, realized_pnl, fee, net_pnl, timestamp, note)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.symbol,
                            record.position_side.value if hasattr(record.position_side, 'value') else str(record.position_side),
                            record.realized_pnl,
                            record.fee,
                            record.net_pnl,
                            record.timestamp,
                            record.note,
                        )
                    )
                    await self._conn.commit()
                    return cursor.lastrowid or 0
            except Exception as e:
                if "locked" in str(e).lower() and attempt < 4:
                    await asyncio.sleep(0.05 * (attempt + 1))
                    continue
                logger.warning(f"Failed to record PnL for {record.symbol}: {e}")
                return 0

    async def get_pnl_summary(self, since_timestamp: float = 0.0, symbol: Optional[str] = None) -> Dict[str, float]:
        """Calculate total realized PnL, total fees, and net PnL from fills table."""
        if self._conn is None:
            return {"total_realized_pnl": 0.0, "total_fee": 0.0, "total_net_pnl": 0.0}

        query = """
            SELECT
                COALESCE(SUM(realized_pnl), 0.0) as total_realized_pnl,
                COALESCE(SUM(fee), 0.0) as total_fee
            FROM fills
            WHERE timestamp >= ?
        """
        params: List[Any] = [since_timestamp]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        async with self._conn.execute(query, tuple(params)) as cursor:
            row = await cursor.fetchone()
            if row:
                tot_pnl = float(row["total_realized_pnl"])
                tot_fee = float(row["total_fee"])
                return {
                    "total_realized_pnl": tot_pnl,
                    "total_fee": tot_fee,
                    "total_net_pnl": tot_pnl - tot_fee,
                }
            return {"total_realized_pnl": 0.0, "total_fee": 0.0, "total_net_pnl": 0.0}

    # ── Helper Converters ──
    @staticmethod
    def _row_to_order(row: aiosqlite.Row) -> OrderRecord:
        raw_resp = orjson.loads(row["raw_response"]) if row["raw_response"] else None
        return OrderRecord(
            id=row["id"],
            client_order_id=row["client_order_id"],
            exchange_order_id=row["exchange_order_id"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            position_side=PositionSide(row["position_side"]),
            order_type=OrderType(row["order_type"]),
            price=float(row["price"]),
            stop_price=float(row["stop_price"]) if row["stop_price"] is not None else None,
            amount=float(row["amount"]),
            filled_amount=float(row["filled_amount"]),
            remaining_amount=float(row["remaining_amount"]),
            status=OrderStatus(row["status"]),
            purpose=OrderPurpose(row["purpose"]),
            level=int(row["level"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            raw_response=raw_resp,
        )

    @staticmethod
    def _row_to_fill(row: aiosqlite.Row) -> FillRecord:
        return FillRecord(
            id=row["id"],
            order_id=row["order_id"],
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            position_side=PositionSide(row["position_side"]),
            price=float(row["price"]),
            amount=float(row["amount"]),
            quote_amount=float(row["quote_amount"]),
            fee=float(row["fee"]),
            fee_currency=row["fee_currency"],
            is_maker=bool(row["is_maker"]),
            timestamp=float(row["timestamp"]),
            realized_pnl=float(row["realized_pnl"]),
        )


# Global singleton instance
db = Database()

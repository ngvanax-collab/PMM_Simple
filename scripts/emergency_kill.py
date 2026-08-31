#!/usr/bin/env python3
"""
STANDALONE CLI EMERGENCY KILL-ALL SCRIPT (HEDGE MODE NATIVE)
Usage: python3 scripts/emergency_kill.py [--exchange binance] [--testnet]
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from loguru import logger
from app.core.gateway import ExchangeGateway
from app.models.state import OrderPurpose, OrderSide, OrderType, PositionSide
from app.persistence.db import db
from app.persistence.store import config_store, credential_store


async def main():
    parser = argparse.ArgumentParser(description="PMM Engine Standalone Emergency Kill-All")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name (binance/bybit)")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    args = parser.parse_args()

    logger.info("Initializing Emergency Kill CLI...")
    await db.connect()

    creds = await credential_store.load_credentials(args.exchange)
    if not creds or not creds.api_key:
        logger.error(f"No credentials found in database for exchange: {args.exchange}")
        sys.exit(1)

    if args.testnet:
        creds.testnet = True

    gateway = ExchangeGateway(creds)
    success = await gateway.initialize()
    if not success:
        logger.critical("Failed to connect to exchange or verify Hedge Mode!")
        sys.exit(1)

    pair_configs = config_store.load_all_pair_configs()
    symbols = list(pair_configs.keys())
    if not symbols:
        symbols = ["SOL/USDT:USDT", "BTC/USDT:USDT", "ETH/USDT:USDT"]

    logger.critical("================ STARTING 6-PHASE EMERGENCY KILL-ALL ================")

    # Phase 2: Cancel all open orders
    logger.critical("Phase 2: Cancelling ALL open orders across all symbols...")
    for sym in symbols:
        await gateway.cancel_all_symbol_orders(sym)

    # Phase 3: Fetch real open positions
    logger.critical("Phase 3: Fetching active hedge positions...")
    positions_to_close = []
    for sym in symbols:
        long_p, short_p = await gateway.fetch_positions_hedge(sym)
        if long_p and long_p.amount > 1e-6:
            positions_to_close.append((sym, PositionSide.LONG, long_p.amount))
        if short_p and short_p.amount > 1e-6:
            positions_to_close.append((sym, PositionSide.SHORT, short_p.amount))

    # Phase 4: Place MARKET close orders with correct positionSide
    logger.critical(f"Phase 4: Closing {len(positions_to_close)} open positions via MARKET orders...")
    for sym, pos_side, amt in positions_to_close:
        trade_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY
        logger.warning(f"Closing: {sym} | {trade_side.value} | {pos_side.value} | Qty: {amt:.4f}")
        await gateway.create_exit_order(
            symbol=sym,
            side=trade_side,
            position_side=pos_side,
            order_type=OrderType.MARKET,
            amount=amt,
            purpose=OrderPurpose.KILL_ALL_EXIT,
        )

    # Phase 5: Re-fetch confirmation & handle residual positions
    await asyncio.sleep(0.5)
    logger.critical("Phase 5: Confirming all positions are 0...")
    residuals = []
    for sym in symbols:
        long_p, short_p = await gateway.fetch_positions_hedge(sym)
        if long_p and long_p.amount > 1e-6:
            residuals.append((sym, PositionSide.LONG, long_p.amount))
        if short_p and short_p.amount > 1e-6:
            residuals.append((sym, PositionSide.SHORT, short_p.amount))

    if residuals:
        logger.warning(f"Residual positions detected ({len(residuals)}). Executing second-pass market close...")
        for sym, pos_side, amt in residuals:
            trade_side = OrderSide.SELL if pos_side == PositionSide.LONG else OrderSide.BUY
            await gateway.create_exit_order(
                symbol=sym,
                side=trade_side,
                position_side=pos_side,
                order_type=OrderType.MARKET,
                amount=amt,
                purpose=OrderPurpose.KILL_ALL_EXIT,
            )
        await asyncio.sleep(0.5)

    # Final check
    final_residuals = []
    for sym in symbols:
        long_p, short_p = await gateway.fetch_positions_hedge(sym)
        if long_p and long_p.amount > 1e-6:
            final_residuals.append((sym, PositionSide.LONG, long_p.amount))
        if short_p and short_p.amount > 1e-6:
            final_residuals.append((sym, PositionSide.SHORT, short_p.amount))

    if not final_residuals:
        logger.info("SUCCESS: All positions completely closed (100% Flat).")
    else:
        logger.critical(f"WARNING: Residual positions remaining after second pass: {final_residuals}")

    await gateway.close()
    await db.close()
    logger.info("Emergency Kill CLI execution finished.")


if __name__ == "__main__":
    asyncio.run(main())

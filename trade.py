#!/usr/bin/env python3
"""
BLKPVNTHR Quant Execution Engine

Features
--------
• No CLI arguments required
• Waits until market open (Alpaca clock)
• Opening auction protection (default 3 minutes)
• ATR risk sizing
• VWAP entry filter
• Relative volume filter
• Bracket orders (1:3 risk reward)
• Daily trade cap (2%)
• Trade logging
"""

from __future__ import annotations

import os
import time
import math
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError as e:
    raise SystemExit(
        "Missing alpaca-py. Install:\n  pip install alpaca-py pandas python-dotenv\n"
        f"Import error: {e}"
    ) from e

load_dotenv()

# -----------------------------
# TIMEZONE
# -----------------------------
NY = timezone.utc
try:
    from zoneinfo import ZoneInfo

    NY = ZoneInfo("America/New_York")
except Exception:
    pass

# -----------------------------
# CONFIG
# -----------------------------
WEIGHTS_FILE = "data_store/markowitz_executable_weights_with_cash.csv"

RISK_PER_TRADE = 0.005  # 0.5% equity risk per symbol
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5  # stop distance = ATR * multiplier
RR_RATIO = 3.0  # reward:risk  = 3:1

DAILY_CAP_PCT = 0.02  # max gross notional per day
VWAP_LOOKBACK_MINUTES = 30
REL_VOL_THRESHOLD = 1.5

OPEN_DELAY_MINUTES = 3  # opening auction protection
WAIT_FOR_MARKET_OPEN = True
WAIT_EXTRA_SECONDS_AFTER_OPEN = 10

LOG_DIR = "data_store/trade_logs"


# -----------------------------
# CLIENTS
# -----------------------------
def get_clients() -> Tuple[TradingClient, StockHistoricalDataClient]:
    key = (os.getenv("APCA_API_KEY_ID") or "").strip()
    secret = (os.getenv("APCA_API_SECRET_KEY") or "").strip()
    if not key or not secret:
        raise SystemExit(
            "Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env or environment"
        )

    trading = TradingClient(key, secret, paper=True)
    data = StockHistoricalDataClient(key, secret)
    return trading, data


# -----------------------------
# MARKET OPEN WAIT
# -----------------------------
def wait_for_market_open(trading: TradingClient) -> None:
    if not WAIT_FOR_MARKET_OPEN:
        return

    clock = trading.get_clock()

    if getattr(clock, "is_open", False):
        # already open; still apply opening auction protection
        if OPEN_DELAY_MINUTES > 0:
            time.sleep(OPEN_DELAY_MINUTES * 60)
        return

    next_open = getattr(clock, "next_open", None)
    if next_open is None:
        print("[WARN] Alpaca clock missing next_open; continuing without wait.")
        return

    print(f"[INFO] Market closed. Sleeping until {next_open.isoformat()}")

    while True:
        now = datetime.now(timezone.utc)
        if now >= next_open:
            break
        time.sleep(30)

    print("[INFO] Market open reached.")
    if WAIT_EXTRA_SECONDS_AFTER_OPEN > 0:
        time.sleep(WAIT_EXTRA_SECONDS_AFTER_OPEN)
    if OPEN_DELAY_MINUTES > 0:
        time.sleep(OPEN_DELAY_MINUTES * 60)


# -----------------------------
# LOAD TARGETS
# -----------------------------
def load_targets() -> Dict[str, float]:
    path = Path(WEIGHTS_FILE).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Weights file not found: {path}")

    df = pd.read_csv(path)

    if "symbol" in df.columns and "weight" in df.columns:
        targets = dict(
            zip(
                df["symbol"].astype(str).str.upper().str.strip(),
                df["weight"].astype(float),
            )
        )
    else:
        df2 = pd.read_csv(path, index_col=0)
        if "weight" not in df2.columns:
            raise SystemExit(
                "Weights CSV must have (symbol, weight) columns or index=symbol with 'weight'"
            )
        targets = {
            str(k).upper().strip(): float(v) for k, v in df2["weight"].to_dict().items()
        }

    targets.pop("CASH", None)
    # drop near-zero
    targets = {k: v for k, v in targets.items() if abs(v) > 1e-8}
    return targets


# -----------------------------
# ACCOUNT DATA
# -----------------------------
def get_equity(trading: TradingClient) -> float:
    return float(trading.get_account().equity)


def get_positions(trading: TradingClient) -> Dict[str, float]:
    pos: Dict[str, float] = {}
    for p in trading.get_all_positions():
        pos[str(p.symbol).upper().strip()] = float(p.qty)
    return pos


# -----------------------------
# MARKET DATA
# -----------------------------
def get_prices(data: StockHistoricalDataClient, symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    req = StockLatestTradeRequest(symbol_or_symbols=symbols)
    resp = data.get_stock_latest_trade(req)

    prices: Dict[str, float] = {}
    for s in symbols:
        t = resp.get(s)
        if t is None:
            continue
        prices[s] = float(t.price)
    return prices


def get_atr(
    data: StockHistoricalDataClient, symbol: str, period: int = ATR_PERIOD
) -> float:
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        limit=period + 1,
    )
    bars_map = data.get_stock_bars(req)
    bars = bars_map.get(symbol)
    if not bars or len(bars) < 2:
        return 0.0

    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]

    trs: List[float] = []
    for i in range(1, len(bars)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(float(tr))

    return float(sum(trs) / max(1, len(trs)))


def get_vwap(data: StockHistoricalDataClient, symbol: str) -> Optional[float]:
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        limit=VWAP_LOOKBACK_MINUTES,
    )
    bars_map = data.get_stock_bars(req)
    bars = bars_map.get(symbol)
    if not bars:
        return None

    pv = 0.0
    vol = 0.0
    for b in bars:
        typical = (float(b.high) + float(b.low) + float(b.close)) / 3.0
        pv += typical * float(b.volume)
        vol += float(b.volume)

    if vol <= 0:
        return None
    return pv / vol


def get_relative_volume(data: StockHistoricalDataClient, symbol: str) -> float:
    # last minute volume / avg of previous minutes
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        limit=30,
    )
    bars_map = data.get_stock_bars(req)
    bars = bars_map.get(symbol)
    if not bars or len(bars) < 3:
        return 1.0

    vols = [float(b.volume) for b in bars]
    last = vols[-1]
    prev = vols[:-1]
    avg_prev = sum(prev) / max(1, len(prev))
    if avg_prev <= 0:
        return 1.0
    return last / avg_prev


# -----------------------------
# SIZING + CAP
# -----------------------------
def calculate_position_size(equity: float, atr: float) -> int:
    risk_dollars = equity * float(RISK_PER_TRADE)
    stop_distance = float(atr) * float(ATR_MULTIPLIER)

    if stop_distance <= 0:
        return 0

    shares = risk_dollars / stop_distance
    return int(math.floor(shares))


# -----------------------------
# ORDER SUBMISSION
# -----------------------------
def submit_bracket(
    trading: TradingClient, symbol: str, qty: int, price: float, atr: float
):
    stop_price = float(price) - float(atr) * float(ATR_MULTIPLIER)
    take_profit = float(price) + (float(atr) * float(ATR_MULTIPLIER) * float(RR_RATIO))

    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class="bracket",
        take_profit={"limit_price": round(take_profit, 2)},
        stop_loss={"stop_price": round(stop_price, 2)},
    )
    return trading.submit_order(req)


# -----------------------------
# LOGGING
# -----------------------------
def log_trade(symbol: str, qty: int, price: float, note: str = "") -> None:
    out_dir = Path(LOG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    file = out_dir / f"trades_{datetime.now(NY).strftime('%Y%m%d')}.csv"
    write_header = not file.exists()

    with open(file, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["time_ny", "symbol", "qty", "price", "note"])
        w.writerow([datetime.now(NY).isoformat(), symbol, qty, price, note])


# -----------------------------
# MAIN
# -----------------------------
def main() -> int:
    print("BLKPVNTHR EXECUTION ENGINE")
    print("--------------------------")

    trading, data = get_clients()
    wait_for_market_open(trading)

    targets = load_targets()
    equity = get_equity(trading)

    symbols = sorted(targets.keys())
    prices = get_prices(data, symbols)

    cap_dollars = equity * DAILY_CAP_PCT
    traded_notional = 0.0

    print(f"[INFO] Equity: ${equity:,.2f} | Daily cap: ${cap_dollars:,.2f}")
    print(f"[INFO] Symbols: {len(symbols)}")

    for sym in symbols:
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue

        atr = get_atr(data, sym, ATR_PERIOD)
        if atr <= 0:
            print(f"[SKIP] {sym}: ATR unavailable")
            continue

        vwap = get_vwap(data, sym)
        if vwap is None:
            print(f"[SKIP] {sym}: VWAP unavailable")
            continue

        if px < vwap:
            print(f"[SKIP] {sym}: VWAP gate failed (px {px:.2f} < vwap {vwap:.2f})")
            continue

        rel_vol = get_relative_volume(data, sym)
        if rel_vol < REL_VOL_THRESHOLD:
            print(
                f"[SKIP] {sym}: rel vol gate failed ({rel_vol:.2f} < {REL_VOL_THRESHOLD:.2f})"
            )
            continue

        qty = calculate_position_size(equity, atr)
        if qty <= 0:
            continue

        notional = qty * px
        if traded_notional + notional > cap_dollars:
            print("[INFO] Daily cap reached. Stopping.")
            break

        print(
            f"[BUY] {sym} qty={qty} px~{px:.2f} atr={atr:.2f} vwap~{vwap:.2f} relvol={rel_vol:.2f}"
        )

        try:
            o = submit_bracket(trading, sym, qty, px, atr)
            oid = str(getattr(o, "id", "") or "")
            traded_notional += notional
            log_trade(sym, qty, px, note=f"bracket id={oid}")
        except Exception as e:
            print(f"[ERROR] {sym}: order failed: {e}")
            log_trade(sym, 0, px, note=f"ERROR: {e}")

        time.sleep(0.2)

    print("[DONE] Execution complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

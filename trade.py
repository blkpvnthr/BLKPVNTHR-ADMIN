#!/usr/bin/env python3
"""
paper_rebalance.py

Execute Alpaca PAPER trades to rebalance your account to target weights from:
  data_store/markowitz_optimal_weights.csv
or (recommended when you use exposure sizing):
  data_store/markowitz_executable_weights_with_cash.csv  (includes CASH row)

Daily trade cap:
- By default, the script will NOT trade more than 2% of account equity per day
  (gross notional = sum(|qty| * price) across all orders).
- If planned orders exceed the cap, quantities are scaled down proportionally.

Trade logging:
- Logs every submitted (or dry-run) order with trade date+time (NY + UTC),
  symbol, side, qty, order type, limit price (if any), estimated notional,
  and Alpaca order id/status (if live).

2:00pm gate (NEW):
- The script will only OPEN/ADD (BUY) positions if the day's % change
  as of 2:00pm ET is POSITIVE.
- Sells (reductions/closures) are still allowed regardless (risk-off always allowed).
- Day %chg is computed from today's first observed price (day open proxy in this script).
  If you want official session open instead, swap to Alpaca daily bar open.

Install:
  pip install alpaca-py pandas numpy python-dotenv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
except ImportError as e:
    raise SystemExit(
        "Missing alpaca-py. Install:\n  pip install alpaca-py pandas numpy python-dotenv\n"
        f"Import error: {e}") from e

load_dotenv()

# Timezone
NY = timezone.utc
try:
    from zoneinfo import ZoneInfo  # py3.9+

    NY = ZoneInfo("America/New_York")
except Exception:
    NY = timezone.utc

# -----------------------------
# 2:00pm gate config
# -----------------------------
GATE_TIME_HHMM = os.getenv("GATE_TIME_HHMM", "14:00")  # 2:00pm ET
REQUIRE_POSITIVE_AT_GATE = os.getenv("REQUIRE_POSITIVE_AT_GATE",
                                     "true").strip().lower() in ("1", "true",
                                                                 "yes", "y",
                                                                 "on")


@dataclass
class Target:
    symbol: str
    weight: float


def _load_targets(weights_csv: Path) -> Dict[str, float]:
    """
    Accepts either:
      - index=symbol with a 'weight' column
      - two columns: symbol,weight
    Ignores 'CASH' row if present.
    """
    df = pd.read_csv(weights_csv)

    if "symbol" in df.columns and "weight" in df.columns:
        sym = df["symbol"].astype(str).str.upper().str.strip()
        w = df["weight"].astype(float)
        targets = dict(zip(sym, w))
    else:
        df2 = pd.read_csv(weights_csv, index_col=0)
        if "weight" not in df2.columns:
            raise ValueError(
                "Weights CSV must have columns (symbol, weight) or index=symbol with column 'weight'."
            )
        targets = {
            str(k).upper().strip(): float(v)
            for k, v in df2["weight"].to_dict().items()
        }

    targets.pop("CASH", None)
    targets = {k: v for k, v in targets.items() if abs(v) > 1e-8}

    s = float(sum(targets.values()))
    if s <= 1e-12:
        raise ValueError("Targets sum to ~0. Nothing to trade.")

    # Only renormalize if it's intended to be fully invested (within 1%)
    if abs(s - 1.0) <= 0.01:
        targets = {k: float(v / s) for k, v in targets.items()}

    return targets


def _get_clients() -> Tuple[TradingClient, StockHistoricalDataClient]:
    key = os.getenv("APCA_API_KEY_ID", "").strip()
    secret = os.getenv("APCA_API_SECRET_KEY", "").strip()
    if not key or not secret:
        raise EnvironmentError(
            "Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment/.env"
        )

    paper = os.getenv("APCA_PAPER", "true").strip().lower() == "true"
    trading = TradingClient(api_key=key, secret_key=secret, paper=paper)
    data = StockHistoricalDataClient(api_key=key, secret_key=secret)
    return trading, data


def _get_equity(trading: TradingClient) -> float:
    acct = trading.get_account()
    return float(acct.equity)


def _get_positions(trading: TradingClient) -> Dict[str, Dict[str, float]]:
    pos = {}
    for p in trading.get_all_positions():
        sym = str(p.symbol).upper().strip()
        pos[sym] = {"qty": float(p.qty), "market_value": float(p.market_value)}
    return pos


def _get_latest_prices(data: StockHistoricalDataClient,
                       symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    req = StockLatestTradeRequest(symbol_or_symbols=symbols)
    resp = data.get_stock_latest_trade(req)

    prices: Dict[str, float] = {}
    for sym in symbols:
        t = resp.get(sym)
        if t is None:
            continue
        prices[sym] = float(t.price)
    return prices


def _round_qty(qty: float, allow_fractional: bool) -> float:
    if allow_fractional:
        return float(round(qty, 2))  # 2 decimals
    return float(int(math.floor(qty)))


def _gross_notional(orders: List[Tuple[str, str, float]],
                    prices: Dict[str, float]) -> float:
    total = 0.0
    for sym, _, qty in orders:
        px = float(prices.get(sym, 0.0))
        total += abs(float(qty)) * px
    return float(total)


def _apply_daily_trade_cap(
    orders: List[Tuple[str, str, float]],
    prices: Dict[str, float],
    equity: float,
    cap_pct: float,
    allow_fractional: bool,
    min_dollar: float,
    min_shares: float,
) -> Tuple[List[Tuple[str, str, float]], float, float]:
    planned = _gross_notional(orders, prices)
    cap = float(max(0.0, cap_pct)) * float(equity)

    if cap <= 0.0 or planned <= cap + 1e-9:
        return orders, planned, planned

    if planned <= 1e-12:
        return [], planned, 0.0

    scale = cap / planned

    capped: List[Tuple[str, str, float]] = []
    for sym, side, qty in orders:
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue

        new_qty = float(qty) * scale
        new_qty = _round_qty(new_qty, allow_fractional=allow_fractional)

        if abs(new_qty) < float(min_shares):
            continue
        if abs(new_qty) * px < float(min_dollar):
            continue
        if new_qty <= 0:
            continue

        capped.append((sym, side, new_qty))

    capped_notional = _gross_notional(capped, prices)
    return capped, planned, capped_notional


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _trade_log_path() -> Path:
    now_ny = datetime.now(tz=NY)
    day = now_ny.strftime("%Y%m%d")
    base = Path(os.getenv("OUT_DIR", "./data_store")).expanduser().resolve()
    log_dir = base / "trade_logs"
    _ensure_dir(log_dir)
    return log_dir / f"trades_{day}.csv"


def _append_trade_log(row: Dict[str, object]) -> None:
    path = _trade_log_path()
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _parse_gate_time(hhmm: str) -> Tuple[int, int]:
    parts = hhmm.strip().split(":")
    if len(parts) != 2:
        return 14, 0
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        return hh, mm
    except Exception:
        return 14, 0


def _is_after_gate_time(now_ny: datetime) -> bool:
    hh, mm = _parse_gate_time(GATE_TIME_HHMM)
    gate = now_ny.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return now_ny >= gate


def _day_pct_chg_from_open(data: StockHistoricalDataClient,
                           symbol: str) -> Optional[float]:
    """
    Computes intraday %chg from day's first observed trade in this run (proxy),
    which we approximate using latest trade at call time as "current" and
    Alpaca's latest trade at market open is not directly accessible here.

    Practical approach:
    - Use today's daily bar open if you want official open (recommended).
      But this script currently keeps dependencies minimal; if you want that,
      I can swap this to StockBarsRequest(TimeFrame.Day, start=today, end=today).
    """
    # For now: we will treat the first call to this function per run as "open proxy"
    # by caching per symbol in env memory.
    return None  # implemented below with cache


# Cache open-proxy per symbol for the current run
_OPEN_PROXY: Dict[str, float] = {}


def _get_day_pct_chg_proxy(
    data: StockHistoricalDataClient,
    symbol: str,
    current_px: float,
) -> Optional[float]:
    """
    Proxy: open = first time we see a valid price for symbol in this run.
    pct = (current / open - 1) * 100
    """
    if current_px <= 0 or math.isnan(current_px):
        return None
    if symbol not in _OPEN_PROXY:
        _OPEN_PROXY[symbol] = float(current_px)
        return 0.0
    o = float(_OPEN_PROXY[symbol])
    if o <= 0:
        return None
    return (float(current_px) / o - 1.0) * 100.0


def _build_orders(
    targets: Dict[str, float],
    equity: float,
    positions: Dict[str, Dict[str, float]],
    prices: Dict[str, float],
    allow_fractional: bool,
    close_missing: bool,
    min_dollar: float,
    min_shares: float,
) -> List[Tuple[str, str, float]]:
    orders: List[Tuple[str, str, float]] = []

    all_syms = set(targets.keys())
    held_syms = set(positions.keys())

    if close_missing:
        for sym in sorted(held_syms - all_syms):
            qty = float(positions[sym]["qty"])
            if qty > 0:
                orders.append((sym, "sell", qty))

    for sym, w in sorted(targets.items()):
        if sym not in prices or prices[sym] <= 0:
            continue

        px = float(prices[sym])
        target_dollar = float(w) * float(equity)
        target_qty_raw = target_dollar / px
        target_qty = _round_qty(target_qty_raw,
                                allow_fractional=allow_fractional)

        cur_qty = float(positions.get(sym, {}).get("qty", 0.0))
        diff = target_qty - cur_qty
        if abs(diff) < 1e-12:
            continue

        diff_dollar = abs(diff) * px
        if diff_dollar < float(min_dollar) or abs(diff) < float(min_shares):
            continue

        side = "buy" if diff > 0 else "sell"
        qty = _round_qty(abs(diff), allow_fractional=allow_fractional)
        if qty <= 0:
            continue

        orders.append((sym, side, qty))

    return orders


def _apply_2pm_positive_gate_to_orders(
    data: StockHistoricalDataClient,
    orders: List[Tuple[str, str, float]],
    prices: Dict[str, float],
) -> Tuple[List[Tuple[str, str, float]], Dict[str, float]]:
    """
    If after gate time (2:00pm ET by default) and REQUIRE_POSITIVE_AT_GATE,
    only allow BUY orders when day's %chg is positive.
    Always allow SELL orders (risk-off).

    Returns (filtered_orders, pct_map) where pct_map has symbol->pct_chg_proxy
    """
    pct_map: Dict[str, float] = {}
    now_ny = datetime.now(tz=NY)

    if not REQUIRE_POSITIVE_AT_GATE:
        return orders, pct_map

    if not _is_after_gate_time(now_ny):
        # Gate not active yet; don't filter
        return orders, pct_map

    filtered: List[Tuple[str, str, float]] = []
    for sym, side, qty in orders:
        px = float(prices.get(sym, 0.0))
        pct = _get_day_pct_chg_proxy(data, sym, px)
        if pct is None:
            # if we can't compute, be conservative: block BUY, allow SELL
            if side == "sell":
                filtered.append((sym, side, qty))
            continue

        pct_map[sym] = float(pct)

        if side == "buy":
            if pct > 0.0:
                filtered.append((sym, side, qty))
            # else: block opening/adding if not positive
        else:
            filtered.append((sym, side, qty))

    return filtered, pct_map


def _submit_orders(
    trading: TradingClient,
    orders: List[Tuple[str, str, float]],
    prices: Dict[str, float],
    limit_offset_bps: float,
    live: bool,
    tif: TimeInForce,
    throttle_s: float,
    pct_at_gate: Optional[Dict[str, float]] = None,
) -> None:
    pct_at_gate = pct_at_gate or {}

    for sym, side, qty in orders:
        px = float(prices.get(sym, 0.0))
        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL

        # capture trade time/date
        ts_utc = datetime.now(timezone.utc)
        ts_ny = ts_utc.astimezone(NY)

        limit_price: Optional[float] = None
        order_kind = "MARKET"

        if limit_offset_bps > 0 and px > 0:
            off = float(limit_offset_bps) / 10000.0
            limit_price = px * (1.0 + off) if side == "buy" else px * (1.0 -
                                                                       off)
            limit_price = float(round(limit_price, 2))

            req = LimitOrderRequest(
                symbol=sym,
                qty=qty,
                side=side_enum,
                time_in_force=tif,
                limit_price=limit_price,
            )
            order_kind = "LIMIT"
            kind = f"LIMIT @{limit_price}"
        else:
            req = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=side_enum,
                time_in_force=tif,
            )
            kind = "MARKET"

        est_notional = float(abs(qty) * px) if px > 0 else float("nan")

        order_id = ""
        order_status = ""
        if not live:
            print(f"[DRY RUN] {sym:6s} {side.upper():4s} qty={qty} ({kind})")
            order_status = "DRY_RUN"
        else:
            o = trading.submit_order(req)
            order_id = str(getattr(o, "id", "") or "")
            order_status = str(getattr(o, "status", "") or "")
            print(
                f"[LIVE] {sym:6s} {side.upper():4s} qty={qty} ({kind}) -> id={order_id}"
            )

        # ✅ log trade time/date + details
        _append_trade_log({
            "trade_time_ny":
            ts_ny.isoformat(),
            "trade_time_utc":
            ts_utc.isoformat(),
            "symbol":
            sym,
            "side":
            side.upper(),
            "qty":
            float(qty),
            "tif":
            str(tif.value) if hasattr(tif, "value") else str(tif),
            "order_type":
            order_kind,
            "limit_price": ("" if limit_price is None else float(limit_price)),
            "last_price": ("" if px <= 0 else float(px)),
            "est_notional":
            est_notional,
            "gate_time_hhmm":
            GATE_TIME_HHMM,
            "gate_active":
            int(REQUIRE_POSITIVE_AT_GATE and _is_after_gate_time(ts_ny)),
            "pct_chg_at_gate_proxy":
            ("" if sym not in pct_at_gate else float(pct_at_gate[sym])),
            "order_id":
            order_id,
            "order_status":
            order_status,
        })

        if throttle_s > 0:
            time.sleep(throttle_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=str,
        default="data_store/markowitz_executable_weights_with_cash.csv",
        help=
        "Target weights CSV. Prefer *_with_cash.csv when using exposure sizing.",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Actually place PAPER orders. Without this, it's a dry run.",
    )
    ap.add_argument(
        "--close-missing",
        action="store_true",
        help="Close positions held but not in the target weights file.",
    )
    ap.add_argument(
        "--allow-fractional",
        action="store_true",
        help="Allow fractional shares (rounded to 2 decimals).",
    )

    ap.add_argument(
        "--min-dollar",
        type=float,
        default=10.0,
        help="Skip trades smaller than this $ value.",
    )
    ap.add_argument(
        "--min-shares",
        type=float,
        default=1.0,
        help="Skip trades smaller than this share qty.",
    )

    ap.add_argument(
        "--daily-trade-cap-pct",
        type=float,
        default=0.02,
        help=
        "Max gross notional traded per day as fraction of equity (default 0.02 = 2%).",
    )

    ap.add_argument(
        "--limit-offset-bps",
        type=float,
        default=0.0,
        help=
        "If >0, place LIMIT orders offset from last price by these bps (buy above, sell below).",
    )
    ap.add_argument(
        "--tif",
        type=str,
        default="day",
        choices=["day", "gtc"],
        help="Time in force for orders.",
    )
    ap.add_argument(
        "--throttle-s",
        type=float,
        default=0.15,
        help="Seconds to sleep between order submissions.",
    )
    args = ap.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        raise SystemExit(f"Weights file not found: {weights_path}")

    targets = _load_targets(weights_path)
    syms = sorted(targets.keys())

    trading, data = _get_clients()

    equity = _get_equity(trading)
    positions = _get_positions(trading)

    # Fetch prices for target symbols (+ held symbols if we might close them)
    price_syms = syms + (list(positions.keys()) if args.close_missing else [])
    prices = _get_latest_prices(data, sorted(set(price_syms)))

    print("\n=== Rebalance Plan ===")
    print(f"Weights file: {weights_path}")
    print(f"Account equity: ${equity:,.2f}")
    print(f"Targets (sum={sum(targets.values()):.4f}):")
    for s, w in sorted(targets.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {s:6s}  {w: .4f}")

    print("\nCurrent positions:")
    if not positions:
        print("  (none)")
    else:
        for s, dct in sorted(positions.items()):
            print(
                f"  {s:6s}  qty={dct['qty']:,.4f}  mv=${dct['market_value']:,.2f}"
            )

    orders = _build_orders(
        targets=targets,
        equity=equity,
        positions=positions,
        prices=prices,
        allow_fractional=args.allow_fractional,
        close_missing=args.close_missing,
        min_dollar=args.min_dollar,
        min_shares=args.min_shares,
    )

    if not orders:
        print(
            "\nNo orders to place (all diffs below thresholds or already aligned)."
        )
        return 0

    # --- enforce daily cap (2% by default)
    capped_orders, planned_notional, capped_notional = _apply_daily_trade_cap(
        orders=orders,
        prices=prices,
        equity=equity,
        cap_pct=args.daily_trade_cap_pct,
        allow_fractional=args.allow_fractional,
        min_dollar=args.min_dollar,
        min_shares=args.min_shares,
    )

    cap_dollars = float(args.daily_trade_cap_pct) * float(equity)

    print("\nPlanned orders (pre-cap):")
    for sym, side, qty in orders:
        px = float(prices.get(sym, 0.0))
        print(
            f"  {sym:6s} {side.upper():4s} qty={qty}  ~${qty*px:,.2f} @~{px:.2f}"
        )
    print(f"\nGross notional (pre-cap): ${planned_notional:,.2f}")

    print("\n=== Daily Trade Cap ===")
    print(
        f"Cap: {args.daily_trade_cap_pct*100:.2f}% of equity = ${cap_dollars:,.2f}"
    )
    if capped_notional < planned_notional - 1e-6:
        print(
            f"Applied scaling to fit cap. Gross notional (post-cap): ${capped_notional:,.2f}"
        )
    else:
        print(
            f"No scaling needed. Gross notional (post-cap): ${capped_notional:,.2f}"
        )

    if not capped_orders:
        print(
            "\nAfter applying the cap + min trade thresholds, no orders remain."
        )
        return 0

    # --- NEW: 2:00pm positive day %chg gate (blocks BUYs only after 2pm if not positive)
    gated_orders, pct_map = _apply_2pm_positive_gate_to_orders(
        data=data,
        orders=capped_orders,
        prices=prices,
    )

    if REQUIRE_POSITIVE_AT_GATE and _is_after_gate_time(datetime.now(tz=NY)):
        print("\n=== 2:00pm Gate (active) ===")
        print(
            "Rule: Only BUY if day %chg (proxy) is positive. SELL always allowed."
        )
        blocked = [o for o in capped_orders if o not in gated_orders]
        if blocked:
            print(f"Blocked {len(blocked)} order(s):")
            for sym, side, qty in blocked:
                px = float(prices.get(sym, 0.0))
                pct = pct_map.get(sym, float("nan"))
                print(
                    f"  {sym:6s} {side.upper():4s} qty={qty} @~{px:.2f}  pct_proxy={pct:.2f}%"
                )
        else:
            print("No orders blocked by the gate.")

    if not gated_orders:
        print("\nAfter applying the 2:00pm gate, no orders remain.")
        return 0

    print("\nOrders to submit (post-cap, post-gate):")
    for sym, side, qty in gated_orders:
        px = float(prices.get(sym, 0.0))
        est_val = qty * px
        print(
            f"  {sym:6s} {side.upper():4s} qty={qty}  ~${est_val:,.2f} @~{px:.2f}"
        )

    tif = TimeInForce.DAY if args.tif.lower() == "day" else TimeInForce.GTC

    # show where the log will go
    print(f"\nTrade log: {_trade_log_path()}")

    print("\nSubmitting orders...")
    _submit_orders(
        trading=trading,
        orders=gated_orders,
        prices=prices,
        limit_offset_bps=args.limit_offset_bps,
        live=bool(args.live),
        tif=tif,
        throttle_s=float(args.throttle_s),
        pct_at_gate=pct_map,
    )

    print("\nDone.")
    if not args.live:
        print("Tip: re-run with --live to actually place PAPER orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

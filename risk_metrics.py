#!/usr/bin/env python3
"""
risk_metrics.py

Reads the latest "session_state_YYYYMMDD.csv" produced by monitor.py and finds
symbols whose latest state == CONFIRMED. For each confirmed symbol, fetches
~1 year of daily bars from Alpaca and computes:

- Expected annual return (arithmetic, annualized): mean(daily_returns) * 252
- Annual variance (annualized): var(daily_returns) * 252
- Annual volatility: sqrt(annual variance)

Outputs a CSV by default and prints a compact table.

Usage:
  python risk_metrics.py
  python risk_metrics.py --out-dir ./data_store
  python risk_metrics.py --lookback-days 365 --output metrics.csv
  python risk_metrics.py --symbols AAPL,MSFT,SPY

Env (same as monitor.py):
  APCA_API_KEY_ID
  APCA_API_SECRET_KEY
"""

from __future__ import annotations

import os
import sys
import math
import csv
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()


# Fetch variables from environment

# -----------------------------
# Config
# -----------------------------
load_dotenv()

APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_PAPER = os.getenv("APCA_PAPER", "true").lower() == "true"

if not APCA_API_KEY_ID:
    raise ValueError("Missing APCA_API_KEY_ID in environment / .env")
if not APCA_API_SECRET_KEY:
    raise ValueError("Missing APCA_API_SECRET_KEY in environment / .env")

try:
    import pandas as pd  # type: ignore
    from alpaca.data.historical import (  # type: ignore[import-untyped]
        StockHistoricalDataClient
    )
    from alpaca.data.requests import StockBarsRequest  # type: ignore[import-untyped]
    from alpaca.data.timeframe import TimeFrame  # type: ignore[import-untyped]
except ImportError as e:
    print(f"ERROR: Missing required package: {e}", file=sys.stderr)
    print("Install with: pip install pandas alpaca-py", file=sys.stderr)
    sys.exit(1)


@dataclass
class MetricRow:
    """Represents a single row of risk metrics for a symbol."""

    symbol: str
    n_days: int
    start: str
    end: str
    exp_return_annual: float
    var_annual: float
    vol_annual: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute yearly expected return & variance for "
            "CONFIRMED symbols."
        )
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get("OUT_DIR", "./data_store"),
        help="Same OUT_DIR used by monitor.py"
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="How many calendar days to look back for daily bars",
    )
    p.add_argument(
        "--risk-free",
        type=float,
        default=0.0,
        help="Optional risk-free rate for excess return (annual)."
    )
    p.add_argument(
        "--output",
        default="",
        help=(
            "Write results to CSV at this path (default: "
            "<out-dir>/metrics/yearly_metrics_YYYYMMDD.csv)"
        ),
    )
    p.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols; bypasses session_state parsing"
    )
    p.add_argument(
        "--require-confirmed",
        action="store_true",
        help=(
            "If set, symbols passed via --symbols are filtered by "
            "CONFIRMED state (if available)."
        ),
    )
    p.add_argument(
        "--trading-days",
        type=int,
        default=252,
        help="Annualization factor, default 252",
    )
    return p.parse_args()


def _latest_session_state_csv(out_dir: str) -> Optional[Path]:
    ss_dir = Path(out_dir) / "session_state"
    if not ss_dir.exists():
        return None
    files = sorted(ss_dir.glob("session_state_*.csv"))
    if not files:
        return None
    # Choose by mtime, not lexicographic, so it works across days.
    return max(files, key=lambda p: p.stat().st_mtime)


def _load_latest_states(session_state_csv: Path) -> Dict[str, str]:
    """
    Returns {symbol: latest_state}.
    session_state CSV rows are time-ordered by append, but we sort by time just in case.
    """
    df = pd.read_csv(session_state_csv)
    if df.empty:
        return {}
    for col in ("time", "symbol", "state"):
        if col not in df.columns:
            msg = f"Missing column '{col}' in {session_state_csv}"
            raise ValueError(msg)

    # Ensure latest by time per symbol
    df["time"] = pd.to_datetime(
        df["time"], errors="coerce", utc=False
    )
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    latest = df.groupby("symbol")["state"].last().to_dict()
    # normalize
    return {
        str(sym).upper(): str(state).upper()
        for sym, state in latest.items()
    }


def _confirmed_symbols(out_dir: str) -> Tuple[List[str], Optional[Path]]:
    session_state_csv = _latest_session_state_csv(out_dir)
    if session_state_csv is None:
        return ([], None)
    latest_states = _load_latest_states(session_state_csv)
    confirmed = sorted(
        [s for s, st in latest_states.items() if st == "CONFIRMED"]
    )
    return (confirmed, session_state_csv)


def _fetch_daily_closes(
    client: StockHistoricalDataClient,
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.Series:
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_utc,
        end=end_utc,
        adjustment="all",
    )
    bars = client.get_stock_bars(req)
    # Alpaca SDK returns a BarSet; .df gives a multi-index DF (symbol, timestamp)
    df = bars.df
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Normalize index -> timestamp, and extract close series
    if isinstance(df.index, pd.MultiIndex):
        # level 0 = symbol, level 1 = timestamp
        df = df.xs(symbol, level=0, drop_level=True)

    if "close" not in df.columns:
        return pd.Series(dtype=float)

    s = df["close"].astype(float).dropna().sort_index()
    return s


def _annual_metrics_from_closes(
    closes: pd.Series,
    trading_days: int,
    risk_free_annual: float = 0.0,
) -> Tuple[float, float, float, int, str, str]:
    """
    Uses arithmetic daily returns: r_t = close_t/close_{t-1} - 1
    Annualize: mean * trading_days; var * trading_days
    """
    if closes is None or closes.empty or len(closes) < 3:
        return (float("nan"), float("nan"), float("nan"), 0, "", "")

    daily = closes.pct_change().dropna()
    if daily.empty:
        return (float("nan"), float("nan"), float("nan"), 0, "", "")

    mu_d = float(daily.mean())
    var_d = float(daily.var(ddof=1))  # sample variance
    mu_a = mu_d * trading_days
    var_a = var_d * trading_days
    vol_a = (
        math.sqrt(var_a)
        if var_a >= 0 and not math.isnan(var_a)
        else float("nan")
    )

    # Excess return (optional)
    if risk_free_annual:
        mu_a = mu_a - float(risk_free_annual)

    start = str(pd.to_datetime(daily.index.min()).date())
    end = str(pd.to_datetime(daily.index.max()).date())
    return (mu_a, var_a, vol_a, int(daily.shape[0]), start, end)


def _default_output_path(out_dir: str) -> Path:
    d = datetime.now().strftime("%Y%m%d")
    outp = Path(out_dir) / "metrics"
    outp.mkdir(parents=True, exist_ok=True)
    return outp / f"yearly_metrics_{d}.csv"


def _print_table(rows: List[MetricRow]) -> None:
    """Print a formatted table of risk metrics."""
    if not rows:
        print("No rows to display.")
        return
    header = (
        f"{'SYMBOL':<8} {'N':>6} {'START':<10} {'END':<10} "
        f"{'E[RET]':>12} {'VAR':>12} {'VOL':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.symbol:<8} {r.n_days:>6} {r.start:<10} {r.end:<10} "
            f"{r.exp_return_annual:>12.6f} {r.var_annual:>12.6f} "
            f"{r.vol_annual:>10.6f}"
        )


def main() -> int:
    args = _parse_args()

    api_key = os.environ.get("APCA_API_KEY_ID", "").strip()
    api_secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    if not api_key or not api_secret:
        print(
            "ERROR: Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment.",
            file=sys.stderr,
        )
        return 2

    # Determine symbols
    symbols: List[str] = []
    session_state_csv: Optional[Path] = None

    if args.symbols.strip():
        symbols = [
            s.strip().upper() for s in args.symbols.split(",") if s.strip()
        ]
        if args.require_confirmed:
            # If session_state exists, filter to confirmed only
            confirmed, session_state_csv = _confirmed_symbols(args.out_dir)
            symbols = [s for s in symbols if s in set(confirmed)]
    else:
        symbols, session_state_csv = _confirmed_symbols(args.out_dir)

    if not symbols:
        hint = ""
        if session_state_csv is None:
            ss_path = Path(args.out_dir) / 'session_state'
            hint = f" (no session_state files found under: {ss_path})"
        print(f"No CONFIRMED symbols found{hint}.", file=sys.stderr)
        print(
            "Tip: run monitor.py long enough to reach 30m "
            "confirmation, or pass --symbols.",
            file=sys.stderr,
        )
        return 1

    client = StockHistoricalDataClient(api_key, api_secret)

    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=int(args.lookback_days))

    rows: List[MetricRow] = []
    for sym in symbols:
        closes = _fetch_daily_closes(
            client, sym, start_utc=start_utc, end_utc=end_utc
        )
        mu_a, var_a, vol_a, n, start, end = _annual_metrics_from_closes(
            closes,
            trading_days=int(args.trading_days),
            risk_free_annual=float(args.risk_free),
        )
        rows.append(
            MetricRow(
                symbol=sym,
                n_days=n,
                start=start,
                end=end,
                exp_return_annual=mu_a,
                var_annual=var_a,
                vol_annual=vol_a,
            )
        )

    # Print table
    _print_table(rows)

    # Write CSV
    output_path = (
        Path(args.output) if args.output.strip()
        else _default_output_path(args.out_dir)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "symbol",
                "n_days",
                "start",
                "end",
                "exp_return_annual",
                "var_annual",
                "vol_annual",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.symbol,
                    r.n_days,
                    r.start,
                    r.end,
                    r.exp_return_annual,
                    r.var_annual,
                    r.vol_annual,
                ]
            )

    print(f"\nWrote {len(rows)} rows to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

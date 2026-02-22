#!/usr/bin/env python3
"""
portfolio_estimators.py

- Reads latest session_state_YYYYMMDD.csv produced by monitor.py
  (OUT_DIR/session_state/)
- Filters symbols whose latest state == "CONFIRMED"
- Fetches ~1 year of DAILY bars from Alpaca
- Computes daily close-to-close returns for each symbol
- Estimates:
    * mu_daily: expected returns vector (sample mean of daily returns)
    * Sigma_daily: variance-covariance matrix
      (sample covariance of daily returns)
- Annualizes:
    * mu_ann = mu_daily * 252
    * Sigma_ann = Sigma_daily * 252

Robust / shrinkage tactics:
- Uses confidence-ellipsoid implied penalty to shrink the mean toward 0:
    gamma = sqrt( chi2_ppf(conf, d) / n )
    S_hat = sqrt( mu^T Sigma^{-1} mu )   (estimated max Sharpe, rf=0)
    alpha = max(0, 1 - gamma / S_hat)
    mu_shrunk = alpha * mu

- Computes tangency-style weights with budget normalization:
    w_raw = Sigma^{-1} mu_shrunk
    w = w_raw / sum(w_raw)

Optional:
- Save 95% confidence ellipse plot for annualized expected returns (2D or PCA-projected)

Usage:
  python portfolio_estimators.py
  python portfolio_estimators.py --out-dir ./data_store
    --lookback-days 365
  python portfolio_estimators.py --symbols AAPL,MSFT,SPY
  python portfolio_estimators.py --robust --confidence 0.95
  python portfolio_estimators.py --robust --long-only
  python portfolio_estimators.py --plot-ellipse --ellipse-symbols AAPL,MSFT

Env:
  APCA_API_KEY_ID
  APCA_API_SECRET_KEY
  ALPACA_DATA_FEED (optional: IEX or SIP; default IEX)
  OUT_DIR (optional default for --out-dir)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import pi
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
except ImportError as e:
    msg = (
        "Missing alpaca-py dependency. Install into your venv:\n"
        "  pip install alpaca-py pandas numpy matplotlib\n"
        f"\nImport error: {e}"
    )
    raise SystemExit(msg) from e

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Estimators:
    symbols: List[str]
    returns: pd.DataFrame  # index: date, columns: symbol (daily returns)
    mu_daily: pd.Series  # index: symbol
    cov_daily: pd.DataFrame  # index/cols: symbol
    mu_annual: pd.Series
    cov_annual: pd.DataFrame


def _parse_feed() -> DataFeed:
    raw = os.environ.get("ALPACA_DATA_FEED", "IEX").strip().upper()
    return DataFeed.SIP if raw == "SIP" else DataFeed.IEX


def _latest_session_state_csv(out_dir: Path) -> Path:
    base = out_dir / "session_state"
    if not base.exists():
        raise FileNotFoundError(f"session_state folder not found: {base}")

    files = sorted(base.glob("session_state_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No session_state_*.csv files found in: {base}")
    return files[-1]


def _read_confirmed_symbols_from_session_state(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    if "time" in df.columns:
        df = df.sort_values("time")

    if "symbol" not in df.columns or "state" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain columns 'symbol' and 'state'")

    latest = df.groupby("symbol", as_index=False).tail(1)
    confirmed = latest.loc[latest["state"].astype(str).str.upper() ==
                           "CONFIRMED", "symbol"]
    syms = sorted({
        s.strip().upper()
        for s in confirmed.astype(str).tolist() if s.strip()
    })
    return syms


def _get_alpaca_client() -> StockHistoricalDataClient:
    key = os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    if not key or not secret:
        raise EnvironmentError(
            "Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment.")
    return StockHistoricalDataClient(api_key=key, secret_key=secret)


def _fetch_daily_closes(
    client: StockHistoricalDataClient,
    symbols: List[str],
    start: datetime,
    end: datetime,
    feed: DataFeed,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=feed,
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.reset_index()
    if ("symbol" not in df.columns or "timestamp" not in df.columns
            or "close" not in df.columns):
        raise ValueError("Unexpected bars dataframe schema from Alpaca.")

    df["date"] = (pd.to_datetime(
        df["timestamp"]).dt.tz_convert("America/New_York").dt.date)
    close_panel = (df.pivot_table(
        index="date", columns="symbol", values="close",
        aggfunc="last").sort_index().sort_index(axis=1))
    close_panel = close_panel[[s for s in symbols if s in close_panel.columns]]
    return close_panel


def _compute_returns_from_closes(closes: pd.DataFrame) -> pd.DataFrame:
    if closes.empty:
        return pd.DataFrame()
    rets = closes.pct_change().dropna(how="all")
    return rets


def estimate_portfolio(
    out_dir: Path,
    lookback_days: int = 365,
    symbols_override: Optional[List[str]] = None,
    min_overlap: int = 120,
) -> Estimators:
    if symbols_override:
        symbols = sorted(
            {s.strip().upper()
             for s in symbols_override if s.strip()})
    else:
        ss_path = _latest_session_state_csv(out_dir)
        symbols = _read_confirmed_symbols_from_session_state(ss_path)

    if not symbols:
        raise RuntimeError(
            "No symbols selected (no CONFIRMED symbols found, and no --symbols override provided)."
        )

    client = _get_alpaca_client()
    feed = _parse_feed()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days) + 7)

    closes = _fetch_daily_closes(client,
                                 symbols,
                                 start=start,
                                 end=end,
                                 feed=feed)
    returns = _compute_returns_from_closes(closes)

    if returns.empty:
        raise RuntimeError(
            "No returns computed (empty close series). Check symbols or Alpaca data access/feed."
        )

    # require enough data per symbol
    counts = returns.notna().sum(axis=0).sort_values(ascending=False)
    keep = counts[counts >= int(min_overlap)].index.tolist()
    if not keep:
        msg = (
            f"No symbols have at least min_overlap={min_overlap} "
            f"daily return observations.\nCounts:\n{counts.to_string()}"
        )
        raise RuntimeError(msg)

    # inner-join dates across kept symbols for a single common sample
    returns = returns[keep].dropna(how="any")

    if len(returns) < int(min_overlap):
        msg = (
            f"After aligning (dropping any date with missing values), "
            f"only {len(returns)} rows remain.\n"
            f"Try lowering --min-overlap."
        )
        raise RuntimeError(msg)

    mu_daily = returns.mean(axis=0)
    cov_daily = returns.cov()

    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR
    cov_annual = cov_daily * TRADING_DAYS_PER_YEAR

    return Estimators(
        symbols=keep,
        returns=returns,
        mu_daily=mu_daily,
        cov_daily=cov_daily,
        mu_annual=mu_annual,
        cov_annual=cov_annual,
    )


def _save_estimators(est: Estimators, out_dir: Path,
                     prefix: str) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {
        "returns": out_dir / f"{prefix}_returns.csv",
        "mu_daily": out_dir / f"{prefix}_mu_daily.csv",
        "mu_annual": out_dir / f"{prefix}_mu_annual.csv",
        "cov_daily": out_dir / f"{prefix}_cov_daily.csv",
        "cov_annual": out_dir / f"{prefix}_cov_annual.csv",
    }

    est.returns.to_csv(paths["returns"], index=True)
    est.mu_daily.to_csv(paths["mu_daily"], index=True, header=["mu_daily"])
    est.mu_annual.to_csv(paths["mu_annual"], index=True, header=["mu_annual"])
    est.cov_daily.to_csv(paths["cov_daily"], index=True)
    est.cov_annual.to_csv(paths["cov_annual"], index=True)

    return paths


# ---------------------------
# Robust / Shrinkage tactics
# ---------------------------


def _chi2_ppf(conf: float, df: int) -> float:
    """
    Chi-square quantile. Uses scipy if available, else fallback:
    - df=2 is exact constant fallback
    - other df: uses Wilson-Hilferty approximation (good for df>=2)
    """
    if conf <= 0.0 or conf >= 1.0:
        raise ValueError("--confidence must be between 0 and 1 (exclusive).")

    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.ppf(conf, df=df))
    except Exception:
        if df == 2:
            # 95% is 5.991..., but we want general conf for df=2:
            # closed form: chi2(df=2) is exponential with mean 2:
            # CDF = 1 - exp(-x/2) => x = -2 ln(1-conf)
            return float(-2.0 * np.log(1.0 - conf))

        # Wilson–Hilferty approximation:
        # If X~chi2_k, then (X/k)^(1/3) approx Normal(1-2/(9k),
        # 2/(9k)) so X ≈ k * (1 - 2/(9k) + z*sqrt(2/(9k)))^3
        z = float(_norm_ppf(conf))
        k = float(df)
        term = 1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))
        return float(k * term**3)


def _norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF. Uses scipy if available; else rational approximation.
    """
    try:
        from scipy.stats import norm  # type: ignore
        return float(norm.ppf(p))
    except ImportError:
        # Peter John Acklam's approximation (good enough for this use)
        # https://web.archive.org/web/20150910044729/http://home.online.no/~pjacklam/notes/invnorm/
        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]

        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = np.sqrt(-2 * np.log(p))
            return (
                ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
                c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        if p > phigh:
            q = np.sqrt(-2 * np.log(1 - p))
            return -(
                ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
                c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
             a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def robust_shrinkage_alpha(mu: np.ndarray, Sigma: np.ndarray, n: int,
                           conf: float, ridge: float) -> float:
    """
    alpha = max(0, 1 - gamma / S_hat)
    where gamma = sqrt(chi2_{d,conf} / n), S_hat = sqrt(mu^T Sigma^{-1} mu)

    mu and Sigma should be in the SAME units (daily OR annualized).
    (Scale cancels in the ratio, so daily vs annual is fine as long as consistent.)
    """
    d = int(len(mu))
    if d < 1:
        return 0.0
    if n <= 1:
        return 0.0

    chi2_val = _chi2_ppf(conf, df=d)
    gamma = float(np.sqrt(chi2_val / n))

    # Stable inverse via ridge
    Sigma_reg = Sigma + float(ridge) * np.eye(d)
    try:
        x = np.linalg.solve(Sigma_reg, mu)
    except np.linalg.LinAlgError as exc:
        x = np.linalg.lstsq(Sigma_reg, mu, rcond=None)[0]

    S_hat = float(np.sqrt(max(mu @ x, 0.0)))  # sqrt(mu^T Sigma^{-1} mu)
    if S_hat <= 1e-12:
        return 0.0

    alpha = max(0.0, 1.0 - gamma / S_hat)
    return float(alpha)


def tangency_weights_budget(mu: np.ndarray, Sigma: np.ndarray, ridge: float,
                            long_only: bool) -> np.ndarray:
    """
    Compute w ∝ Sigma^{-1} mu, then normalize to sum(w)=1.

    If long_only=True: clip negatives to 0 and renormalize (simple heuristic).
    """
    d = int(len(mu))
    Sigma_reg = Sigma + float(ridge) * np.eye(d)

    try:
        w = np.linalg.solve(Sigma_reg, mu)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Sigma_reg, mu, rcond=None)[0]

    if long_only:
        w = np.maximum(w, 0.0)

    s = float(np.sum(w))
    if abs(s) <= 1e-12:
        # fallback: equal weight
        return np.ones(d) / d
    return w / s


def compute_robust_portfolio(est: Estimators, conf: float, ridge: float,
                             long_only: bool) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      symbol, mu_daily, mu_shrunk_daily, weight_robust, weight_plain
    """
    mu = est.mu_daily.values.astype(float)
    Sigma = est.cov_daily.values.astype(float)
    n = int(len(est.returns))

    alpha = robust_shrinkage_alpha(mu=mu,
                                   Sigma=Sigma,
                                   n=n,
                                   conf=conf,
                                   ridge=ridge)
    mu_shrunk = alpha * mu

    w_plain = tangency_weights_budget(mu=mu,
                                      Sigma=Sigma,
                                      ridge=ridge,
                                      long_only=long_only)
    w_robust = tangency_weights_budget(mu=mu_shrunk,
                                       Sigma=Sigma,
                                       ridge=ridge,
                                       long_only=long_only)

    out = pd.DataFrame({
        "symbol": est.symbols,
        "mu_daily": mu,
        "mu_shrunk_daily": mu_shrunk,
        "weight_plain": w_plain,
        "weight_robust": w_robust,
    }).set_index("symbol")

    # also include annualized means for readability
    out["mu_annual"] = est.mu_annual.values.astype(float)
    out["mu_shrunk_annual"] = alpha * est.mu_annual.values.astype(float)

    out.attrs["alpha"] = alpha
    out.attrs["n"] = n
    out.attrs["conf"] = conf
    return out


# ---------------------------
# Confidence ellipse plotting
# ---------------------------


def _pca_project_2d(S: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(vals)[::-1]
    top2 = vecs[:, idx[:2]]
    q, _ = np.linalg.qr(top2)
    return q[:, :2]


def plot_mu_confidence_ellipse_95(
    est: Estimators,
    save_path: Path,
    conf: float,
    symbols_2d: Optional[List[str]] = None,
) -> Path:
    """
    Plot a confidence ellipse for the ANNUALIZED expected return vector in 2D.

    - If symbols_2d provided (length 2): plot in that 2-symbol plane.
    - Else if len(est.symbols)==2: plot in that plane.
    - Else: project to 2D using PCA on cov_daily.

    Sampling covariance of annualized sample mean:
      Cov(mu_hat_annual) = (252^2) * Sigma_daily / n
    Ellipse is defined by chi2(df=2, conf).
    """
    n = int(len(est.returns))
    if n <= 2:
        raise RuntimeError(
            "Not enough return observations to form a confidence ellipse.")

    mu_ann = est.mu_annual.values.astype(float)  # (d,)
    Sigma_daily = est.cov_daily.values.astype(float)  # (d,d)
    Cov_mu_ann = (TRADING_DAYS_PER_YEAR**2) * Sigma_daily / n  # (d,d)

    d = len(est.symbols)
    sym_index = {s: i for i, s in enumerate(est.symbols)}

    if symbols_2d is not None:
        if len(symbols_2d) != 2:
            raise ValueError(
                "--ellipse-symbols must contain exactly 2 symbols")
        a, b = symbols_2d[0].upper(), symbols_2d[1].upper()
        if a not in sym_index or b not in sym_index:
            raise ValueError(
                f"Symbols for ellipse must be among kept symbols: {est.symbols}"
            )
        idx = [sym_index[a], sym_index[b]]
        W = np.zeros((d, 2), dtype=float)
        W[idx[0], 0] = 1.0
        W[idx[1], 1] = 1.0
        plane_label = f"{a} vs {b}"
        xlab, ylab = f"μ_annual({a})", f"μ_annual({b})"
    elif d == 2:
        W = np.eye(2, dtype=float)
        plane_label = f"{est.symbols[0]} vs {est.symbols[1]}"
        xlab, ylab = f"μ_annual({est.symbols[0]})", f"μ_annual({est.symbols[1]})"
    else:
        W = _pca_project_2d(Sigma_daily)  # (d,2)
        plane_label = "PCA(returns covariance) plane"
        xlab, ylab = "PC1 (μ_annual projection)", "PC2 (μ_annual projection)"

    mu2 = W.T @ mu_ann
    Cov2 = W.T @ Cov_mu_ann @ W

    chi2_val = _chi2_ppf(conf, df=2)
    evals, evecs = np.linalg.eigh(Cov2)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    r1 = np.sqrt(chi2_val * max(float(evals[0]), 0.0))
    r2 = np.sqrt(chi2_val * max(float(evals[1]), 0.0))

    theta = np.linspace(0.0, 2.0 * pi, 400)
    circle = np.vstack([np.cos(theta) * r1, np.sin(theta) * r2])
    ellipse = (evecs @ circle) + mu2.reshape(2, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ellipse[0, :], ellipse[1, :], linewidth=2)
    ax.scatter([mu2[0]], [mu2[1]])
    ax.set_title(
        f"{int(conf*100)}% Confidence Ellipse for Annualized Mean Returns\n({plane_label}, n={n})"
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",
                   default=os.environ.get("OUT_DIR", "./data_store"))
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--symbols", type=str, default="")
    p.add_argument("--min-overlap", type=int, default=120)

    p.add_argument("--save-dir", type=str, default="")
    p.add_argument("--prefix", type=str, default="estimators")

    # robust tactics
    p.add_argument(
        "--robust",
        action="store_true",
        help="Compute robust shrinkage + robust tangency weights",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for uncertainty set (e.g. 0.95)",
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-8,
        help="Diagonal ridge added to covariance for stability",
    )
    p.add_argument(
        "--long-only",
        action="store_true",
        help="Force long-only via clipping negatives (heuristic)",
    )

    # ellipse plot
    p.add_argument(
        "--plot-ellipse",
        action="store_true",
        help="Save confidence ellipse plot for annualized mean returns",
    )
    p.add_argument(
        "--ellipse-symbols",
        type=str,
        default="",
        help="Two symbols comma-separated to plot in that 2D plane",
    )
    p.add_argument("--ellipse-out",
                   type=str,
                   default="",
                   help="Output path for ellipse PNG")

    args = p.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    save_dir = Path(
        args.save_dir).expanduser().resolve() if args.save_dir else out_dir

    symbols_override = None
    if args.symbols.strip():
        symbols_override = [
            s.strip().upper() for s in args.symbols.split(",") if s.strip()
        ]

    est = estimate_portfolio(
        out_dir=out_dir,
        lookback_days=args.lookback_days,
        symbols_override=symbols_override,
        min_overlap=args.min_overlap,
    )

    paths = _save_estimators(est, save_dir, prefix=args.prefix)

    print("\n=== Estimators ===")
    print(f"Symbols (kept): {', '.join(est.symbols)}")
    print(
        f"Aligned return rows: {len(est.returns)} (dates {est.returns.index.min()} → {est.returns.index.max()})"
    )
    print("\nmu (annualized):")
    print(
        est.mu_annual.sort_values(ascending=False).to_string(
            float_format=lambda x: f"{x: .4f}"))
    print("\nSaved estimator CSVs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    if args.robust:
        rob = compute_robust_portfolio(est,
                                       conf=args.confidence,
                                       ridge=args.ridge,
                                       long_only=args.long_only)
        alpha = float(rob.attrs.get("alpha", np.nan))
        n = int(rob.attrs.get("n", len(est.returns)))

        # save
        w_path = save_dir / f"{args.prefix}_weights_robust.csv"
        rob.to_csv(w_path, index=True)

        print("\n=== Robust / Shrinkage ===")
        print(
            f"confidence={args.confidence:.3f}, n={n}, ridge={args.ridge:g}, long_only={args.long_only}"
        )
        print(
            f"alpha (shrink factor) = {alpha:.6f}  (mu_shrunk = alpha * mu_hat)"
        )
        print("\nTop weights (robust):")
        print(rob["weight_robust"].sort_values(
            ascending=False).head(15).to_string(
                float_format=lambda x: f"{x: .4f}"))
        print(f"\nSaved: {w_path}")

    if args.plot_ellipse:
        symbols_2d = None
        if args.ellipse_symbols.strip():
            symbols_2d = [
                s.strip().upper() for s in args.ellipse_symbols.split(",")
                if s.strip()
            ]

        ellipse_path = (Path(
            args.ellipse_out).expanduser().resolve() if args.ellipse_out else (
                save_dir /
                f"{args.prefix}_mu_ellipse_{int(args.confidence*100)}.png"))
        plot_mu_confidence_ellipse_95(est,
                                      ellipse_path,
                                      conf=args.confidence,
                                      symbols_2d=symbols_2d)
        print(f"\nSaved ellipse plot: {ellipse_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

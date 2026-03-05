#!/usr/bin/env python3
"""
portfolio_estimators.py

This file now does ALL of the following (rf = 0 implicit everywhere):
- Reads latest session_state_YYYYMMDD.csv from OUT_DIR/session_state/
- Filters symbols whose latest state == CONFIRMED
- Fetches ~1 year of DAILY closes from Alpaca -> daily close-to-close returns
- Estimates:
    * mu_daily (sample mean)
    * Sigma_daily (Ledoit–Wolf shrinkage)
  and annualizes both for readability.

Robust / institutional tactics (optional, but fully integrated):
1) Confidence-ellipsoid mean shrink (global):
     mu_global = alpha * mu_hat
2) Per-asset uncertainty shrink (Bayesian-ish):
     mu_unc = shrink(mu_global, stderr(mu_hat))
3) Multi-alpha forecast blending:
     base (mu_unc) + momentum + short-term reversal + vol-adjusted trend
4) Online learning of alpha blend weights (persistent JSON):
     weights_{t+1} ∝ weights_t * exp(eta * score_t)
5) Regime-dependent risk aversion:
     gamma_t = base_gamma * multiplier(vol_ratio)
6) Turnover-aware Markowitz utility optimizer:
     maximize mu^T w - (gamma_t/2) w^T Σ w - λ ||w - w0||^2
     subject to sum(w)=1, long-only + caps optional
7) Confidence-weighted exposure sizing with CASH:
     exposure in [e_min, e_max] based on forecast consensus;
     execute w_exec = exposure * w*, CASH = 1 - sum(w_exec)

Also keeps:
- Optional 95% confidence ellipse plot for mean (2D or PCA).

Install:
  pip install alpaca-py pandas numpy matplotlib scikit-learn cvxpy python-dotenv

Outputs (default to data_store/):
- estimators_returns.csv, estimators_mu_daily.csv, estimators_mu_annual.csv
- estimators_cov_daily.csv, estimators_cov_annual.csv
- (if --robust) estimators_weights_robust.csv  (legacy robust weights)
- (if --markowitz) markowitz_optimal_weights.csv
- (if --exposure-sizing) markowitz_executable_weights_with_cash.csv
- alpha_weights.json (if --learn-alphas or --markowitz)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import pi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.covariance import LedoitWolf
import cvxpy as cp

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
except ImportError as e:
    msg = (
        "Missing alpaca-py dependency. Install into your venv:\n"
        "  pip install alpaca-py pandas numpy matplotlib scikit-learn cvxpy python-dotenv\n"
        f"\nImport error: {e}"
    )
    raise SystemExit(msg) from e

# Load environment variables from .env (if present)
load_dotenv()

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Estimators:
    symbols: List[str]
    returns: pd.DataFrame  # index: date, columns: symbol (daily returns)
    mu_daily: pd.Series  # index: symbol
    cov_daily: pd.DataFrame  # index/cols: symbol
    mu_annual: pd.Series
    cov_annual: pd.DataFrame


# -----------------------------
# Alpaca + session_state
# -----------------------------

def _parse_feed() -> DataFeed:
    raw = os.environ.get("ALPACA_DATA_FEED", "IEX").strip().upper()
    return DataFeed.SIP if raw == "SIP" else DataFeed.IEX


def _latest_session_state_csv(out_dir: Path) -> Path:
    base = out_dir / "session_state"
    if not base.exists():
        raise FileNotFoundError(f"session_state folder not found: {base}")

    files = sorted(base.glob("session_state_*.csv"))
    if not files:
        raise FileNotFoundError(f"No session_state_*.csv files found in: {base}")
    return files[-1]


def _read_confirmed_symbols_from_session_state(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    if "time" in df.columns:
        df = df.sort_values("time")
    if "symbol" not in df.columns or "state" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns 'symbol' and 'state'")

    latest = df.groupby("symbol", as_index=False).tail(1)
    confirmed = latest.loc[latest["state"].astype(str).str.upper() == "CONFIRMED", "symbol"]
    return sorted({s.strip().upper() for s in confirmed.astype(str).tolist() if s.strip()})


def _get_alpaca_client() -> StockHistoricalDataClient:
    key = os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    if not key or not secret:
        raise EnvironmentError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment / .env")
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
    if ("symbol" not in df.columns) or ("timestamp" not in df.columns) or ("close" not in df.columns):
        raise ValueError("Unexpected bars dataframe schema from Alpaca.")

    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York").dt.date
    close_panel = (
        df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )
    close_panel = close_panel[[s for s in symbols if s in close_panel.columns]]
    return close_panel


def _compute_returns_from_closes(closes: pd.DataFrame) -> pd.DataFrame:
    if closes.empty:
        return pd.DataFrame()
    return closes.pct_change().dropna(how="all")


def _ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf()
    lw.fit(returns.values)
    cov = lw.covariance_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def estimate_portfolio(
    out_dir: Path,
    lookback_days: int = 365,
    symbols_override: Optional[List[str]] = None,
    min_overlap: int = 120,
) -> Estimators:
    if symbols_override:
        symbols = sorted({s.strip().upper() for s in symbols_override if s.strip()})
    else:
        ss_path = _latest_session_state_csv(out_dir)
        symbols = _read_confirmed_symbols_from_session_state(ss_path)

    if not symbols:
        raise RuntimeError("No symbols selected (no CONFIRMED symbols found, and no --symbols override provided).")

    client = _get_alpaca_client()
    feed = _parse_feed()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days) + 7)

    closes = _fetch_daily_closes(client, symbols, start=start, end=end, feed=feed)
    returns = _compute_returns_from_closes(closes)

    if returns.empty:
        raise RuntimeError("No returns computed (empty close series). Check symbols or Alpaca data access/feed.")

    counts = returns.notna().sum(axis=0).sort_values(ascending=False)
    keep = counts[counts >= int(min_overlap)].index.tolist()
    if not keep:
        raise RuntimeError(
            f"No symbols have at least min_overlap={min_overlap} daily return observations.\n"
            f"Counts:\n{counts.to_string()}"
        )

    returns = returns[keep].dropna(how="any")
    if len(returns) < int(min_overlap):
        raise RuntimeError(
            f"After aligning (dropping any date with missing values), only {len(returns)} rows remain.\n"
            f"Try lowering --min-overlap."
        )

    mu_daily = returns.mean(axis=0)
    cov_daily = _ledoit_wolf_covariance(returns)

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


def _save_estimators(est: Estimators, out_dir: Path, prefix: str) -> Dict[str, Path]:
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
# Robust mean shrinkage (global)
# ---------------------------

def _chi2_ppf(conf: float, df: int) -> float:
    if conf <= 0.0 or conf >= 1.0:
        raise ValueError("--confidence must be between 0 and 1 (exclusive).")
    try:
        from scipy.stats import chi2  # type: ignore
        return float(chi2.ppf(conf, df=df))
    except Exception:
        if df == 2:
            return float(-2.0 * np.log(1.0 - conf))
        z = float(_norm_ppf(conf))
        k = float(df)
        term = 1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))
        return float(k * term**3)


def _norm_ppf(p: float) -> float:
    try:
        from scipy.stats import norm  # type: ignore
        return float(norm.ppf(p))
    except ImportError:
        # Acklam approximation
        a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
             1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
        b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
             6.680131188771972e01, -1.328068155288572e01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
             -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
             3.754408661907416e00]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = np.sqrt(-2 * np.log(p))
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        if p > phigh:
            q = np.sqrt(-2 * np.log(1 - p))
            return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        q = p - 0.5
        r = q*q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def robust_shrinkage_alpha(mu: np.ndarray, Sigma: np.ndarray, n: int, conf: float, ridge: float) -> float:
    d = int(len(mu))
    if d < 1 or n <= 1:
        return 0.0

    chi2_val = _chi2_ppf(conf, df=d)
    gamma = float(np.sqrt(chi2_val / n))

    Sigma_reg = Sigma + float(ridge) * np.eye(d)
    try:
        x = np.linalg.solve(Sigma_reg, mu)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(Sigma_reg, mu, rcond=None)[0]

    S_hat = float(np.sqrt(max(mu @ x, 0.0)))
    if S_hat <= 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - gamma / S_hat))


# ---------------------------
# Legacy robust weights (kept)
# ---------------------------

def constrained_mean_variance_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    long_only: bool,
    max_weight: float,
    risk_aversion: float,
) -> np.ndarray:
    d = int(len(mu))
    if d == 0:
        return np.array([])

    w = cp.Variable(d)
    Sigma_psd = 0.5 * (Sigma + Sigma.T)

    objective = cp.Maximize(mu @ w - 0.5 * float(risk_aversion) * cp.quad_form(w, Sigma_psd))
    constraints = [cp.sum(w) == 1.0]

    if long_only:
        constraints += [w >= 0.0, w <= float(max_weight)]
    else:
        constraints += [w >= -float(max_weight), w <= float(max_weight)]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            raise RuntimeError("cvxpy returned None weights")
        wv = np.array(w.value).reshape(-1)
        if long_only:
            wv = np.maximum(wv, 0.0)
        s = float(np.sum(wv))
        if abs(s) <= 1e-12:
            return np.ones(d) / d
        return wv / s
    except Exception:
        return np.ones(d) / d


def compute_robust_portfolio(
    est: Estimators,
    conf: float,
    ridge: float,
    long_only: bool,
    max_weight: float,
    risk_aversion: float,
) -> pd.DataFrame:
    mu = est.mu_daily.values.astype(float)
    Sigma = est.cov_daily.values.astype(float)
    n = int(len(est.returns))

    alpha = robust_shrinkage_alpha(mu=mu, Sigma=Sigma, n=n, conf=conf, ridge=ridge)
    mu_shrunk = alpha * mu

    w_plain = constrained_mean_variance_weights(mu, Sigma, long_only, max_weight, risk_aversion)
    w_robust = constrained_mean_variance_weights(mu_shrunk, Sigma, long_only, max_weight, risk_aversion)

    out = pd.DataFrame(
        {
            "symbol": est.symbols,
            "mu_daily": mu,
            "mu_shrunk_daily": mu_shrunk,
            "weight_plain": w_plain,
            "weight_robust": w_robust,
            "mu_annual": est.mu_annual.values.astype(float),
            "mu_shrunk_annual": alpha * est.mu_annual.values.astype(float),
        }
    ).set_index("symbol")

    out.attrs["alpha"] = alpha
    out.attrs["n"] = n
    out.attrs["conf"] = conf
    return out


# ---------------------------
# Uncertainty-adjusted mu (per asset)
# ---------------------------

def uncertainty_adjusted_mu(
    returns: pd.DataFrame,
    mu_daily: np.ndarray,
    shrink_strength: float = 1.5,
) -> np.ndarray:
    n = int(len(returns))
    if n <= 2:
        return mu_daily.copy()

    sigma = returns.std(axis=0).values.astype(float)
    stderr = sigma / np.sqrt(n)

    mu_adj = mu_daily.copy().astype(float)
    for i in range(len(mu_adj)):
        m = float(mu_adj[i])
        se = float(stderr[i])
        if abs(m) < 1e-12:
            mu_adj[i] = 0.0
            continue
        uncertainty_ratio = se / abs(m)
        shrink = 1.0 / (1.0 + float(shrink_strength) * uncertainty_ratio)
        mu_adj[i] = m * shrink
    return mu_adj


# ---------------------------
# Alpha forecasts + online learning
# ---------------------------

def _rolling_prod_one_plus(x: np.ndarray) -> float:
    return float(np.prod(1.0 + x) - 1.0)


def momentum_alpha(returns: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    if len(returns) < lookback:
        lookback = max(2, len(returns))
    mom = (1.0 + returns).rolling(lookback).apply(lambda x: np.prod(x), raw=True) - 1.0
    return mom.iloc[-1].values.astype(float)


def reversion_alpha(returns: pd.DataFrame, short_window: int = 5) -> np.ndarray:
    w = min(max(2, int(short_window)), len(returns))
    short_ret = returns.tail(w).mean(axis=0)
    return (-short_ret).values.astype(float)


def volatility_adjusted_alpha(returns: pd.DataFrame, window: int = 30) -> np.ndarray:
    w = min(max(5, int(window)), len(returns))
    vol = returns.rolling(w).std().iloc[-1].values.astype(float)
    trend = returns.rolling(w).mean().iloc[-1].values.astype(float)
    return trend / (vol + 1e-8)


def compute_alpha_forecasts(returns: pd.DataFrame, mu_base: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "base": np.asarray(mu_base, dtype=float),
        "momentum": momentum_alpha(returns),
        "reversion": reversion_alpha(returns),
        "vol_adj": volatility_adjusted_alpha(returns),
    }


def load_alpha_weights(path: Path) -> Dict[str, float]:
    if path.exists():
        try:
            obj = json.loads(path.read_text())
            if isinstance(obj, dict) and obj:
                return {str(k): float(v) for k, v in obj.items()}
        except Exception:
            pass
    # default equal weights
    return {"base": 0.25, "momentum": 0.25, "reversion": 0.25, "vol_adj": 0.25}


def save_alpha_weights(path: Path, weights: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: float(v) for k, v in weights.items()}, indent=2))


def blended_expected_returns_dynamic(alpha_forecasts: Dict[str, np.ndarray], alpha_weights: Dict[str, float]) -> np.ndarray:
    mu = np.zeros_like(next(iter(alpha_forecasts.values())), dtype=float)
    total = 0.0
    for k, f in alpha_forecasts.items():
        wk = float(alpha_weights.get(k, 0.0))
        if wk != 0.0:
            mu += wk * np.asarray(f, dtype=float)
            total += abs(wk)
    if total <= 1e-12:
        return np.asarray(alpha_forecasts["base"], dtype=float)
    return mu


def update_alpha_weights(
    alpha_forecasts: Dict[str, np.ndarray],
    realized_returns: np.ndarray,
    prev_weights: Dict[str, float],
    learning_rate: float = 5.0,
) -> Dict[str, float]:
    # score_k = <forecast_k, realized>
    scores: Dict[str, float] = {}
    r = np.asarray(realized_returns, dtype=float).reshape(-1)
    for name, forecast in alpha_forecasts.items():
        f = np.asarray(forecast, dtype=float).reshape(-1)
        scores[name] = float(np.dot(f, r))

    exp_scores: Dict[str, float] = {}
    for k, s in scores.items():
        w0 = float(prev_weights.get(k, 1.0))
        exp_scores[k] = w0 * float(np.exp(float(learning_rate) * s))

    total = float(sum(exp_scores.values()))
    if total <= 1e-18:
        # fallback: equal
        n = len(exp_scores) if exp_scores else 1
        return {k: 1.0 / n for k in exp_scores}

    return {k: float(v / total) for k, v in exp_scores.items()}


# ---------------------------
# Regime-dependent gamma
# ---------------------------

def compute_volatility_regime(
    returns: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 120,
) -> Tuple[str, float]:
    """
    Vol regime via realized vol ratio on equal-weight proxy market.
    """
    if returns.empty:
        return "NORMAL", 1.0

    r_mkt = returns.mean(axis=1)  # equal-weight proxy
    sw = min(max(5, int(short_window)), len(r_mkt))
    lw = min(max(sw + 1, int(long_window)), len(r_mkt))

    vol_short = r_mkt.rolling(sw).std().iloc[-1]
    vol_long = r_mkt.rolling(lw).std().iloc[-1]

    if pd.isna(vol_short) or pd.isna(vol_long) or float(vol_long) <= 1e-12:
        return "NORMAL", 1.0

    ratio = float(vol_short / vol_long)
    if ratio < 0.8:
        return "LOW_VOL", ratio
    if ratio < 1.3:
        return "NORMAL", ratio
    return "HIGH_VOL", ratio


def regime_gamma(base_gamma: float, regime: str) -> float:
    mult = {"LOW_VOL": 0.6, "NORMAL": 1.0, "HIGH_VOL": 2.5}.get(regime, 1.0)
    return float(base_gamma) * float(mult)


# ---------------------------
# Turnover-aware Markowitz utility optimizer
# ---------------------------

def _load_current_weights_csv(path: Path, symbols: List[str]) -> np.ndarray:
    """
    CSV format: symbol,weight  OR index=symbol with column 'weight'
    Missing symbols assumed 0. Renormalize to sum=1 if possible.
    """
    df = pd.read_csv(path)
    if df.shape[1] >= 2 and "symbol" in df.columns and "weight" in df.columns:
        wmap = {str(r["symbol"]).strip().upper(): float(r["weight"]) for _, r in df.iterrows()}
    else:
        # try index-based
        df2 = pd.read_csv(path, index_col=0)
        if "weight" not in df2.columns:
            raise ValueError("current weights CSV must have 'weight' column (and optional 'symbol' column).")
        wmap = {str(k).strip().upper(): float(v) for k, v in df2["weight"].to_dict().items()}

    w0 = np.array([float(wmap.get(s.upper(), 0.0)) for s in symbols], dtype=float)
    s = float(np.sum(w0))
    if abs(s) <= 1e-12:
        return np.ones(len(symbols), dtype=float) / max(len(symbols), 1)
    return w0 / s


def markowitz_with_turnover(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_current: np.ndarray,
    gamma: float,
    turnover_penalty: float,
    long_only: bool,
    max_weight: float,
    eps: float = 1e-10,
) -> np.ndarray:
    d = int(len(mu))
    if d == 0:
        return np.array([])

    Sigma_psd = 0.5 * (Sigma + Sigma.T) + eps * np.eye(d)
    w = cp.Variable(d)

    utility = (
        mu @ w
        - 0.5 * float(gamma) * cp.quad_form(w, Sigma_psd)
        - float(turnover_penalty) * cp.sum_squares(w - np.asarray(w_current, dtype=float))
    )

    constraints = [cp.sum(w) == 1.0]
    if long_only:
        constraints += [w >= 0.0, w <= float(max_weight)]
    else:
        constraints += [w >= -float(max_weight), w <= float(max_weight)]

    prob = cp.Problem(cp.Maximize(utility), constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            return np.asarray(w_current, dtype=float)
        wv = np.array(w.value).reshape(-1)
        if long_only:
            wv = np.maximum(wv, 0.0)
        s = float(np.sum(wv))
        if abs(s) <= 1e-12:
            return np.asarray(w_current, dtype=float)
        return wv / s
    except Exception:
        return np.asarray(w_current, dtype=float)


# ---------------------------
# Confidence-weighted exposure sizing + CASH
# ---------------------------

def portfolio_forecast_consensus(
    alpha_forecasts: Dict[str, np.ndarray],
    alpha_weights: Dict[str, float],
    w_star: np.ndarray,
    use_keys: Tuple[str, ...] = ("momentum", "reversion", "vol_adj"),
) -> float:
    w = np.asarray(w_star, dtype=float).reshape(-1)

    scores: List[float] = []
    weights: List[float] = []
    for k in use_keys:
        if k not in alpha_forecasts:
            continue
        mu_k = np.asarray(alpha_forecasts[k], dtype=float).reshape(-1)
        scores.append(float(np.dot(w, mu_k)))
        weights.append(float(alpha_weights.get(k, 0.0)))

    if len(scores) < 2:
        return 0.0

    s = np.asarray(scores, dtype=float)
    wk = np.asarray(weights, dtype=float)
    if float(wk.sum()) <= 1e-12:
        wk = np.ones_like(wk) / len(wk)
    else:
        wk = wk / float(wk.sum())

    s_bar = float(np.dot(wk, s))
    sign_ref = 1.0 if s_bar >= 0 else -1.0
    agree = (np.sign(s) == sign_ref).astype(float)
    sign_agreement = float(np.dot(wk, agree))  # [0,1]

    s_abs = np.abs(s) + 1e-12
    disp = float(np.sqrt(np.dot(wk, (s - s_bar) ** 2)) / float(np.dot(wk, s_abs)))
    mag_agreement = float(1.0 / (1.0 + 3.0 * disp))  # [0,1] approx

    consensus = 0.6 * sign_agreement + 0.4 * mag_agreement
    return float(np.clip(consensus, 0.0, 1.0))


def exposure_from_consensus(
    consensus: float,
    e_min: float = 0.10,
    e_max: float = 1.00,
    deadband: float = 0.55,
    sharpness: float = 10.0,
) -> float:
    c = float(np.clip(consensus, 0.0, 1.0))
    x = float(sharpness) * (c - float(deadband))
    s = 1.0 / (1.0 + float(np.exp(-x)))
    return float(e_min + (e_max - e_min) * s)


def apply_exposure_and_cash(w_star: np.ndarray, exposure: float, symbols: List[str]) -> pd.Series:
    w_star = np.asarray(w_star, dtype=float).reshape(-1)
    e = float(np.clip(exposure, 0.0, 1.0))
    w_exec = e * w_star
    cash = 1.0 - float(np.sum(w_exec))
    out = pd.Series(w_exec, index=symbols, dtype=float)
    out.loc["CASH"] = cash
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
    n = int(len(est.returns))
    if n <= 2:
        raise RuntimeError("Not enough return observations to form a confidence ellipse.")

    mu_ann = est.mu_annual.values.astype(float)
    Sigma_daily = est.cov_daily.values.astype(float)
    Cov_mu_ann = (TRADING_DAYS_PER_YEAR**2) * Sigma_daily / n

    d = len(est.symbols)
    sym_index = {s: i for i, s in enumerate(est.symbols)}

    if symbols_2d is not None:
        if len(symbols_2d) != 2:
            raise ValueError("--ellipse-symbols must contain exactly 2 symbols")
        a, b = symbols_2d[0].upper(), symbols_2d[1].upper()
        if a not in sym_index or b not in sym_index:
            raise ValueError(f"Symbols for ellipse must be among kept symbols: {est.symbols}")
        idx2 = [sym_index[a], sym_index[b]]
        W = np.zeros((d, 2), dtype=float)
        W[idx2[0], 0] = 1.0
        W[idx2[1], 1] = 1.0
        plane_label = f"{a} vs {b}"
        xlab, ylab = f"μ_annual({a})", f"μ_annual({b})"
    elif d == 2:
        W = np.eye(2, dtype=float)
        plane_label = f"{est.symbols[0]} vs {est.symbols[1]}"
        xlab, ylab = f"μ_annual({est.symbols[0]})", f"μ_annual({est.symbols[1]})"
    else:
        W = _pca_project_2d(Sigma_daily)
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
    ax.set_title(f"{int(conf*100)}% Confidence Ellipse for Annualized Mean Returns\n({plane_label}, n={n})")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


# ---------------------------
# Main
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "./data_store"))
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--symbols", type=str, default="")
    p.add_argument("--min-overlap", type=int, default=120)

    p.add_argument("--save-dir", type=str, default="")
    p.add_argument("--prefix", type=str, default="estimators")

    # robust mean shrink + legacy robust weights
    p.add_argument("--robust", action="store_true", help="Compute global-robust shrink + constrained weights CSV")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--ridge", type=float, default=1e-10, help="Tiny ridge for alpha calc stability only")
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--max-weight", type=float, default=0.15, help="Per-asset cap (long-only uses 0..max)")
    p.add_argument("--risk-aversion", type=float, default=1.0, help="Legacy MV risk-aversion for --robust output")

    # markowitz utility (turnover + regime gamma + alpha learning + confidence exposure)
    p.add_argument("--markowitz", action="store_true", help="Run the full Markowitz utility allocator (recommended)")
    p.add_argument("--gamma", type=float, default=6.0, help="Base risk aversion for Markowitz utility")
    p.add_argument("--turnover-penalty", type=float, default=12.0, help="Higher => trade less (penalize ||w-w0||^2)")
    p.add_argument("--current-weights", type=str, default="", help="CSV of current weights (symbol,weight) or index=Symbol with 'weight' col")

    # regime gamma
    p.add_argument("--regime-short-window", type=int, default=20)
    p.add_argument("--regime-long-window", type=int, default=120)

    # mu uncertainty shrink
    p.add_argument("--mu-shrink", type=float, default=1.5, help="Per-asset uncertainty shrink strength")

    # alpha learning persistence
    p.add_argument("--alpha-weights-path", type=str, default="", help="Path to alpha_weights.json (default: save_dir/alpha_weights.json)")
    p.add_argument("--learn-alphas", action="store_true", help="Update alpha weights using last realized return")
    p.add_argument("--learning-rate", type=float, default=5.0, help="Online learning rate for alpha weights")
    p.add_argument("--mom-lookback", type=int, default=60)
    p.add_argument("--rev-window", type=int, default=5)
    p.add_argument("--vol-window", type=int, default=30)

    # confidence exposure sizing
    p.add_argument("--exposure-sizing", action="store_true", help="Scale exposure based on alpha consensus; put remainder in CASH")
    p.add_argument("--e-min", type=float, default=0.10)
    p.add_argument("--e-max", type=float, default=1.00)
    p.add_argument("--consensus-deadband", type=float, default=0.55)
    p.add_argument("--consensus-sharpness", type=float, default=10.0)

    # ellipse plot
    p.add_argument("--plot-ellipse", action="store_true")
    p.add_argument("--ellipse-symbols", type=str, default="")
    p.add_argument("--ellipse-out", type=str, default="")

    args = p.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else out_dir

    # symbols override
    symbols_override = None
    if args.symbols.strip():
        symbols_override = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    est = estimate_portfolio(
        out_dir=out_dir,
        lookback_days=args.lookback_days,
        symbols_override=symbols_override,
        min_overlap=args.min_overlap,
    )

    # base outputs
    paths = _save_estimators(est, save_dir, prefix=args.prefix)

    print("\n=== Estimators (Ledoit–Wolf Σ) ===")
    print(f"Symbols (kept): {', '.join(est.symbols)}")
    print(f"Aligned return rows: {len(est.returns)} (dates {est.returns.index.min()} → {est.returns.index.max()})")
    print("\nmu (annualized):")
    print(est.mu_annual.sort_values(ascending=False).to_string(float_format=lambda x: f"{x: .4f}"))
    print("\nSaved estimator CSVs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    # feasibility guard for caps (if long-only)
    d = len(est.symbols)
    if args.long_only and args.max_weight * d < 1.0 - 1e-12:
        needed = 1.0 / max(d, 1)
        raise SystemExit(
            f"Infeasible constraints: long-only with max_weight={args.max_weight:.3f} "
            f"and d={d} symbols cannot sum to 1. Need max_weight >= {needed:.3f}."
        )

    # ----- legacy robust weights output -----
    alpha_global = None
    if args.robust:
        rob = compute_robust_portfolio(
            est,
            conf=args.confidence,
            ridge=args.ridge,
            long_only=args.long_only,
            max_weight=args.max_weight,
            risk_aversion=args.risk_aversion,
        )
        alpha_global = float(rob.attrs.get("alpha", np.nan))
        n = int(rob.attrs.get("n", len(est.returns)))

        w_path = save_dir / f"{args.prefix}_weights_robust.csv"
        rob.to_csv(w_path, index=True)

        print("\n=== Robust (global shrink + constrained MV weights) ===")
        print(
            f"confidence={args.confidence:.3f}, n={n}, long_only={args.long_only}, "
            f"max_weight={args.max_weight:.3f}, risk_aversion={args.risk_aversion:g}"
        )
        print(f"alpha (shrink factor) = {alpha_global:.6f}  (mu_shrunk = alpha * mu_hat)")
        print("\nTop weights (robust):")
        print(
            rob["weight_robust"].sort_values(ascending=False).head(15).to_string(
                float_format=lambda x: f"{x: .4f}"
            )
        )
        print(f"\nSaved: {w_path}")

    # ----- full markowitz allocator -----
    if args.markowitz:
        # start from mu_hat
        mu_hat = est.mu_daily.values.astype(float)
        Sigma = est.cov_daily.values.astype(float)
        n = int(len(est.returns))

        # 1) global robust shrink (if not run above, compute alpha anyway)
        if alpha_global is None:
            alpha_global = robust_shrinkage_alpha(mu_hat, Sigma, n=n, conf=args.confidence, ridge=args.ridge)
        mu_global = float(alpha_global) * mu_hat

        # 2) per-asset uncertainty shrink
        mu_unc = uncertainty_adjusted_mu(est.returns, mu_global, shrink_strength=args.mu_shrink)

        # 3) multi-alpha forecasts (with user windows)
        # override alpha functions windows by temporarily adjusting lookbacks
        # (simple: use globals via closures not needed; just recompute here)
        # Recompute using requested windows:
        def _mom_alpha_custom() -> np.ndarray:
            lb = min(max(2, int(args.mom_lookback)), len(est.returns))
            mom = (1.0 + est.returns).rolling(lb).apply(lambda x: np.prod(x), raw=True) - 1.0
            return mom.iloc[-1].values.astype(float)

        def _rev_alpha_custom() -> np.ndarray:
            w = min(max(2, int(args.rev_window)), len(est.returns))
            return (-est.returns.tail(w).mean(axis=0).values.astype(float))

        def _vol_alpha_custom() -> np.ndarray:
            w = min(max(5, int(args.vol_window)), len(est.returns))
            vol = est.returns.rolling(w).std().iloc[-1].values.astype(float)
            trend = est.returns.rolling(w).mean().iloc[-1].values.astype(float)
            return trend / (vol + 1e-8)

        alpha_forecasts = {
            "base": mu_unc,
            "momentum": _mom_alpha_custom(),
            "reversion": _rev_alpha_custom(),
            "vol_adj": _vol_alpha_custom(),
        }

        # 4) load alpha weights + blend
        alpha_path = Path(args.alpha_weights_path).expanduser().resolve() if args.alpha_weights_path.strip() else (save_dir / "alpha_weights.json")
        alpha_weights = load_alpha_weights(alpha_path)

        mu_blend = blended_expected_returns_dynamic(alpha_forecasts, alpha_weights)

        # 5) regime-dependent gamma
        regime, vol_ratio = compute_volatility_regime(est.returns, short_window=args.regime_short_window, long_window=args.regime_long_window)
        gamma_t = regime_gamma(args.gamma, regime)

        # 6) current weights + turnover-aware utility optimization
        if args.current_weights.strip():
            w0 = _load_current_weights_csv(Path(args.current_weights).expanduser().resolve(), est.symbols)
        else:
            w0 = np.ones(d, dtype=float) / d

        w_star = markowitz_with_turnover(
            mu=mu_blend,
            Sigma=Sigma,
            w_current=w0,
            gamma=gamma_t,
            turnover_penalty=args.turnover_penalty,
            long_only=args.long_only,
            max_weight=args.max_weight,
        )

        # save w*
        w_star_ser = pd.Series(w_star, index=est.symbols).sort_values(ascending=False)
        w_star_out = save_dir / "markowitz_optimal_weights.csv"
        w_star_ser.to_csv(w_star_out, header=["weight"])

        # 7) optional exposure sizing + CASH
        if args.exposure_sizing:
            consensus = portfolio_forecast_consensus(alpha_forecasts, alpha_weights, w_star)
            exposure = exposure_from_consensus(
                consensus,
                e_min=args.e_min,
                e_max=args.e_max,
                deadband=args.consensus_deadband,
                sharpness=args.consensus_sharpness,
            )
            w_exec = apply_exposure_and_cash(w_star, exposure, est.symbols)
            w_exec_out = save_dir / "markowitz_executable_weights_with_cash.csv"
            w_exec.sort_values(ascending=False).to_csv(w_exec_out, header=["weight"])
        else:
            consensus = None
            exposure = None
            w_exec_out = None

        # 8) online learning update (optional)
        if args.learn_alphas:
            realized = est.returns.iloc[-1].values.astype(float)
            alpha_weights_new = update_alpha_weights(
                alpha_forecasts=alpha_forecasts,
                realized_returns=realized,
                prev_weights=alpha_weights,
                learning_rate=args.learning_rate,
            )
            save_alpha_weights(alpha_path, alpha_weights_new)
            alpha_weights = alpha_weights_new  # for printing

        # print summary
        mu_p_ann = float(np.dot(mu_blend, w_star)) * TRADING_DAYS_PER_YEAR
        vol_p_ann = float(np.sqrt(max(w_star @ Sigma @ w_star, 0.0))) * np.sqrt(TRADING_DAYS_PER_YEAR)
        util_daily = float(mu_blend @ w_star - 0.5 * gamma_t * (w_star @ Sigma @ w_star) - args.turnover_penalty * np.sum((w_star - w0) ** 2))
        util_ann = util_daily * TRADING_DAYS_PER_YEAR

        print("\n=== Markowitz Utility Allocator (full stack) ===")
        print(f"global_alpha={float(alpha_global):.6f}  mu_shrink={args.mu_shrink:g}")
        print(f"regime={regime}  vol_ratio={vol_ratio:.3f}  gamma_base={args.gamma:g}  gamma_t={gamma_t:.3f}")
        print(f"turnover_penalty={args.turnover_penalty:g}  long_only={args.long_only}  max_weight={args.max_weight:g}")
        print(f"\nE[R_p] annualized (blend): {mu_p_ann: .4f}")
        print(f"Vol annualized:            {vol_p_ann: .4f}")
        print(f"Utility daily:             {util_daily: .8f}")
        print(f"Utility annual:            {util_ann: .4f}")

        print("\nAlpha weights:")
        for k in ["base", "momentum", "reversion", "vol_adj"]:
            print(f"  {k:9s}: {float(alpha_weights.get(k, 0.0)):.3f}")
        print(f"Saved alpha weights: {alpha_path}")

        print("\nOptimal weights (w*):")
        print(w_star_ser.to_string(float_format=lambda x: f"{x: .4f}"))
        print(f"Saved: {w_star_out}")

        if args.exposure_sizing:
            print("\nConfidence exposure sizing:")
            print(f"  consensus={float(consensus):.3f}  exposure={float(exposure):.3f}")
            print(f"Saved executable weights (incl CASH): {w_exec_out}")

    # ----- ellipse plot -----
    if args.plot_ellipse:
        symbols_2d = None
        if args.ellipse_symbols.strip():
            symbols_2d = [s.strip().upper() for s in args.ellipse_symbols.split(",") if s.strip()]

        ellipse_path = (
            Path(args.ellipse_out).expanduser().resolve()
            if args.ellipse_out
            else (save_dir / f"{args.prefix}_mu_ellipse_{int(args.confidence * 100)}.png")
        )
        plot_mu_confidence_ellipse_95(est, ellipse_path, conf=args.confidence, symbols_2d=symbols_2d)
        print(f"\nSaved ellipse plot: {ellipse_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
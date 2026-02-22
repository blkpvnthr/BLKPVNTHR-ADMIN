import os
import csv
import math
import json
import hmac
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import deque
from typing import Deque, Dict, Optional, List, Tuple, Any
from pathlib import Path

import asyncio
import httpx
from dotenv import load_dotenv

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingModuleSource]

from alpaca.data.live import StockDataStream  # pyright: ignore[reportMissingImports]
from alpaca.data.enums import DataFeed  # pyright: ignore[reportMissingImports]
from alpaca.data.historical import StockHistoricalDataClient  # pyright: ignore[reportMissingImports]
from alpaca.data.requests import StockLatestQuoteRequest  # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()


# Fetch variables from environment

# -----------------------------
# Config
# -----------------------------
NY = ZoneInfo("America/New_York")
load_dotenv()

APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_PAPER = os.getenv("APCA_PAPER", "true").lower() == "true"

if not APCA_API_KEY_ID:
    raise ValueError("Missing APCA_API_KEY_ID in environment / .env")
if not APCA_API_SECRET_KEY:
    raise ValueError("Missing APCA_API_SECRET_KEY in environment / .env")

WS_URL = os.getenv("ALPACA_WS_URL", "wss://stream.data.alpaca.markets/v2/iex")
RUNNER_IMAGE = os.getenv("RUNNER_IMAGE", "trading-runner:latest")
WORK_ROOT = Path(os.getenv("WORK_ROOT", "workspaces"))

# Safety caps
RUN_TIMEOUT_SEC_DEFAULT = int(os.getenv("RUN_TIMEOUT_SEC", "30"))
RUN_CPUS = os.getenv("RUN_CPUS", "1.0")
RUN_MEMORY = os.getenv("RUN_MEMORY", "768m")
RUN_PIDS = os.getenv("RUN_PIDS", "256")

# Webhook -> FastAPI (UI / feed)
FASTAPI_EVENTS_URL = os.getenv("FASTAPI_EVENTS_URL", "http://127.0.0.1:8000/api/events").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
WEBHOOK_TIMEOUT_SEC = float(os.getenv("WEBHOOK_TIMEOUT_SEC", "3.5"))
WEBHOOK_ENABLED = os.getenv("WEBHOOK_ENABLED", "true").strip().lower() in ("1", "true", "yes", "y", "on")


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def today_yyyymmdd(ts: datetime) -> str:
    return ts.astimezone(NY).strftime("%Y%m%d")


def day_key(ts: datetime) -> Tuple[int, int, int]:
    t = ts.astimezone(NY)
    return (t.year, t.month, t.day)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _utc_iso(ts: Optional[datetime] = None) -> str:
    t = ts or datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).isoformat()


def _sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


class FastAPIWebhookPublisher:
    """
    Non-blocking event publisher to FastAPI.
    Sends JSON to FASTAPI_EVENTS_URL with optional HMAC signature header.
    """

    def __init__(self, url: str, secret: str = "", timeout_sec: float = 3.5, enabled: bool = True):
        self.url = url
        self.secret = secret
        self.timeout_sec = timeout_sec
        self.enabled = enabled

    async def publish(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        if "ts" not in payload:
            payload["ts"] = _utc_iso()

        body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["X-Webhook-Signature"] = _sign(self.secret, body)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                r = await client.post(self.url, content=body, headers=headers)
                # swallow errors, do not break trading loop
                if r.status_code >= 400:
                    # You can uncomment for debugging:
                    # print(f"[WEBHOOK ERROR] {r.status_code} {r.text[:200]}")
                    return
        except Exception:
            # swallow network errors
            return


WEBHOOK = FastAPIWebhookPublisher(
    url=FASTAPI_EVENTS_URL,
    secret=WEBHOOK_SECRET,
    timeout_sec=WEBHOOK_TIMEOUT_SEC,
    enabled=WEBHOOK_ENABLED,
)


def push_ui(event_type: str, **data: Any) -> None:
    """
    Fire-and-forget publish into FastAPI for UI / dashboard.
    Safe to call inside async handlers; never blocks.
    """
    if not WEBHOOK_ENABLED:
        return

    payload = {
        "type": event_type,
        "ts": _utc_iso(),
        **data,
    }

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(WEBHOOK.publish(payload))
    except RuntimeError:
        # No running event loop (unlikely in stream callbacks, but safe)
        return


# -----------------------------
# Bar model
# -----------------------------
@dataclass
class OHLCV:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# -----------------------------
# Quote model (context only)
# -----------------------------
@dataclass
class QuoteSnap:
    time: Optional[datetime] = None
    bid_price: float = float("nan")
    bid_size: float = float("nan")
    ask_price: float = float("nan")
    ask_size: float = float("nan")

    @property
    def mid(self) -> float:
        if math.isnan(self.bid_price) or math.isnan(self.ask_price):
            return float("nan")
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread(self) -> float:
        if math.isnan(self.bid_price) or math.isnan(self.ask_price):
            return float("nan")
        return self.ask_price - self.bid_price

    @property
    def spread_pct_mid(self) -> float:
        m = self.mid
        if math.isnan(m) or m == 0:
            return float("nan")
        return (self.spread / m) * 100.0


# -----------------------------
# Bar aggregation (1m -> tf)  ✅ emits ONLY CLOSED bars
# -----------------------------
class BarAggregator:
    def __init__(self, timeframe_minutes: int):
        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be > 0")
        self.tf = timeframe_minutes
        self.current: Optional[OHLCV] = None
        self.bucket_key: Optional[Tuple[Tuple[int, int, int], int, int]] = None

    def _calc_bucket_key(self, ts: datetime) -> Tuple[Tuple[int, int, int], int, int]:
        ts = ts.astimezone(NY)
        d = (ts.year, ts.month, ts.day)
        hour = ts.hour
        bucket_start_min = ts.minute - (ts.minute % self.tf)
        return (d, hour, bucket_start_min)

    def update(self, bar_1m: OHLCV) -> Optional[OHLCV]:
        ts = bar_1m.time.astimezone(NY)
        key = self._calc_bucket_key(ts)

        if self.current is None:
            self.bucket_key = key
            self.current = OHLCV(
                time=ts.replace(second=0, microsecond=0),
                open=bar_1m.open,
                high=bar_1m.high,
                low=bar_1m.low,
                close=bar_1m.close,
                volume=bar_1m.volume,
            )
            return None

        if key != self.bucket_key:
            finished = self.current
            self.bucket_key = key
            self.current = OHLCV(
                time=ts.replace(second=0, microsecond=0),
                open=bar_1m.open,
                high=bar_1m.high,
                low=bar_1m.low,
                close=bar_1m.close,
                volume=bar_1m.volume,
            )
            return finished

        self.current.high = max(self.current.high, bar_1m.high)
        self.current.low = min(self.current.low, bar_1m.low)
        self.current.close = bar_1m.close
        self.current.volume += bar_1m.volume
        return None


# -----------------------------
# Features (computed only from CLOSED bars)
# -----------------------------
class RollingSeries:
    def __init__(self, maxlen: int = 1200):
        self.bars: Deque[OHLCV] = deque(maxlen=maxlen)
        self._vwap_day: Optional[Tuple[int, int, int]] = None
        self._cum_pv: float = 0.0
        self._cum_v: float = 0.0

    def add(self, bar: OHLCV) -> None:
        self.bars.append(bar)

        dk = day_key(bar.time)
        if self._vwap_day != dk:
            self._vwap_day = dk
            self._cum_pv = 0.0
            self._cum_v = 0.0

        tp = (bar.high + bar.low + bar.close) / 3.0
        v = float(bar.volume) if bar.volume is not None else 0.0
        if not math.isnan(tp) and not math.isnan(v) and v > 0:
            self._cum_pv += tp * v
            self._cum_v += v

    def current_vwap(self) -> float:
        if self._cum_v <= 0:
            return float("nan")
        return float(self._cum_pv / self._cum_v)

    def _to_df(self) -> pd.DataFrame:
        if not self.bars:
            return pd.DataFrame()
        df = pd.DataFrame(
            [
                {"time": b.time, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
                for b in self.bars
            ]
        ).set_index("time")
        return df

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float:
        if len(close) < period + 1:
            return float("nan")
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace({0: np.nan})
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return float("nan")
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return float(atr.iloc[-1])

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period * 2 + 2:
            return float("nan")

        high = df["high"]
        low = df["low"]
        close = df["close"]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100.0 * (plus_dm_s / atr.replace({0: np.nan}))
        minus_di = 100.0 * (minus_dm_s / atr.replace({0: np.nan}))

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace({0: np.nan}))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        return float(adx.iloc[-1])

    def compute_features(self) -> Dict[str, float]:
        df = self._to_df()
        if df.empty or len(df) < 3:
            return {}

        close = df["close"]
        volume = df["volume"]

        ret_1 = float((close.iloc[-1] / close.iloc[-2]) - 1.0) if close.iloc[-2] else float("nan")
        ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1]) if len(close) >= 5 else float("nan")
        ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 5 else float("nan")
        rsi_14 = self._rsi(close, 14)

        atr_14 = self._atr(df, 14)
        atrp_14 = float((atr_14 / close.iloc[-1]) * 100.0) if close.iloc[-1] else float("nan")
        adx_14 = self._adx(df, 14)

        vwap = self.current_vwap()
        vwap_dist_pct = (
            float(((close.iloc[-1] - vwap) / vwap) * 100.0) if vwap and not math.isnan(vwap) else float("nan")
        )

        vol_roll = volume.rolling(20)
        vol_mean = vol_roll.mean().iloc[-1]
        vol_std = vol_roll.std(ddof=0).iloc[-1]
        vol_z_20 = (
            float((volume.iloc[-1] - vol_mean) / vol_std) if vol_std and not math.isnan(vol_std) else float("nan")
        )

        return {
            "ret_1": ret_1,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "rsi_14": rsi_14,
            "atr_14": atr_14,
            "atrp_14": atrp_14,
            "adx_14": adx_14,
            "vwap": vwap,
            "vwap_dist_pct": vwap_dist_pct,
            "vol_z_20": vol_z_20,
        }


# -----------------------------
# Eligibility Engine (closed bars only) + VWAP checkpoint rules
# -----------------------------
class CandidateState(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    RECONSIDER = "RECONSIDER"
    CONFIRMED = "CONFIRMED"


class DirectionBias(str, Enum):
    LONG_ONLY = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    BOTH = "BOTH"


@dataclass
class SymbolSession:
    state: CandidateState = CandidateState.ACTIVE
    dir_5m: List[int] = field(default_factory=list)
    flip_count: int = 0
    score: float = 1.0

    vwap_dist_5m: float = float("nan")
    vwap_dist_15m: float = float("nan")
    vwap_dist_30m: float = float("nan")
    vwap_dist_1h: float = float("nan")

    bias: DirectionBias = DirectionBias.BOTH
    allow_trend: bool = True
    allow_mean_reversion: bool = True
    sizing_mult: float = 1.0

    dir_15m: int = 0
    dir_30m: int = 0
    last_reason: str = ""


def candle_dir(open_: float, close: float, min_body_pct: float = 0.0005) -> int:
    if open_ <= 0 or math.isnan(open_) or math.isnan(close):
        return 0
    body = (close - open_) / open_
    if body >= min_body_pct:
        return +1
    if body <= -min_body_pct:
        return -1
    return 0


class EligibilityEngine:
    def __init__(
        self,
        vwap_extreme_dist_pct: float = 1.25,
        vwap_near_pct: float = 0.10,
        min_body_pct: float = 0.0005,
        max_initial_5m_bars: int = 6,
    ):
        self.vwap_extreme_dist_pct = vwap_extreme_dist_pct
        self.vwap_near_pct = vwap_near_pct
        self.min_body_pct = min_body_pct
        self.max_initial_5m_bars = max_initial_5m_bars
        self.sessions: Dict[str, SymbolSession] = {}

    def session(self, symbol: str) -> SymbolSession:
        if symbol not in self.sessions:
            self.sessions[symbol] = SymbolSession()
        return self.sessions[symbol]

    def risk_multiplier(self, symbol: str) -> float:
        s = self.session(symbol)
        if s.state == CandidateState.SUSPENDED:
            return 0.0
        if s.state == CandidateState.RECONSIDER:
            base = 0.5
        elif s.state == CandidateState.ACTIVE:
            base = 0.75
        else:
            base = 1.0
        return float(base * s.sizing_mult)

    def update_on_5m_close(self, symbol: str, bar5: OHLCV, vwap_dist_pct_5m: float, bar_index_5m: int) -> SymbolSession:
        s = self.session(symbol)
        s.vwap_dist_5m = float(vwap_dist_pct_5m) if vwap_dist_pct_5m is not None else float("nan")

        d = candle_dir(bar5.open, bar5.close, self.min_body_pct)

        if len(s.dir_5m) < self.max_initial_5m_bars:
            if s.dir_5m and d != 0 and s.dir_5m[-1] != 0 and d != s.dir_5m[-1]:
                s.flip_count += 1
            s.dir_5m.append(d)

        if bar_index_5m == 1:
            if (not math.isnan(s.vwap_dist_5m)) and abs(s.vwap_dist_5m) >= self.vwap_extreme_dist_pct:
                s.score *= 0.25
                s.state = CandidateState.SUSPENDED
                s.last_reason = f"Suspended: 5m#1 extreme VWAP dislocation ({s.vwap_dist_5m:.2f}%)"
                return s
            s.last_reason = "5m#1 ok (no extreme VWAP dislocation)"
            return s

        if bar_index_5m == 2 and len(s.dir_5m) >= 2:
            d1, d2 = s.dir_5m[0], s.dir_5m[1]
            if d1 != 0 and d2 != 0 and d1 != d2:
                near_vwap = (not math.isnan(s.vwap_dist_5m)) and (abs(s.vwap_dist_5m) <= self.vwap_near_pct)
                s.score *= 0.40 if near_vwap else 0.55
                s.state = CandidateState.SUSPENDED
                s.last_reason = (
                    f"Suspended: 5m#1/#2 conflict + near VWAP ({s.vwap_dist_5m:.2f}%)"
                    if near_vwap
                    else f"Suspended: 5m#1/#2 conflict (VWAP dist {s.vwap_dist_5m:.2f}%)"
                )
                return s

        if len(s.dir_5m) >= 4 and s.flip_count >= 3:
            s.score *= 0.80
            s.allow_trend = False
            s.allow_mean_reversion = True
            s.last_reason = f"Chop detected: flip_count={s.flip_count}; trend disabled"

        return s

    def update_on_15m_close(self, symbol: str, bar15: OHLCV, vwap_dist_pct_15m: float) -> SymbolSession:
        s = self.session(symbol)
        s.vwap_dist_15m = float(vwap_dist_pct_15m) if vwap_dist_pct_15m is not None else float("nan")
        s.dir_15m = candle_dir(bar15.open, bar15.close, self.min_body_pct)

        if not math.isnan(s.vwap_dist_15m):
            if s.vwap_dist_15m > self.vwap_near_pct:
                s.bias = DirectionBias.LONG_ONLY
            elif s.vwap_dist_15m < -self.vwap_near_pct:
                s.bias = DirectionBias.SHORT_ONLY
            else:
                s.bias = DirectionBias.BOTH

        near_vwap = (not math.isnan(s.vwap_dist_15m)) and (abs(s.vwap_dist_15m) <= self.vwap_near_pct)
        if near_vwap:
            s.allow_mean_reversion = True
            if s.flip_count >= 2:
                s.allow_trend = False
            s.last_reason = f"15m near VWAP ({s.vwap_dist_15m:.2f}%): bias=BOTH; favor reversion"
        else:
            s.allow_trend = s.flip_count < 3
            s.allow_mean_reversion = True
            s.last_reason = f"15m bias={s.bias.value} from VWAP ({s.vwap_dist_15m:.2f}%)"

        if s.state == CandidateState.SUSPENDED and s.dir_15m != 0:
            s.score *= 1.15
            s.state = CandidateState.RECONSIDER
            s.last_reason = f"Reconsider: 15m candle directional (dir_15m={s.dir_15m})"

        return s

    def update_on_30m_close(self, symbol: str, bar30: OHLCV, vwap_dist_pct_30m: float) -> SymbolSession:
        s = self.session(symbol)
        s.vwap_dist_30m = float(vwap_dist_pct_30m) if vwap_dist_pct_30m is not None else float("nan")
        s.dir_30m = candle_dir(bar30.open, bar30.close, self.min_body_pct)

        if s.state == CandidateState.SUSPENDED and s.dir_30m != 0:
            s.state = CandidateState.RECONSIDER
            s.last_reason = f"Reconsider: 30m candle directional (dir_30m={s.dir_30m})"

        aligns_with_vwap: Optional[bool] = None
        if not math.isnan(s.vwap_dist_30m):
            if s.dir_30m > 0:
                aligns_with_vwap = s.vwap_dist_30m > self.vwap_near_pct
            elif s.dir_30m < 0:
                aligns_with_vwap = s.vwap_dist_30m < -self.vwap_near_pct

        if s.dir_30m == 0:
            near_vwap = (not math.isnan(s.vwap_dist_30m)) and (abs(s.vwap_dist_30m) <= self.vwap_near_pct)
            if near_vwap:
                s.allow_trend = False
                s.allow_mean_reversion = True
                if s.state != CandidateState.SUSPENDED:
                    s.state = CandidateState.RECONSIDER
                s.last_reason = f"30m neutral + near VWAP ({s.vwap_dist_30m:.2f}%): trend blocked"
            else:
                s.last_reason = "30m neutral: no confirmation"
            return s

        if aligns_with_vwap is False:
            s.score *= 0.70
            s.allow_trend = False
            s.allow_mean_reversion = True
            if s.state != CandidateState.SUSPENDED:
                s.state = CandidateState.RECONSIDER
            s.last_reason = f"30m fights VWAP ({s.vwap_dist_30m:.2f}%): trend blocked; RECONSIDER"
            return s

        if s.flip_count <= 3:
            s.score *= 1.20
            s.state = CandidateState.CONFIRMED
        else:
            s.score *= 0.90
            s.state = CandidateState.RECONSIDER
            s.allow_trend = False
            s.allow_mean_reversion = True
            s.last_reason = f"30m confirms but choppy (flips={s.flip_count}): trend blocked"

        if not math.isnan(s.vwap_dist_30m):
            if s.vwap_dist_30m > self.vwap_near_pct:
                s.bias = DirectionBias.LONG_ONLY
            elif s.vwap_dist_30m < -self.vwap_near_pct:
                s.bias = DirectionBias.SHORT_ONLY
            else:
                s.bias = DirectionBias.BOTH

        return s

    def update_on_1h_close(self, symbol: str, vwap_dist_pct_1h: float) -> SymbolSession:
        s = self.session(symbol)
        s.vwap_dist_1h = float(vwap_dist_pct_1h) if vwap_dist_pct_1h is not None else float("nan")

        if math.isnan(s.vwap_dist_1h):
            s.sizing_mult = 1.0
            return s

        near_vwap = abs(s.vwap_dist_1h) <= self.vwap_near_pct
        if near_vwap:
            s.sizing_mult = 0.85
            s.last_reason = f"1h sizing down: near VWAP ({s.vwap_dist_1h:.2f}%)"
            return s

        if s.bias == DirectionBias.LONG_ONLY and s.vwap_dist_1h > 0:
            s.sizing_mult = 1.10
            s.last_reason = f"1h sizing up: above VWAP ({s.vwap_dist_1h:.2f}%)"
        elif s.bias == DirectionBias.SHORT_ONLY and s.vwap_dist_1h < 0:
            s.sizing_mult = 1.10
            s.last_reason = f"1h sizing up: below VWAP ({s.vwap_dist_1h:.2f}%)"
        else:
            s.sizing_mult = 1.0
            s.last_reason = f"1h sizing neutral: VWAP ({s.vwap_dist_1h:.2f}%) vs bias {s.bias.value}"

        return s


# -----------------------------
# Regime labeling (1h CLOSED only)
# -----------------------------
def label_regime_from_1h(adx_14: float, atrp_14: float) -> Dict[str, str]:
    trend_label = "TREND" if (not math.isnan(adx_14) and adx_14 >= 22.0) else "RANGE"
    if math.isnan(atrp_14):
        vol_label = "UNKNOWN_VOL"
    else:
        vol_label = "HIGH_VOL" if atrp_14 >= 0.60 else "LOW_VOL"
    return {"trend_label": trend_label, "vol_label": vol_label, "regime": f"{trend_label}_{vol_label}"}


# -----------------------------
# Disk writer (CSV append)
# -----------------------------
class CSVAppender:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        ensure_dir(out_dir)

    def _path(self, kind: str, tf: str, symbol: str, ts: datetime) -> str:
        d = today_yyyymmdd(ts)
        base = os.path.join(self.out_dir, kind, tf)
        ensure_dir(base)
        return os.path.join(base, f"{symbol}_{d}.csv")

    def append_row(self, kind: str, tf: str, symbol: str, ts: datetime, row: Dict[str, object]) -> None:
        path = self._path(kind, tf, symbol, ts)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def append_market_regime(self, ts: datetime, row: Dict[str, object]) -> None:
        base = os.path.join(self.out_dir, "regimes")
        ensure_dir(base)
        path = os.path.join(base, "market_regime.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def append_session_state(self, ts: datetime, row: Dict[str, object]) -> None:
        base = os.path.join(self.out_dir, "session_state")
        ensure_dir(base)
        path = os.path.join(base, f"session_state_{today_yyyymmdd(ts)}.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def append_quote_log(self, ts: datetime, row: Dict[str, object]) -> None:
        base = os.path.join(self.out_dir, "quotes")
        ensure_dir(base)
        path = os.path.join(base, f"quotes_{today_yyyymmdd(ts)}.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# -----------------------------
# Quote Cache (latest snapshot per symbol) + initial fetch
# -----------------------------
class QuoteCache:
    def __init__(self, historical_client: StockHistoricalDataClient, writer: CSVAppender, log_quotes: bool):
        self.client = historical_client
        self.writer = writer
        self.log_quotes = log_quotes
        self.latest: Dict[str, QuoteSnap] = {}

    def get(self, symbol: str) -> QuoteSnap:
        return self.latest.get(symbol, QuoteSnap())

    def seed_latest_quotes(self, symbols: List[str]) -> None:
        if not symbols:
            return
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        data = self.client.get_stock_latest_quote(req)  # dict keyed by symbol
        for sym in symbols:
            q = data.get(sym)
            if q is None:
                continue
            self.latest[sym] = QuoteSnap(
                time=getattr(q, "timestamp", None),
                bid_price=safe_float(getattr(q, "bid_price", float("nan"))),
                bid_size=safe_float(getattr(q, "bid_size", float("nan"))),
                ask_price=safe_float(getattr(q, "ask_price", float("nan"))),
                ask_size=safe_float(getattr(q, "ask_size", float("nan"))),
            )

    def update_from_stream(self, q: Any) -> None:
        sym = getattr(q, "symbol", None)
        if not sym:
            return
        snap = QuoteSnap(
            time=getattr(q, "timestamp", None),
            bid_price=safe_float(getattr(q, "bid_price", float("nan"))),
            bid_size=safe_float(getattr(q, "bid_size", float("nan"))),
            ask_price=safe_float(getattr(q, "ask_price", float("nan"))),
            ask_size=safe_float(getattr(q, "ask_size", float("nan"))),
        )
        self.latest[sym] = snap

        if self.log_quotes:
            ts = snap.time or datetime.now(tz=NY)
            row = {
                "time": (ts.astimezone(NY).isoformat() if hasattr(ts, "astimezone") else str(ts)),
                "symbol": sym,
                "bid_price": snap.bid_price,
                "bid_size": snap.bid_size,
                "ask_price": snap.ask_price,
                "ask_size": snap.ask_size,
                "mid": snap.mid,
                "spread": snap.spread,
                "spread_pct_mid": snap.spread_pct_mid,
            }
            self.writer.append_quote_log(datetime.now(tz=NY), row)

            # ✅ also publish quote events to FastAPI (optional)
            push_ui(
                "quote",
                symbol=sym,
                quote_time=row["time"],
                bid_price=snap.bid_price,
                ask_price=snap.ask_price,
                mid=snap.mid,
                spread=snap.spread,
                spread_pct_mid=snap.spread_pct_mid,
            )


# -----------------------------
# Market Monitor
# -----------------------------
class MarketMonitor:
    def __init__(self, symbols: List[str], out_dir: str, feed: DataFeed, regime_symbol: str):
        self.symbols = symbols
        self.writer = CSVAppender(out_dir)

        api_key = os.environ.get("APCA_API_KEY_ID", "")
        api_secret = os.environ.get("APCA_API_SECRET_KEY", "")
        if not api_key or not api_secret:
            raise RuntimeError("Missing API keys. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
        self.api_key = api_key
        self.api_secret = api_secret

        vwap_extreme = env_float("VWAP_EXTREME_DIST_PCT", 1.25)
        vwap_near = env_float("VWAP_NEAR_PCT", 0.10)
        self.engine = EligibilityEngine(vwap_extreme_dist_pct=vwap_extreme, vwap_near_pct=vwap_near)

        self.regime_symbol_override = (regime_symbol or "").strip().upper()
        self.regime_symbol: Optional[str] = self.regime_symbol_override if self.regime_symbol_override in symbols else None

        self.aggs: Dict[str, Dict[str, BarAggregator]] = {
            "5m": {s: BarAggregator(5) for s in symbols},
            "15m": {s: BarAggregator(15) for s in symbols},
            "30m": {s: BarAggregator(30) for s in symbols},
            "1h": {s: BarAggregator(60) for s in symbols},
        }

        self.series: Dict[str, Dict[str, RollingSeries]] = {
            "5m": {s: RollingSeries(maxlen=1600) for s in symbols},
            "15m": {s: RollingSeries(maxlen=1600) for s in symbols},
            "30m": {s: RollingSeries(maxlen=1600) for s in symbols},
            "1h": {s: RollingSeries(maxlen=4000) for s in symbols},
        }

        self.closed_5m_count: Dict[str, int] = {s: 0 for s in symbols}

        hist_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        self.quote_cache = QuoteCache(
            historical_client=hist_client,
            writer=self.writer,
            log_quotes=env_bool("LOG_QUOTES", False),
        )
        self.quote_cache.seed_latest_quotes(self.symbols)

        self.feed = feed

    def _maybe_select_regime_symbol(self) -> None:
        if self.regime_symbol_override and self.regime_symbol_override in self.symbols:
            self.regime_symbol = self.regime_symbol_override
            return

        best_sym = None
        best_score = -1.0
        for sym in self.symbols:
            sess = self.engine.session(sym)
            if sess.state == CandidateState.CONFIRMED and sess.score > best_score:
                best_score = sess.score
                best_sym = sym

        if best_sym and self.regime_symbol != best_sym:
            self.regime_symbol = best_sym
            print(f"[REGIME_SYMBOL SELECTED] {best_sym} (highest CONFIRMED score={best_score:.3f})")
            push_ui("regime_symbol_selected", symbol=best_sym, score=float(best_score))

    def _write_bar(self, tf: str, symbol: str, bar: OHLCV) -> None:
        row = {
            "time": bar.time.astimezone(NY).isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        self.writer.append_row(kind="bars", tf=tf, symbol=symbol, ts=bar.time, row=row)

        # ✅ publish closed bar
        push_ui(
            "bar_closed",
            tf=tf,
            symbol=symbol,
            time=row["time"],
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )

    def _write_features(self, tf: str, symbol: str, ts: datetime, feats: Dict[str, float]) -> None:
        row = {"time": ts.astimezone(NY).isoformat(), **feats}
        self.writer.append_row(kind="features", tf=tf, symbol=symbol, ts=ts, row=row)

        # ✅ publish features (optional)
        push_ui("features", tf=tf, symbol=symbol, **row)

    def _write_market_regime(self, ts: datetime, symbol: str, feats_1h: Dict[str, float]) -> None:
        labels = label_regime_from_1h(
            adx_14=float(feats_1h.get("adx_14", float("nan"))),
            atrp_14=float(feats_1h.get("atrp_14", float("nan"))),
        )
        row = {
            "time": ts.astimezone(NY).isoformat(),
            "symbol": symbol,
            "adx_14": float(feats_1h.get("adx_14", float("nan"))),
            "atrp_14": float(feats_1h.get("atrp_14", float("nan"))),
            "trend_label": labels["trend_label"],
            "vol_label": labels["vol_label"],
            "regime": labels["regime"],
        }
        self.writer.append_market_regime(ts, row)

        push_ui("market_regime", **row)

    def _write_session_state(self, ts: datetime, symbol: str, event: str) -> None:
        s = self.engine.session(symbol)
        q = self.quote_cache.get(symbol)
        qt = (
            q.time.astimezone(NY).isoformat()
            if (q.time and hasattr(q.time, "astimezone"))
            else (str(q.time) if q.time else "")
        )

        row = {
            "time": ts.astimezone(NY).isoformat(),
            "symbol": symbol,
            "event": event,
            "score": round(float(s.score), 6),
            "state": s.state.value,
            "bias": s.bias.value,
            "allow_trend": int(s.allow_trend),
            "allow_mean_reversion": int(s.allow_mean_reversion),
            "flip_count": int(s.flip_count),
            "dir_5m_1": s.dir_5m[0] if len(s.dir_5m) > 0 else 0,
            "dir_5m_2": s.dir_5m[1] if len(s.dir_5m) > 1 else 0,
            "dir_5m_3": s.dir_5m[2] if len(s.dir_5m) > 2 else 0,
            "dir_5m_4": s.dir_5m[3] if len(s.dir_5m) > 3 else 0,
            "dir_5m_5": s.dir_5m[4] if len(s.dir_5m) > 4 else 0,
            "dir_5m_6": s.dir_5m[5] if len(s.dir_5m) > 5 else 0,
            "dir_15m": int(s.dir_15m),
            "dir_30m": int(s.dir_30m),
            "vwap_dist_5m": s.vwap_dist_5m,
            "vwap_dist_15m": s.vwap_dist_15m,
            "vwap_dist_30m": s.vwap_dist_30m,
            "vwap_dist_1h": s.vwap_dist_1h,
            "sizing_mult": float(s.sizing_mult),
            "risk_mult": float(self.engine.risk_multiplier(symbol)),
            "reason": s.last_reason,
            "quote_time": qt,
            "bid_price": q.bid_price,
            "bid_size": q.bid_size,
            "ask_price": q.ask_price,
            "ask_size": q.ask_size,
            "mid": q.mid,
            "spread": q.spread,
            "spread_pct_mid": q.spread_pct_mid,
        }

        self.writer.append_session_state(ts, row)

        # ✅ publish session state into FastAPI so your UI can render "what's going on"
        push_ui("session_state", **row)

    # --- Live handlers ---
    async def on_quote(self, q) -> None:
        self.quote_cache.update_from_stream(q)

    async def on_1m_bar(self, bar) -> None:
        ts = bar.timestamp.astimezone(NY)
        bar_1m = OHLCV(
            time=ts.replace(second=0, microsecond=0),
            open=safe_float(bar.open),
            high=safe_float(bar.high),
            low=safe_float(bar.low),
            close=safe_float(bar.close),
            volume=safe_float(bar.volume),
        )

        sym = bar.symbol
        if sym not in self.symbols:
            return

        for tf in ("1h", "30m", "15m", "5m"):
            finished = self.aggs[tf][sym].update(bar_1m)
            if not finished:
                continue

            self._write_bar(tf, sym, finished)

            self.series[tf][sym].add(finished)
            feats = self.series[tf][sym].compute_features()
            if feats:
                self._write_features(tf, sym, finished.time, feats)

            vwap_dist = float(feats.get("vwap_dist_pct", float("nan"))) if feats else float("nan")

            if tf == "5m":
                self.closed_5m_count[sym] += 1
                idx = self.closed_5m_count[sym]
                self.engine.update_on_5m_close(sym, finished, vwap_dist_pct_5m=vwap_dist, bar_index_5m=idx)
                self._write_session_state(finished.time, sym, event="5m_close")

            elif tf == "15m":
                self.engine.update_on_15m_close(sym, finished, vwap_dist_pct_15m=vwap_dist)
                self._write_session_state(finished.time, sym, event="15m_close")

            elif tf == "30m":
                self.engine.update_on_30m_close(sym, finished, vwap_dist_pct_30m=vwap_dist)
                self._write_session_state(finished.time, sym, event="30m_close")
                self._maybe_select_regime_symbol()

            elif tf == "1h":
                self.engine.update_on_1h_close(sym, vwap_dist_pct_1h=vwap_dist)
                self._write_session_state(finished.time, sym, event="1h_close")

                if self.regime_symbol is not None and sym == self.regime_symbol and feats:
                    self._write_market_regime(finished.time, sym, feats)

            tf_label = {"1h": "1H", "30m": "30M", "15m": "15M", "5m": "5M"}[tf]
            print(f"[{tf_label} CLOSED] {sym} {finished.time.astimezone(NY).strftime('%H:%M')} close={finished.close:.2f}")

    def run(self) -> None:
        stream = StockDataStream(self.api_key, self.api_secret, feed=self.feed)
        stream.subscribe_bars(self.on_1m_bar, *self.symbols)
        stream.subscribe_quotes(self.on_quote, *self.symbols)

        print(f"Streaming 1m bars + live quotes for {len(self.symbols)} symbols on feed={self.feed} ...")
        print("CLOSED-BAR ONLY: eligibility + features updated ONLY on closed 5m/15m/30m/1h bars.")
        print(f"VWAP thresholds: extreme={self.engine.vwap_extreme_dist_pct:.2f}%  near={self.engine.vwap_near_pct:.2f}%")
        print(f"Webhook -> FastAPI: {'ENABLED' if WEBHOOK_ENABLED else 'DISABLED'} url={FASTAPI_EVENTS_URL}")

        if self.regime_symbol:
            print(f"Regime benchmark symbol (preset): {self.regime_symbol}")
        else:
            print("Regime benchmark symbol: AUTO (select first CONFIRMED at 30m)")

        if env_bool("LOG_QUOTES", False):
            print("Quote logging enabled: OUT_DIR/quotes/quotes_YYYYMMDD.csv")

        stream.run()


# -----------------------------
# Entrypoint
# -----------------------------
def parse_symbols() -> List[str]:
    raw = os.environ.get(
        "SYMBOLS",
        "XLE,XLV,DIA,QQQ,TQQQ,QTUM,BUZZ,RGTI,HEPS,OUST,IONQ,CHAT,QBTS,LUNR,SOUN,UBER,MIND,WAVE,OPEN,KXIN,NGD,NKTR,IAG,ASRT,KITT,LUNR,HFUS,UNG,NVD,GORO",
    ).strip("\n")
    syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not syms:
        raise ValueError("No symbols provided.")
    return syms


def parse_feed() -> DataFeed:
    raw = os.environ.get("ALPACA_DATA_FEED", "IEX").strip().upper()
    return DataFeed.SIP if raw == "SIP" else DataFeed.IEX


if __name__ == "__main__":
    symbols = parse_symbols()
    out_dir = os.environ.get("OUT_DIR", "./data_store")
    feed = parse_feed()
    regime_symbol = os.environ.get("REGIME_SYMBOL", "").strip().upper()

    monitor = MarketMonitor(symbols=symbols, out_dir=out_dir, feed=feed, regime_symbol=regime_symbol)
    monitor.run()
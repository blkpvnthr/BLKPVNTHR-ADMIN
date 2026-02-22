import os
import csv
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

# -----------------------------
# Supabase config (for /sync)
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SYNC_SECRET = os.getenv("SYNC_SECRET", "").strip()

if not SUPABASE_URL or not SERVICE_ROLE:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

REST_BASE = f"{SUPABASE_URL}/rest/v1"


def sb_headers() -> Dict[str, str]:
    return {
        "apikey": SERVICE_ROLE,
        "Authorization": f"Bearer {SERVICE_ROLE}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def safe_json(r: requests.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return r.text


def sb_upsert(
    table: str, rows: List[Dict[str, Any]], on_conflict: Optional[str] = None
) -> Any:
    """
    Upsert rows into Supabase using REST.
    Uses Prefer: resolution=merge-duplicates
    Optional on_conflict (comma-separated unique constraint columns).
    """
    if not rows:
        return {"ok": True, "count": 0}

    params = {}
    if on_conflict:
        params["on_conflict"] = on_conflict

    url = f"{REST_BASE}/{table}"
    headers = sb_headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=representation"

    r = requests.post(url, headers=headers, params=params, json=rows, timeout=30)
    if r.status_code >= 300:
        raise HTTPException(
            status_code=500,
            detail={
                "table": table,
                "status": r.status_code,
                "body": safe_json(r),
            },
        )
    return r.json()


# -----------------------------
# Pydantic payload models
# -----------------------------
class AccountIn(BaseModel):
    id: str  # uuid (client-generated or existing)
    user_id: str
    name: str
    type: str = "brokerage"
    institution: str = "Fidelity"


class BalanceIn(BaseModel):
    user_id: str
    account_id: str
    as_of: date
    balance: float = 0
    day_change: float = 0


class PerfIn(BaseModel):
    user_id: str
    account_id: str
    as_of: date
    equity: float = 0


class HoldingIn(BaseModel):
    user_id: str
    account_id: str
    as_of: Optional[str] = None  # ISO timestamp; if omitted we'll use server time
    symbol: str
    qty: float = 0
    market_value: float = 0
    day_change_pct: Optional[float] = None


class SyncPayload(BaseModel):
    user_id: Optional[str] = None
    accounts: List[AccountIn] = Field(default_factory=list)
    balances_daily: List[BalanceIn] = Field(default_factory=list)
    performance_daily: List[PerfIn] = Field(default_factory=list)
    holdings_snapshot: List[HoldingIn] = Field(default_factory=list)


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="BLKPVNTHR API")

# If you serve frontend elsewhere, keep CORS.
# If you serve frontend from this same FastAPI (recommended below), CORS isn't required.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


def require_secret(x_sync_secret: Optional[str]):
    # If you set SYNC_SECRET, require it. If empty, no protection.
    if SYNC_SECRET and x_sync_secret != SYNC_SECRET:
        raise HTTPException(status_code=401, detail="Invalid X-Sync-Secret")


# -----------------------------
# /sync  (your existing endpoint)
# -----------------------------
@app.post("/sync")
def sync(payload: SyncPayload, x_sync_secret: Optional[str] = Header(default=None)):
    """
    Bulk upsert into Supabase tables:
      - accounts (conflict: id)
      - account_balances_daily (conflict: account_id,as_of)
      - account_performance_daily (conflict: account_id,as_of)
      - holdings_snapshot (append-only insert)
    """
    require_secret(x_sync_secret)

    any_user = payload.user_id or (payload.accounts[0].user_id if payload.accounts else None)
    if not any_user:
        raise HTTPException(status_code=400, detail="Missing user_id")

    accounts_rows = [a.model_dump() for a in payload.accounts]
    balances_rows = [b.model_dump() for b in payload.balances_daily]
    perf_rows = [p.model_dump() for p in payload.performance_daily]
    holdings_rows = [h.model_dump() for h in payload.holdings_snapshot]

    accounts_result = sb_upsert("accounts", accounts_rows, on_conflict="id")

    balances_result = sb_upsert(
        "account_balances_daily", balances_rows, on_conflict="account_id,as_of"
    )

    perf_result = sb_upsert(
        "account_performance_daily", perf_rows, on_conflict="account_id,as_of"
    )

    holdings_result = None
    if holdings_rows:
        url = f"{REST_BASE}/holdings_snapshot"
        headers = sb_headers()
        headers["Prefer"] = "return=representation"
        r = requests.post(url, headers=headers, json=holdings_rows, timeout=30)
        if r.status_code >= 300:
            raise HTTPException(
                status_code=500,
                detail={
                    "table": "holdings_snapshot",
                    "status": r.status_code,
                    "body": safe_json(r),
                },
            )
        holdings_result = r.json()

    return {
        "ok": True,
        "upserted": {
            "accounts": len(accounts_rows),
            "balances_daily": len(balances_rows),
            "performance_daily": len(perf_rows),
            "holdings_snapshot": len(holdings_rows),
        },
        "results": {
            "accounts": accounts_result,
            "balances_daily": balances_result,
            "performance_daily": perf_result,
            "holdings_snapshot": holdings_result,
        },
    }


# -----------------------------
# /api/market-status  (NEW)
# -----------------------------
# We'll use Stooq (free CSV) so it works without keys.
# You can swap this later to Alpaca/Polygon.
STOOQ_MAP = {
    "SPY": "spy.us",
    "QQQ": "qqq.us",
    "DIA": "dia.us",
    "IWM": "iwm.us",
    "VIX": "vix",  # stooq symbol
}

_market_cache: Tuple[float, Dict[str, Any]] = (0.0, {})


def stooq_daily_closes(symbol: str, limit: int = 60) -> List[Tuple[str, float]]:
    """
    Returns list of (date_str, close) ascending by date.
    """
    s = STOOQ_MAP.get(symbol, symbol).lower()
    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    rows: List[Tuple[str, float]] = []
    reader = csv.DictReader(r.text.splitlines())
    for row in reader:
        d = (row.get("Date") or "").strip()
        c = (row.get("Close") or "").strip()
        if not d or not c:
            continue
        try:
            rows.append((d, float(c)))
        except Exception:
            continue

    if not rows:
        return []

    # keep last N
    rows = rows[-limit:]
    return rows


def pct_change_from_last_two(closes: List[Tuple[str, float]]) -> Optional[float]:
    if len(closes) < 2:
        return None
    prev = closes[-2][1]
    last = closes[-1][1]
    if prev == 0:
        return None
    return (last / prev - 1.0) * 100.0


def sma(values: List[float], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


@app.get("/api/market-status")
def market_status():
    """
    Frontend calls this every ~30s.
    Response:
      {
        as_of, spy_pct, qqq_pct, dia_pct, iwm_pct, vix_pct,
        regime, risk
      }
    """
    now = time.time()
    cache_ts, cache_val = _market_cache
    if cache_val and (now - cache_ts) < 25:  # basic throttle
        return cache_val

    try:
        spy = stooq_daily_closes("SPY", limit=80)
        qqq = stooq_daily_closes("QQQ", limit=80)
        dia = stooq_daily_closes("DIA", limit=80)
        iwm = stooq_daily_closes("IWM", limit=80)
        vix = stooq_daily_closes("VIX", limit=80)

        spy_pct = pct_change_from_last_two(spy)
        qqq_pct = pct_change_from_last_two(qqq)
        dia_pct = pct_change_from_last_two(dia)
        iwm_pct = pct_change_from_last_two(iwm)
        vix_pct = pct_change_from_last_two(vix)

        # Simple regime: SPY close vs SMA20
        spy_closes = [c for _, c in spy]
        spy_last = spy_closes[-1] if spy_closes else None
        spy_sma20 = sma(spy_closes, 20)

        if spy_last is None or spy_sma20 is None:
            regime = "—"
        else:
            if spy_last > spy_sma20 * 1.01:
                regime = "TRENDING_UP"
            elif spy_last < spy_sma20 * 0.99:
                regime = "TRENDING_DOWN"
            else:
                regime = "RANGE"

        # Simple risk: SPY green + VIX red => risk_on; SPY red + VIX green => risk_off
        if spy_pct is None or vix_pct is None:
            risk = "—"
        else:
            if spy_pct > 0 and vix_pct < 0:
                risk = "RISK_ON"
            elif spy_pct < 0 and vix_pct > 0:
                risk = "RISK_OFF"
            else:
                risk = "MIXED"

        as_of = spy[-1][0] if spy else None

        out = {
            "as_of": as_of,  # YYYY-MM-DD (daily close date from stooq)
            "spy_pct": spy_pct,
            "qqq_pct": qqq_pct,
            "dia_pct": dia_pct,
            "iwm_pct": iwm_pct,
            "vix_pct": vix_pct,
            "regime": regime,
            "risk": risk,
        }

        global _market_cache
        _market_cache = (now, out)
        return out

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"market-status fetch failed: {e}")


# -----------------------------
# OPTIONAL: serve your frontend from the same FastAPI
# This makes your index.html fetch("/api/market-status") work perfectly.
# -----------------------------
# Put your static site files in ./public (index.html, styles.css, app.js, navbar.html, etc.)
PUBLIC_DIR = os.getenv("PUBLIC_DIR", "public")
if os.path.isdir(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

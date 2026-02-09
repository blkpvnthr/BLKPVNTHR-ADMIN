#!/usr/bin/env python3
"""
snaptrade_roth_publisher.py

Polls SnapTrade for the connected user's Roth IRA account data (accounts + holdings)
and posts a normalized snapshot to a FastAPI ingest endpoint.

Uses the maintained SnapTrade Python SDK (snaptrade-python-sdk). :contentReference[oaicite:2]{index=2}

ENV VARS REQUIRED:
  SNAPTRADE_CLIENT_ID
  SNAPTRADE_CONSUMER_KEY
  SNAPTRADE_USER_ID
  SNAPTRADE_USER_SECRET

  INGEST_URL                 e.g. http://localhost:8000/ingest/snaptrade
  INGEST_HMAC_SECRET         shared secret used to sign outgoing payload

OPTIONAL:
  POLL_SECONDS               default 60
  ROTH_ACCOUNT_ID            if you already know the SnapTrade accountId for the Roth
  ROTH_KEYWORDS              comma-separated keywords to identify Roth if accountId not provided
                             default: "roth,ira,roth ira"
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Optional
from pprint import pprint

import requests
from snaptrade_client import (
    SnapTrade,
)  # pip install snaptrade-python-sdk


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise SystemExit(f"Missing required env var: {name}")
    return v


@dataclass(frozen=True)
class Cfg:
    client_id: str
    consumer_key: str
    user_id: str
    user_secret: str
    ingest_url: str
    ingest_hmac_secret: str
    poll_seconds: int
    roth_account_id: Optional[str]
    roth_keywords: List[str]


def load_cfg() -> Cfg:
    kws = os.getenv("ROTH_KEYWORDS", "roth,ira,roth ira")
    return Cfg(
        client_id=_env("SNAPTRADE_CLIENT_ID"),
        consumer_key=_env("SNAPTRADE_CONSUMER_KEY"),
        user_id=_env("SNAPTRADE_USER_ID"),
        user_secret=_env("SNAPTRADE_USER_SECRET"),
        ingest_url=_env("INGEST_URL"),
        ingest_hmac_secret=_env("INGEST_HMAC_SECRET"),
        poll_seconds=int(os.getenv("POLL_SECONDS", "60")),
        roth_account_id=os.getenv("ROTH_ACCOUNT_ID") or None,
        roth_keywords=[k.strip().lower() for k in kws.split(",") if k.strip()],
    )


def sign_payload(payload_bytes: bytes, secret: str) -> str:
    """
    Returns base64(HMAC_SHA256(payload_bytes, secret)).
    """
    mac = hmac.new(secret.encode("utf-8"), payload_bytes, sha256).digest()
    return base64.b64encode(mac).decode("utf-8")


def pick_roth_account(
    accounts: List[Dict[str, Any]], keywords: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Heuristic match by displayName/nickname/name/accountType fields.
    If you can, prefer setting ROTH_ACCOUNT_ID explicitly.
    """

    def hay(a: Dict[str, Any]) -> str:
        parts = [
            str(a.get("display_name") or a.get("displayName") or ""),
            str(a.get("name") or ""),
            str(a.get("nickname") or ""),
            str(a.get("account_type") or a.get("accountType") or ""),
            str(a.get("type") or ""),
        ]
        return " ".join(parts).lower()

    scored = []
    for a in accounts:
        h = hay(a)
        score = sum(1 for k in keywords if k in h)
        scored.append((score, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def normalize_snapshot(
    *,
    user_id: str,
    account: Dict[str, Any],
    holdings: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Keep this fairly raw so you don’t lose data; your FastAPI can store/transform later.
    Holdings endpoint returns balances + positions + recent orders for an account. :contentReference[oaicite:4]{index=4}
    """
    return {
        "ts": int(time.time()),
        "source": "snaptrade",
        "userId": user_id,
        "account": {
            "id": account.get("id")
            or account.get("account_id")
            or account.get("accountId"),
            "name": account.get("name")
            or account.get("display_name")
            or account.get("displayName"),
            "type": account.get("account_type")
            or account.get("accountType")
            or account.get("type"),
            "raw": account,
        },
        "holdings": holdings,
    }


def post_snapshot(cfg: Cfg, snapshot: Dict[str, Any]) -> None:
    body = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    sig = sign_payload(body, cfg.ingest_hmac_secret)

    headers = {
        "Content-Type": "application/json",
        "X-Signature": sig,
    }
    r = requests.post(cfg.ingest_url, data=body, headers=headers, timeout=30)
    r.raise_for_status()


def main() -> None:
    cfg = load_cfg()

    snaptrade = SnapTrade(
        consumer_key=cfg.consumer_key,
        client_id=cfg.client_id,
    )  # SDK signs/authenticates requests using clientId+consumerKey. :contentReference[oaicite:5]{index=5}

    while True:
        # List accounts for this SnapTrade user. :contentReference[oaicite:6]{index=6}
        accounts_resp = snaptrade.account_information.list_user_accounts(
            user_id=cfg.user_id,
            user_secret=cfg.user_secret,
        )
        accounts = (
            accounts_resp.body if hasattr(accounts_resp, "body") else accounts_resp
        )  # SDK versions differ

        roth_account: Optional[Dict[str, Any]] = None
        if cfg.roth_account_id:
            roth_account = next(
                (
                    a
                    for a in accounts
                    if (a.get("id") or a.get("accountId") or a.get("account_id"))
                    == cfg.roth_account_id
                ),
                None,
            )
            if not roth_account:
                raise RuntimeError(
                    f"ROTH_ACCOUNT_ID={cfg.roth_account_id} not found in SnapTrade accounts response."
                )
        else:
            roth_account = pick_roth_account(accounts, cfg.roth_keywords)
            if not roth_account:
                raise RuntimeError(
                    "Couldn't identify a Roth account automatically. Set ROTH_ACCOUNT_ID to the SnapTrade accountId."
                )

        account_id = (
            roth_account.get("id")
            or roth_account.get("accountId")
            or roth_account.get("account_id")
        )
        if not account_id:
            raise RuntimeError("Account object missing id/accountId.")

        # Fetch holdings for that account. :contentReference[oaicite:7]{index=7}
        holdings_resp = snaptrade.account_information.get_user_holdings(
            account_id=account_id,
            user_id=cfg.user_id,
            user_secret=cfg.user_secret,
        )
        holdings = (
            holdings_resp.body if hasattr(holdings_resp, "body") else holdings_resp
        )

        snapshot = normalize_snapshot(
            user_id=cfg.user_id, account=roth_account, holdings=holdings
        )
        post_snapshot(cfg, snapshot)

        print(
            f"Posted Roth snapshot for accountId={account_id} @ {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()

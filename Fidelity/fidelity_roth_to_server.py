#!/usr/bin/env python3
"""
fidelity_roth_to_server.py

Pulls account data via an OAuth2 + FDX-style API (e.g., via an approved Fidelity Access
data-sharing provider) and POSTs a normalized snapshot to your server.

✅ Designed for secure, permissioned access (no Fidelity username/password handling).
❗ You must supply your provider's endpoints (AUTH_URL, TOKEN_URL, API_BASE_URL),
   scopes, and client credentials.

Typical flow:
1) Run once with --print-auth-url, open the URL, consent, copy the redirected ?code=...
2) Run with --exchange-code "<CODE>" to store tokens locally.
3) Run normally to poll + post snapshots to your server.

Requirements:
  pip install requests

Environment variables:
  OAUTH_CLIENT_ID
  OAUTH_CLIENT_SECRET
  OAUTH_REDIRECT_URI
  OAUTH_AUTH_URL          (provider specific)
  OAUTH_TOKEN_URL         (provider specific)
  API_BASE_URL            (provider specific, often .../fdx/v1 or similar)
  OAUTH_SCOPES            (space-separated)
  SERVER_POST_URL         (your server endpoint to receive snapshots)
Optional:
  POLL_SECONDS            (default 60)
  TOKEN_CACHE_PATH        (default ./token_cache.json)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests


# ----------------------------
# Config
# ----------------------------


def env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise SystemExit(f"Missing required env var: {name}")
    return v


@dataclass(frozen=True)
class Config:
    client_id: str
    client_secret: str
    redirect_uri: str
    auth_url: str
    token_url: str
    api_base_url: str
    scopes: str
    server_post_url: str
    poll_seconds: int
    token_cache_path: str


def load_config() -> Config:
    return Config(
        client_id=env("OAUTH_CLIENT_ID"),
        client_secret=env("OAUTH_CLIENT_SECRET"),
        redirect_uri=env("OAUTH_REDIRECT_URI"),
        auth_url=env("OAUTH_AUTH_URL"),
        token_url=env("OAUTH_TOKEN_URL"),
        api_base_url=env("API_BASE_URL"),
        scopes=env("OAUTH_SCOPES"),
        server_post_url=env("SERVER_POST_URL"),
        poll_seconds=int(os.getenv("POLL_SECONDS", "60")),
        token_cache_path=os.getenv("TOKEN_CACHE_PATH", "./token_cache.json"),
    )


# ----------------------------
# Token cache + OAuth2
# ----------------------------


def read_token_cache(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_token_cache(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def now_epoch() -> int:
    return int(time.time())


def is_token_valid(cache: Dict[str, Any], skew_seconds: int = 60) -> bool:
    access_token = cache.get("access_token")
    expires_at = cache.get("expires_at")  # epoch seconds
    if not access_token or not expires_at:
        return False
    return (expires_at - skew_seconds) > now_epoch()


def build_auth_url(cfg: Config, state: str = "state123") -> str:
    # Provider may require extra params (audience, resource, prompt, etc.)
    params = {
        "response_type": "code",
        "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri,
        "scope": cfg.scopes,
        "state": state,
    }
    return f"{cfg.auth_url}?{urlencode(params)}"


def exchange_code_for_tokens(cfg: Config, code: str) -> Dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": cfg.redirect_uri,
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
    }
    r = requests.post(cfg.token_url, data=data, timeout=30)
    r.raise_for_status()
    tok = r.json()

    # Normalize to include expires_at
    expires_in = int(tok.get("expires_in", 3600))
    tok["expires_at"] = now_epoch() + expires_in
    return tok


def refresh_access_token(cfg: Config, refresh_token: str) -> Dict[str, Any]:
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
    }
    r = requests.post(cfg.token_url, data=data, timeout=30)
    r.raise_for_status()
    tok = r.json()
    expires_in = int(tok.get("expires_in", 3600))
    tok["expires_at"] = now_epoch() + expires_in

    # Some providers rotate refresh_token; preserve old if not returned
    if "refresh_token" not in tok:
        tok["refresh_token"] = refresh_token
    return tok


def get_valid_access_token(cfg: Config) -> Tuple[str, Dict[str, Any]]:
    cache = read_token_cache(cfg.token_cache_path)
    if is_token_valid(cache):
        return cache["access_token"], cache

    rt = cache.get("refresh_token")
    if not rt:
        raise SystemExit(
            "No valid access token and no refresh token found.\n"
            "Run with --print-auth-url then --exchange-code to bootstrap tokens."
        )

    new_cache = refresh_access_token(cfg, rt)
    write_token_cache(cfg.token_cache_path, new_cache)
    return new_cache["access_token"], new_cache


# ----------------------------
# FDX-style API client (provider specific)
# ----------------------------


class FDXClient:
    """
    FDX endpoint paths vary by provider. Replace these paths with your provider's docs.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _get(
        self, access_token: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = self.cfg.api_base_url.rstrip("/") + "/" + path.lstrip("/")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
        r = requests.get(url, headers=headers, params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_accounts(self, access_token: str) -> Dict[str, Any]:
        # Common FDX-ish: GET /accounts
        return self._get(access_token, "/accounts")

    def get_account(self, access_token: str, account_id: str) -> Dict[str, Any]:
        # Common: GET /accounts/{accountId}
        return self._get(access_token, f"/accounts/{account_id}")

    def get_investments(self, access_token: str, account_id: str) -> Dict[str, Any]:
        # Provider may use /investments or /accounts/{id}/investments
        return self._get(access_token, f"/accounts/{account_id}/investments")

    def get_transactions(
        self, access_token: str, account_id: str, start_time: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {}
        if start_time:
            params["startTime"] = start_time
        return self._get(
            access_token, f"/accounts/{account_id}/transactions", params=params
        )


def normalize_snapshot(
    accounts_payload: Dict[str, Any], roth_account_id_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalizes provider JSON into a simple snapshot.
    You should tailor this based on your provider's exact schema.

    If you have multiple accounts, set roth_account_id_hint to the ROTH accountId
    to avoid guessing.
    """
    snapshot: Dict[str, Any] = {
        "ts": now_epoch(),
        "source": "fdx_provider",
        "accounts": [],
    }

    accounts = accounts_payload.get("accounts") or accounts_payload.get("data") or []
    for a in accounts:
        snapshot["accounts"].append(
            {
                "accountId": a.get("accountId") or a.get("id"),
                "displayName": a.get("displayName")
                or a.get("nickname")
                or a.get("name"),
                "accountType": a.get("accountType") or a.get("type"),
                "accountNumberMasked": a.get("accountNumberMasked")
                or a.get("maskedNumber"),
            }
        )

    if roth_account_id_hint:
        snapshot["rothAccountId"] = roth_account_id_hint

    return snapshot


# ----------------------------
# Post to your server
# ----------------------------


def post_snapshot(cfg: Config, snapshot: Dict[str, Any]) -> None:
    """
    Posts JSON to your server.
    Your server should authenticate this request (API key, HMAC signature, mTLS, etc.)
    """
    headers = {"Content-Type": "application/json"}
    r = requests.post(cfg.server_post_url, headers=headers, json=snapshot, timeout=30)
    r.raise_for_status()


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    cfg = load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--print-auth-url",
        action="store_true",
        help="Print the OAuth consent URL and exit.",
    )
    ap.add_argument(
        "--exchange-code",
        type=str,
        default="",
        help="Exchange an OAuth authorization code for tokens.",
    )
    ap.add_argument(
        "--roth-account-id",
        type=str,
        default="",
        help="Optional: explicitly set your Roth accountId.",
    )
    ap.add_argument("--once", action="store_true", help="Run once and exit (no loop).")
    args = ap.parse_args()

    if args.print_auth_url:
        print(build_auth_url(cfg))
        return

    if args.exchange_code:
        tok = exchange_code_for_tokens(cfg, args.exchange_code.strip())
        write_token_cache(cfg.token_cache_path, tok)
        print(
            f"Saved tokens to {cfg.token_cache_path}. You can now run without --exchange-code."
        )
        return

    client = FDXClient(cfg)

    def run_one_cycle() -> None:
        access_token, _cache = get_valid_access_token(cfg)

        accounts_payload = client.list_accounts(access_token)
        snapshot = normalize_snapshot(
            accounts_payload, roth_account_id_hint=args.roth_account_id or None
        )

        # OPTIONAL: If you want positions/holdings, uncomment and adapt to your provider:
        # roth_id = args.roth_account_id
        # if roth_id:
        #     investments_payload = client.get_investments(access_token, roth_id)
        #     snapshot["rothInvestments"] = investments_payload

        post_snapshot(cfg, snapshot)
        print(f"Posted snapshot @ {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.once:
        run_one_cycle()
        return

    while True:
        try:
            run_one_cycle()
        except Exception as e:
            # In production: structured logging + backoff
            print(f"[error] {e}")
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()

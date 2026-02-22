# BLKPVNTHR-ADMIN Dashboard

**Private bookkeeping + trading ops dashboard for BLKPVNTHR LLC**

A modern, offline-friendly web UI for tracking investments, business payments, monthly totals, trust distributions, and tax documents — plus a trading workflow that streams live market data, builds multi-timeframe structure using **closed candles only**, and only allows trade qualification after confirming broad market direction.

---

## 🧭 System Architecture

BLKPVNTHR-ADMIN separates **market intelligence**, **portfolio construction**, and **operator visualization** into independent layers connected through deterministic data artifacts.

The system is intentionally file-driven rather than API-driven to ensure reproducibility, auditability, and offline resilience.

---

### High-Level Architecture

```text
                    ┌─────────────────────────────┐
                    │            Live             │
                    │         Market Data         │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │        monitor.py            │
                    │  Market Structure Engine     │
                    │                              │
                    │ • Closed candle analysis     │
                    │ • Multi-timeframe VWAP       │
                    │ • Direction lock             │
                    │ • Eligibility scoring        │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
         data_store/session_state/session_state_YYYYMMDD.csv
                                   │
                                   │ (single source of truth)
                                   ▼
        ┌───────────────────────────────────────────────────┐
        │                 Frontend Dashboard                │
        │                    (index.html)                   │
        │                                                   │
        │  • Watchlist Grid                                 │
        │  • Activity Feed                                  │
        │  • Momentum Leader                                │
        │  • TradingView Chart Carousel                     │
        └──────────────┬────────────────────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────────────────────────┐
        │          portfolio_estimators.py                  │
        │                                                   │
        │  • Return estimation                              │
        │  • Ledoit-Wolf covariance                         │
        │  • Expected return modeling                       │
        │  • Robust shrinkage                               │
        └──────────────┬────────────────────────────────────┘
                       │
                       ▼
        markowitz_executable_weights_with_cash.csv
                       │
                       ▼
        ┌───────────────────────────────────────────────────┐
        │                    trade.py                       │
        │                                                   │
        │  • Portfolio optimization                         │
        │  • Exposure sizing                                │
        │  • Risk constraints                               │
        │  • Alpaca paper execution                         │
        └───────────────────────────────────────────────────┘
```

---

## Stack

- Vanilla JS (no framework)
- Supabase Auth + Postgres
- DataTables v2
- Alpaca (data + paper trading)
- Python analytics pipeline (monitoring, estimators, Markowitz optimizer, execution tooling)

---

## ✨ Features

### Authentication & Privacy

- **Supabase Account** — required for any database writes  
- **Optional PIN gate** — client-side privacy screen for local viewing  
- Session persistence with real auth state sync

> The PIN is **not security** — it’s a convenience privacy layer.

### Bookkeeping

- Payments ledger
- Monthly / TTM totals view
- Snapshot historical totals
- Health & education flags per payment
- Memo field for notes

### Trading Ops (Quant Pipeline)

- Confirmed-symbol selection from `session_state_YYYYMMDD.csv` (generated by `monitor.py`)
- **Closed-bar only** signal updates (no intrabar repainting)
- Multi-timeframe structure (5m / 15m / 30m / 1h / daily, depending on config)
- Daily close-to-close return estimation (≈ 1 year lookback)
- **Ledoit–Wolf covariance shrinkage** for stability
- **Mean/variance estimators** (daily + annualized)
- Optional robust mean shrinkage based on a confidence ellipsoid
- Markowitz utility allocator (risk aversion + constraints)
- Optional confidence-based exposure sizing (adds implied CASH)
- **Daily notional trade cap** (default: ≤ 2% of account equity per day)
- Paper rebalancing execution via Alpaca

### Markets Screener UI (session_state viewer)

- Reads latest `session_state_YYYYMMDD.csv` from `/data_store/session_state/`
- Auto-fallback: if today’s file is missing, the UI can search backward to load the most recent available session file
- Watchlist grid: filters to **LONG_ONLY** + positive **15m VWAP distance**, with search + optional `window.WATCHLIST`
- Activity Feed: de-duplicated events (only logs when `state|score|bias|reason` changes)
- TradingView carousel: auto-builds a chart deck from the current grid symbols

### UI/UX

- Mobile responsive layout
- Modal workflow (DataTables / Editor where used)
- Optional blurred lock screen (if enabled)
- Local input persistence for planner-style fields

---

## 🔐 Security Model

| Action        | Requires      |
| ------------- | ------------- |
| View with PIN | Client only   |
| Save payment  | Supabase auth |
| Delete        | Supabase auth |
| Snapshots     | Supabase auth |

---

## 📁 Structure

Actual repository layout (simplified):

```text
BLKPVNTHR-ADMIN/
│
├── index.html                # Main dashboard / markets screener
├── ledger.html               # Bookkeeping ledger
├── paybills.html             # Bill tracking workflow
├── payments-history.html     # Payment archive view
├── TTM.html                  # Trailing twelve month summaries
├── navbar.html               # Shared navbar (dynamically injected)
├── styles.css                # Global UI styling
│
├── includes/
│   └── config.php            # PHP configuration (legacy/admin integration)
│
├── data_store/
│   ├── session_state/        # Daily trading eligibility output
│   │   └── session_state_YYYYMMDD.csv
│   └── weights/
│       └── markowitz_executable_weights_with_cash.csv
│
├── Fidelity/                 # Brokerage exports / reconciliation data
│
├── prisma/                   # Database schema (Supabase/Postgres tooling)
├── logs/                     # Runtime + pipeline logs
│
├── monitor.py                # Market monitoring + eligibility engine
├── trade.py                  # Portfolio rebalance executor (Alpaca)
├── backtest.py               # Historical strategy validation
├── portfolio_estimators.py   # Return + covariance estimation
├── risk_metrics.py           # Risk calculations + exposure metrics
├── tv_webhook_api.py         # TradingView webhook ingestion
├── server.py                 # Python API server (local services)
├── server.js                 # Node helper server (static/dev tooling)
│
├── universe.yaml             # Tradable universe configuration
├── requirements.txt          # Python dependencies
├── app.js                    # Shared frontend utilities
│
├── .env                      # Local configuration (NOT committed)
├── .gitignore
└── README.md
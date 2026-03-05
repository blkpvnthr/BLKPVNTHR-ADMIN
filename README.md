# BLKPVNTHR-ADMIN Dashboard

**Private bookkeeping + trading ops dashboard for BLKPVNTHR LLC**

A modern, offline-friendly web UI for tracking investments, business payments, monthly totals, trust distributions, and tax documents — plus a trading workflow that streams live market data, builds multi-timeframe structure using **closed candles only**, and only allows trade qualification after confirming broad market direction.

---
# Usage

1. Clone the repository and navigate to the project directory:
 ```bash
git clone https://github.com/blkpvnthr/BLKPVNTHR-ADMIN.git
 cd BLKPVNTHR-ADMIN
 ```

2. Install Python dependencies:
 ```bash
   pip install -r requirements.txt
   ```
3. Make a [supabase](https://supabase.com/) account and create a new project (its free). Note your Supabase URL and API key for the next step.

4. Set up [Alpaca](https://app.alpaca.markets/account/login) paper trading account for live market data stream. (Also free).

5. Set up environment variables in a `.env` file (see `.env.example` for reference):
 ```bash
   # Example .env content
        APCA_API_KEY_ID=your-alpaca-key-id
        APCA_API_SECRET_KEY=your-alpaca-secret-key
        APCA_PAPER=true

        SUPABASE_URL=https://xtespuzbuepublmiciyf.supabase.co
        SUPABASE_KEY=your-supabase-api-key

        FASTAPI_EVENTS_URL=http://127.0.0.1:8000/api/events
        WEBHOOK_SECRET=supersecret
        WEBHOOK_ENABLED=true
```
6. Run the market monitoring engine (generates daily `session_state_YYYYMMDD.csv`):
 ```bash
   python monitor.py
   ```

7. Utilize live server capabilities to serve the frontend dashboard (e.g., using `live-server` npm package or Python's `http.server`):
 ```bash
   # Using live-server (install globally with npm install -g live-server)
   live-server --port=8080
   # Or using Python's built-in server
   python -m http.server 8080
   ```
 Then access the dashboard at `http://localhost:8080/index.html`
   
> 8. To execute trades based on the latest signals and portfolio optimization, run: (Still in development)
 ```bash
   python trade.py
   ```
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
                    │        monitor.py           │
                    │  Market Structure Engine    │
                    │                             │
                    │ • Closed candle analysis    │
                    │ • Multi-timeframe VWAP      │
                    │ • Direction lock            │
                    │ • Eligibility scoring       │
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

## ✨ Features

### Authentication & Privacy

- **Supabase Account** — required for any database writes  
- **Optional PIN gate** — client-side privacy screen for local viewing  
- Session persistence with real auth state sync

> The PIN is **not security** — it’s a convenience privacy layer.

### Bookkeeping

- Income ledger
- Monthly / TTM Income view
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
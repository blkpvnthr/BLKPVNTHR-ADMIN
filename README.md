# BLKPVNTHR-ADMIN Dashboard

**Private bookkeeping dashboard for BLKPVNTHR LLC**
Built for tracking APL contractor payments, monthly totals, and trust-style distributions with a modern, offline-friendly UI.

**Stack**

* Vanilla JS (no framework)
* Supabase Auth + Postgres
* DataTables v2
* Client-side PIN privacy gate
* Responsive glass-style UI

---

## ✨ Features

### Authentication & Privacy

* **Magic-link sign-in (Supabase)** – required for any database writes
* **Optional PIN gate** – client-side privacy screen when you just want to view locally
* Session persistence with real auth state sync

### Payment Entry Engine

* Supports **gross-first** or **net-first** entry
* Auto-detect which field you typed last
* Per-payment override rates:

  * Pension %
  * Tax %
  * SS + Medicare %
* Live reconciliation panel validating:

```
gross − deductions = net
```

### Bookkeeping

* APL payments table with edit/delete
* Monthly live totals view
* Snapshot historical totals
* Health & education flags per payment
* Memo field for notes

### UI/UX

* Mobile responsive
* Modal workflow
* Blurred lock screen
* Local input persistence for dashboard planner

---

## 🚀 Getting Started

### 1. Supabase Setup

Create tables:

**apl_payments**

```sql
id uuid primary key default gen_random_uuid(),
user_id uuid not null references auth.users(id),
payment_date date not null,
amount_gross numeric,
amount_net numeric,
pension_deduction numeric,
tax_deduction numeric,
ss_deduction numeric,
health_insurance_paid boolean,
education_paid boolean,
status text,
memo text,
created_at timestamptz default now()
```

**apl_monthly_live** – view or function
**monthly_totals** – snapshot table

---

### 2. Environment

Add to `index.html`:

```html
<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
<script>
const SUPABASE_URL = "https://YOUR.supabase.co";
const SUPABASE_KEY = "sb_publishable_xxx";
window.sb = supabase.createClient(SUPABASE_URL, SUPABASE_KEY);
</script>
```

---

### 3. Run

Serve statically:

```bash
npx serve .
```

No build step required.

---

## 🧮 Calculation Logic

### Gross → Net

```
pension = gross × pensionRate  
tax     = gross × taxRate  
ss      = gross × ssRate  

net = gross − pension − tax − ss
```

### Net → Gross

```
gross = net / (1 − (pensionRate + taxRate + ssRate))
```

### Reconciliation

* OK: difference < $1
* Warn: < $5
* Error: ≥ $5

---

## 🔐 Security Model

| Action        | Requires      |
| ------------- | ------------- |
| View with PIN | Client only   |
| Save payment  | Supabase auth |
| Delete        | Supabase auth |
| Snapshots     | Supabase auth |

> The PIN is **not security**—it’s a convenience privacy layer.

---

## 📁 Structure

```
/index.html      UI + tables + modal
/styles.css      theme + overlay + responsive
/app.js          logic + Supabase + DataTables
```

---

## 🛠 Roadmap

* [ ] CSV export
* [ ] Paystub image attachment
* [ ] Multi-company mode
* [ ] Trust distribution planner
* [ ] Year-end 1099 report

---

## ⚖ License

Internal use – BLKPVNTHR LLC
All rights reserved.

---

## 🤝 Author

**Asmaa Abdul-Amin**
BLKPVNTHR LLC
Maryland, USA

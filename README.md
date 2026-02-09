# BLKPVNTHR-ADMIN Dashboard

**Private bookkeeping dashboard for BLKPVNTHR LLC**
Built for tracking investments, business payments, monthly totals, trust distributions, and tax documents with a modern, offline-friendly UI.

**Stack**

* Vanilla JS (no framework)
* Supabase Auth + Postgres
* DataTables v2
* Responsive glass-style UI

---

## ✨ Features

### Authentication & Privacy

* **Supabase Account** – required for any database writes
> * **Optional PIN gate** – client-side privacy screen when you just want to view locally disable this once Magic-link is integrated fully.
* Session persistence with real auth state sync

### Bookkeeping

* Payments ledger
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
/ledger.html     Income logging
/TTM.html        Monthly income aggregate
/navbar.html     Navigation
/footer.html     Footer
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

> Internal use – © BLKPVNTHR LLC, All rights reserved.

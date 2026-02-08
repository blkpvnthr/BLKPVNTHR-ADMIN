# BLKPVNTHR-ADMIN Dashboard

**Private bookkeeping dashboard for BLKPVNTHR LLC**
Built for tracking business payments, monthly totals, and trust-style distributions with a modern, offline-friendly UI.

**Stack**

* Vanilla JS (no framework)
* Supabase Auth + Postgres
* DataTables v2
* Responsive glass-style UI

---

## ✨ Features

### Authentication & Privacy

* **Magic-link sign-in (Supabase)** – required for any database writes
* **Optional PIN gate** – client-side privacy screen when you just want to view locally
* Session persistence with real auth state sync

### Bookkeeping

* Payments ledger with edit/delete
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

> © BLKPVNTHR LLC

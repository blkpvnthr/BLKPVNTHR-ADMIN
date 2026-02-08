/* =========================================================
    BLKPVNTHR Dashboard + Bookkeeping
    - Supabase auth (magic link)
    - Optional PIN privacy screen
    - DataTables UI
    - Payment modal: gross / net / auto
    ========================================================= */

(() => {
  "use strict";

  /* =========================
        Helpers
        ========================= */
  const el = (id) => document.getElementById(id);
  const num = (v) => {
    const n = parseFloat(String(v ?? "").replace(/,/g, ""));
    return Number.isFinite(n) ? n : 0;
  };
  const fmtMoney = (n) => "$" + (Number(n) || 0).toFixed(2);
  const setText = (n, t) => n && (n.textContent = t);
  const safeOn = (n, e, f) => n && n.addEventListener(e, f);

  /* =========================
        Config
        ========================= */
  const LS_KEYS = { pinAuthed: "blk_pin_authed" };

  const DEFAULT_RATES = {
    pension: 0.36,
    tax: 0.0543,
    ss: 0.0765,
  };

  /* =========================
        DOM
        ========================= */
  const authOverlay = el("authOverlay");
  const authMsg = el("authMsg");
  const emailInput = el("email");
  const sendLinkBtn = el("sendLinkBtn");
  const pinInput = el("pin");
  const unlockBtn = el("unlockBtn");
  const logoutBtn = el("logoutBtn");
  const app = el("app");

  const addPaymentBtn = el("addPaymentBtn");
  const snapshotMonthBtn = el("snapshotMonthBtn");

  /* =========================
        Lock / Unlock UI
        ========================= */
  function showLock() {
    document.body.classList.add("locked");
    authOverlay?.classList.remove("hidden");
    app?.classList.add("hidden");
    pinInput && (pinInput.value = "");
  }

  function showApp() {
    document.body.classList.remove("locked");
    authOverlay?.classList.add("hidden");
    app?.classList.remove("hidden");
  }

  function isPinAuthed() {
    return localStorage.getItem(LS_KEYS.pinAuthed) === "1";
  }
  function setPinAuthed() {
    localStorage.setItem(LS_KEYS.pinAuthed, "1");
  }
  function clearPinAuthed() {
    localStorage.removeItem(LS_KEYS.pinAuthed);
  }

  /* =========================
        Supabase helpers
        ========================= */
  const hasSB = () => !!window.sb;
  const hasDT = () => typeof window.DataTable !== "undefined";

  async function getUser() {
    if (!hasSB()) return null;
    const { data } = await window.sb.auth.getSession();
    return data?.session?.user || null;
  }

  async function sendMagicLink() {
    if (!hasSB()) return setText(authMsg, "Supabase not ready.");
    const email = emailInput.value.trim();
    if (!email) return setText(authMsg, "Enter your email.");
    setText(authMsg, "Sending link…");

    const { error } = await window.sb.auth.signInWithOtp({ email });
    if (error) return setText(authMsg, error.message);
    setText(authMsg, "Check your email for the sign-in link.");
  }

  async function signOut() {
    clearPinAuthed();
    hasSB() && (await window.sb.auth.signOut());
    showLock();
  }

  /* =========================
        Payment math
        ========================= */
  let modalMode = "gross"; // gross | net | auto
  let lastTyped = null;

  function getRates() {
    const p = el("rate_pension")?.value;
    const t = el("rate_tax")?.value;
    const s = el("rate_ss")?.value;

    return {
      pension: p ? num(p) / 100 : DEFAULT_RATES.pension,
      tax: t ? num(t) / 100 : DEFAULT_RATES.tax,
      ss: s ? num(s) / 100 : DEFAULT_RATES.ss,
    };
  }

  function fromGross(g) {
    if (!g) return null;
    const r = getRates();
    const pension = g * r.pension;
    const tax = g * r.tax;
    const ss = g * r.ss;
    return {
      gross: g,
      net: g - pension - tax - ss,
      pension,
      tax,
      ss,
    };
  }

  function fromNet(n) {
    if (!n) return null;
    const r = getRates();
    const sum = r.pension + r.tax + r.ss;
    if (sum >= 0.95) return null;
    return fromGross(n / (1 - sum));
  }

  function recalcModal() {
    const gEl = el("m_amount_gross");
    const nEl = el("m_amount_net");
    const pEl = el("m_pension");
    const tEl = el("m_tax_ded");
    const sEl = el("m_ss");

    const g = num(gEl?.value);
    const n = num(nEl?.value);

    let mode = modalMode;
    if (mode === "auto") {
      mode = lastTyped || (g ? "gross" : "net");
    }

    const c = mode === "gross" ? fromGross(g) : fromNet(n);
    if (!c) return;

    if (mode === "gross") nEl.value = c.net.toFixed(2);
    else gEl.value = c.gross.toFixed(2);

    pEl.value = c.pension.toFixed(2);
    tEl.value = c.tax.toFixed(2);
    sEl.value = c.ss.toFixed(2);
  }

  /* =========================
        Payment modal
        ========================= */
  let editingId = null;

  function openPaymentModal(row = null) {
    editingId = row?.id || null;

    el("paymentModalTitle").textContent = row ? "Edit Payment" : "Add Payment";
    el("m_payment_date").value =
      row?.payment_date || new Date().toISOString().slice(0, 10);
    el("m_status").value = row?.status || "received";
    el("m_memo").value = row?.memo || "";
    el("m_health").checked = !!row?.health_insurance_paid;
    el("m_edu").checked = !!row?.education_paid;

    el("m_amount_net").value = row?.amount_net || "";
    el("m_amount_gross").value = row?.amount_gross || "";

    modalMode = "auto";
    lastTyped = "net";
    recalcModal();

    el("paymentModal").classList.remove("hidden");
  }

  function closePaymentModal() {
    el("paymentModal").classList.add("hidden");
  }

  async function savePayment() {
    const user = await getUser();
    if (!user) return alert("Sign in to save.");

    recalcModal();

    const payload = {
      user_id: user.id,
      payment_date: el("m_payment_date").value,
      status: el("m_status").value,
      memo: el("m_memo").value || null,
      amount_gross: num(el("m_amount_gross").value),
      amount_net: num(el("m_amount_net").value),
      pension_deduction: num(el("m_pension").value),
      tax_deduction: num(el("m_tax_ded").value),
      ss_deduction: num(el("m_ss").value),
      health_insurance_paid: el("m_health").checked,
      education_paid: el("m_edu").checked,
    };

    if (!payload.amount_gross) return alert("Invalid payment amount.");

    const q = editingId
      ? window.sb.from("apl_payments").update(payload).eq("id", editingId)
      : window.sb.from("apl_payments").insert(payload);

    const { error } = await q;
    if (error) return alert(error.message);

    closePaymentModal();
    refreshBookkeeping();
  }

  /* =========================
        Tables
        ========================= */
  let paymentsDT;

  async function refreshBookkeeping() {
    const { data } = await window.sb
      .from("apl_payments")
      .select("*")
      .order("payment_date", { ascending: false });

    paymentsDT.clear().rows.add(data).draw();
  }

  function initPaymentsTable(rows) {
    paymentsDT = new DataTable("#paymentsTable", {
      data: rows,
      columns: [
        { data: "payment_date" },
        { data: "amount_gross", render: fmtMoney },
        { data: "pension_deduction", render: fmtMoney },
        { data: "tax_deduction", render: fmtMoney },
        { data: "ss_deduction", render: fmtMoney },
        { data: "health_insurance_paid", render: (d) => (d ? "✓" : "") },
        { data: "education_paid", render: (d) => (d ? "✓" : "") },
        { data: "status" },
        { data: "memo" },
        {
          data: null,
          orderable: false,
          render: (_, __, r) => `
        <button class="dt-btn" data-action="edit" data-id="${r.id}">Edit</button>
        <button class="dt-btn danger" data-action="del" data-id="${r.id}">Delete</button>
        `,
        },
      ],
    });

    el("paymentsTable").addEventListener("click", (e) => {
      const btn = e.target.closest(".dt-btn");
      if (!btn) return;

      const id = btn.dataset.id;
      const action = btn.dataset.action;

      if (action === "edit") {
        window.sb
          .from("apl_payments")
          .select("*")
          .eq("id", id)
          .single()
          .then(({ data }) => openPaymentModal(data));
      }

      if (action === "del") {
        if (!confirm("Delete this payment?")) return;

        window.sb
          .from("apl_payments")
          .delete()
          .eq("id", id)
          .then(() => refreshBookkeeping());
      }
    });
  }

  /* =========================
        Init
        ========================= */
  async function snapshotSelectedMonth() {
    const month = el("monthSelect")?.value;
    if (!month) return alert("Pick a month first.");

    const { data, error } = await window.sb
      .from("apl_monthly_live")
      .select("*")
      .eq("month", month)
      .limit(1)
      .maybeSingle();

    if (error) throw error;
    if (!data) return alert("No live data for that month yet.");

    const { error: insErr } = await window.sb.from("monthly_totals").insert({
      month: data.month,
      total_gross: data.total_gross,
      payment_count: data.payment_count,
    });

    if (insErr) throw insErr;
    await refreshBookkeeping();
  }

  document.addEventListener("DOMContentLoaded", async () => {
    const toggle = document.querySelector(".nav-toggle");
    const menu = document.querySelector(".nav-menu");

    toggle?.addEventListener("click", () => {
      menu.classList.toggle("open");
    });

    // ===== MAIN BUTTON BINDINGS =====
    safeOn(addPaymentBtn, "click", () => openPaymentModal());

    safeOn(snapshotMonthBtn, "click", () => {
      snapshotSelectedMonth().catch((err) =>
        alert(err?.message || String(err)),
      );
    });

    safeOn(el("paymentModalSave"), "click", () =>
      savePayment().catch((err) => alert(err?.message || String(err))),
    );

    safeOn(el("paymentModalCancel"), "click", closePaymentModal);
    safeOn(el("paymentModalClose"), "click", closePaymentModal);

    // close when clicking backdrop
    safeOn(el("paymentModal"), "click", (e) => {
      if (e.target.id === "paymentModal") closePaymentModal();
    });
    safeOn(sendLinkBtn, "click", sendMagicLink);
    safeOn(unlockBtn, "click", () => {
      setPinAuthed();
      showApp();
    });
    safeOn(logoutBtn, "click", signOut);

    ["m_amount_gross", "m_amount_net"].forEach((id) =>
      safeOn(el(id), "input", (e) => {
        lastTyped = id.includes("gross") ? "gross" : "net";
        recalcModal();
      }),
    );
    safeOn(el("pay_mode"), "change", (e) => (modalMode = e.target.value));
    safeOn(el("paymentModalSave"), "click", savePayment);
    safeOn(el("paymentModalCancel"), "click", closePaymentModal);

    const user = await getUser();
    if (user) {
      showApp();
    } else if (isPinAuthed()) {
      showApp();
    } else {
      showLock();
    }

    if (user && hasDT()) {
      const { data } = await window.sb.from("apl_payments").select("*");
      initPaymentsTable(data || []);
    }

    window.sb.auth.onAuthStateChange(async (_e, s) => {
      s?.user ? showApp() : showLock();
    });
  });
})();

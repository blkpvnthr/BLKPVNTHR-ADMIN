/* =========================================================
   BLKPVNTHR Dashboard + Bookkeeping
   - Supabase auth (magic link)
   - Optional PIN privacy screen
   - DataTables UI
   - Payment modal: gross / net / auto
   - AUTO-CALC: pension + fed/md/moco + ss/med + overall tax %
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

  // Effective rates (your table)
  const DEFAULT_RATES = {
    pension: 0.36,

    // Income tax components (effective %)
    fed: 0.087,   // 8.7%
    md: 0.045,    // 4.5%
    moco: 0.032,  // 3.2%

    // Payroll
    ss: 0.0765,   // 7.65%
  };

  /* =========================
        Supabase Config
     ========================= */
  const PROJECT_URL = "https://xtespuzbuepublmiciyf.supabase.co";
  const PUBLISHABLE_KEY = "sb_publishable_giqkrVUBnF8PS-7sWS_WaQ_2lk-8wOL";

  function ensureSupabaseClient() {
    if (window.sb) return;
    if (!window.supabase?.createClient) {
      console.error("Supabase JS not loaded. Include @supabase/supabase-js@2");
      return;
    }
    window.sb = window.supabase.createClient(PROJECT_URL, PUBLISHABLE_KEY);
    console.log("✅ Supabase initialized:", PROJECT_URL);
  }

  // Handles BOTH PKCE (?code=...) and older hash token (#access_token=...)
  async function handleMagicLinkReturn() {
    ensureSupabaseClient();
    if (!window.sb) return;

    const url = new URL(window.location.href);
    const code = url.searchParams.get("code");

    try {
      if (code) {
        const { error } = await window.sb.auth.exchangeCodeForSession(
          window.location.href,
        );
        if (error) console.error("exchangeCodeForSession error:", error);

        url.searchParams.delete("code");
        history.replaceState({}, document.title, url.toString());
        return;
      }

      if (window.location.hash.includes("access_token")) {
        await window.sb.auth.getSession();
        history.replaceState({}, document.title, window.location.pathname);
      }
    } catch (e) {
      console.error("Magic link handling failed:", e);
    }
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
    ensureSupabaseClient();
    if (!hasSB()) return setText(authMsg, "Supabase not ready.");

    const email = emailInput?.value?.trim() || "";
    if (!email) return setText(authMsg, "Enter your email.");

    setText(authMsg, "Sending link…");
    const emailRedirectTo = `${window.location.origin}${window.location.pathname}`;

    try {
      const { error } = await window.sb.auth.signInWithOtp({
        email,
        options: { emailRedirectTo },
      });
      if (error) return setText(authMsg, error.message);
      setText(authMsg, "Check your email for the sign-in link.");
    } catch (e) {
      console.error(e);
      setText(authMsg, "Network error. Can’t reach Supabase.");
    }
  }

  async function signOut() {
    clearPinAuthed();
    if (hasSB()) await window.sb.auth.signOut();
    showLock();
  }

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
    if (pinInput) pinInput.value = "";
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
        Payment math (AUTO)
     ========================= */
  let modalMode = "auto"; // gross | net | auto
  let lastTyped = null;   // "gross" | "net"

  function clampRate(r) {
    return Math.max(0, Math.min(0.95, r));
  }

  // Read optional overrides (still allowed), but all deductions are auto-calculated
  // rate_tax = combined income tax (fed+md+moco), rate_ss = ss/med, rate_pension = pension
  function getRates() {
    const p = el("rate_pension")?.value;
    const t = el("rate_tax")?.value; // combined income tax override
    const s = el("rate_ss")?.value;

    const pension = clampRate(p ? num(p) / 100 : DEFAULT_RATES.pension);
    const ss = clampRate(s ? num(s) / 100 : DEFAULT_RATES.ss);

    const defaultIncomeTax = DEFAULT_RATES.fed + DEFAULT_RATES.md + DEFAULT_RATES.moco;
    const incomeTaxCombined = clampRate(t ? num(t) / 100 : defaultIncomeTax);

    // Split combined income tax into components using default proportions
    const wFed = DEFAULT_RATES.fed / defaultIncomeTax;
    const wMd = DEFAULT_RATES.md / defaultIncomeTax;
    const wMo = DEFAULT_RATES.moco / defaultIncomeTax;

    return {
      pension,
      ss,
      fed: incomeTaxCombined * wFed,
      md: incomeTaxCombined * wMd,
      moco: incomeTaxCombined * wMo,
      incomeTaxCombined,
    };
  }

  function calcFromGross(g) {
    if (!g) return null;
    const r = getRates();

    const pension = g * r.pension;

    const fedTax = g * r.fed;
    const mdTax = g * r.md;
    const mocoTax = g * r.moco;
    const incomeTaxTotal = fedTax + mdTax + mocoTax;

    const ss = g * r.ss;

    const totalTaxes = incomeTaxTotal + ss; // taxes (not pension)
    const overallTaxPct = totalTaxes / g;   // per-payment effective

    const net = g - pension - totalTaxes;

    return {
      gross: g,
      net,
      pension,
      // keep DB fields:
      tax: incomeTaxTotal, // combined income tax (fed+md+moco)
      ss,
      // extra breakdown + overall:
      fedTax,
      mdTax,
      mocoTax,
      totalTaxes,
      overallTaxPct,
      rates: r,
    };
  }

  function calcFromNet(n) {
    if (!n) return null;
    const r = getRates();
    const incomeTax = r.fed + r.md + r.moco;
    const sum = r.pension + incomeTax + r.ss;
    if (sum >= 0.95) return null;

    const gross = n / (1 - sum);
    return calcFromGross(gross);
  }

  function setReadOnlyCalculatedFields() {
    // Make computed fields read-only so users only type Gross or Net
    ["m_pension", "m_tax_ded", "m_ss"].forEach((id) => {
      const x = el(id);
      if (x) {
        x.readOnly = true;
        x.setAttribute("aria-readonly", "true");
      }
    });
  }

  function updateReconcileUI(c) {
    const box = el("reconcileBox");
    const msg = el("reconcileMsg");
    if (!box || !msg || !c) return;

    const overallPct = (c.overallTaxPct * 100).toFixed(2);

    msg.textContent =
      `Fed ${fmtMoney(c.fedTax)} · MD ${fmtMoney(c.mdTax)} · MoCo ${fmtMoney(c.mocoTax)} · ` +
      `SS/Med ${fmtMoney(c.ss)} = Taxes ${fmtMoney(c.totalTaxes)} (${overallPct}% of gross)`;

    // simple state coloring
    box.classList.remove("neutral", "good", "bad");
    box.classList.add("good");
  }

  function recalcModal() {
    const gEl = el("m_amount_gross");
    const nEl = el("m_amount_net");
    const pEl = el("m_pension");
    const tEl = el("m_tax_ded");
    const sEl = el("m_ss");

    if (!gEl || !nEl || !pEl || !tEl || !sEl) return;

    const g = num(gEl.value);
    const n = num(nEl.value);

    // Decide mode
    let mode = modalMode;
    if (mode === "auto") {
      mode = lastTyped || (g ? "gross" : "net");
    }

    const c = mode === "gross" ? calcFromGross(g) : calcFromNet(n);
    if (!c) return;

    // Always sync both amount fields to computed values
    gEl.value = c.gross.toFixed(2);
    nEl.value = c.net.toFixed(2);

    // Auto-calculated deductions
    pEl.value = c.pension.toFixed(2);
    tEl.value = c.tax.toFixed(2); // combined income tax (fed+md+moco)
    sEl.value = c.ss.toFixed(2);

    updateReconcileUI(c);
  }

  /* =========================
        Payment modal CRUD
     ========================= */
  let editingId = null;

  function openPaymentModal(row = null) {
    editingId = row?.id || null;

    const title = el("paymentModalTitle");
    if (title) title.textContent = row ? "Edit Payment" : "Add Payment";

    el("m_payment_date").value =
      row?.payment_date || new Date().toISOString().slice(0, 10);
    el("m_status").value = row?.status || "received";
    el("m_memo").value = row?.memo || "";
    el("m_health").checked = !!row?.health_insurance_paid;
    el("m_edu").checked = !!row?.education_paid;

    // Load amounts (deductions will be recalculated)
    el("m_amount_net").value = row?.amount_net ?? "";
    el("m_amount_gross").value = row?.amount_gross ?? "";

    modalMode = "auto";
    lastTyped = row?.amount_gross ? "gross" : "net";

    setReadOnlyCalculatedFields();
    recalcModal();

    el("paymentModal")?.classList.remove("hidden");
  }

  function closePaymentModal() {
    el("paymentModal")?.classList.add("hidden");
  }

  async function savePayment() {
    const user = await getUser();
    if (!user) return alert("Sign in to save.");

    // Ensure latest calculation is applied before saving
    recalcModal();

    const payload = {
      user_id: user.id,
      payment_date: el("m_payment_date").value,
      status: el("m_status").value,
      memo: el("m_memo").value || null,

      amount_gross: num(el("m_amount_gross").value),
      amount_net: num(el("m_amount_net").value),

      // these are AUTO-CALCULATED
      pension_deduction: num(el("m_pension").value),
      tax_deduction: num(el("m_tax_ded").value), // fed+md+moco combined
      ss_deduction: num(el("m_ss").value),

      health_insurance_paid: el("m_health").checked,
      education_paid: el("m_edu").checked,
    };

    if (!payload.amount_gross) return alert("Invalid payment amount.");

    const q = editingId
      ? window.sb
          .from("apl_payments")
          .update(payload)
          .eq("id", editingId)
          .eq("user_id", user.id)
      : window.sb.from("apl_payments").insert(payload);

    const { error } = await q;
    if (error) return alert(error.message);

    closePaymentModal();
    await refreshBookkeeping();
  }

  /* =========================
        Tables
     ========================= */
  let paymentsDT;

  async function refreshBookkeeping() {
  const user = await getUser();
  if (!user || !paymentsDT) return;

  const start = new Date();
  start.setDate(1);
  start.setHours(0, 0, 0, 0);

  const end = new Date(start);
  end.setMonth(end.getMonth() + 1);

  const startISO = start.toISOString().slice(0, 10);
  const endISO = end.toISOString().slice(0, 10);

  const { data, error } = await window.sb
    .from("apl_payments")
    .select("*")
    .eq("user_id", user.id)
    .gte("payment_date", startISO)
    .lt("payment_date", endISO)
    .order("payment_date", { ascending: false });

  if (error) {
    console.error(error);
    return;
  }

  paymentsDT.clear().rows.add(data || []).draw();
}
  function initPaymentsTable(rows) {
    paymentsDT = new DataTable("#paymentsTable", {
      data: rows,
      order: [[0, "desc"]],
      scrollX: true,
      autoWidth: false,
      columnDefs: [
        { targets: 0, className: "dt-body-nowrap" },  // date
        { targets: 10, orderable: false },             // actions
      ],
      columns: [
  { data: "payment_date" },
  { data: "amount_net", render: fmtMoney },     // ✅ Net (new)
  { data: "amount_gross", render: fmtMoney },   // Gross
  { data: "pension_deduction", render: fmtMoney },
  { data: "tax_deduction", render: fmtMoney },
  { data: "ss_deduction", render: fmtMoney },
  { data: "health_insurance_paid", render: (d) => (d ? "✓" : "") },
  { data: "education_paid", render: (d) => (d ? "✓" : "") },
  { data: "status" },
  { data: "memo" },
  {
    data: null,
    render: (_, __, r) => `
      <button class="dt-btn" data-action="edit" data-id="${r.id}">Edit</button>
      <button class="dt-btn danger" data-action="del" data-id="${r.id}">Delete</button>
    `,
  },
],
    });

    el("paymentsTable")?.addEventListener("click", async (e) => {
      const btn = e.target.closest(".dt-btn");
      if (!btn) return;

      const user = await getUser();
      if (!user) return alert("Sign in first.");

      const id = btn.dataset.id;
      const action = btn.dataset.action;

      if (action === "edit") {
        const { data, error } = await window.sb
          .from("apl_payments")
          .select("*")
          .eq("id", id)
          .eq("user_id", user.id)
          .single();

        if (error) return alert(error.message);
        openPaymentModal(data);
      }

      if (action === "del") {
        if (!confirm("Delete this payment?")) return;

        const { error } = await window.sb
          .from("apl_payments")
          .delete()
          .eq("id", id)
          .eq("user_id", user.id);

        if (error) return alert(error.message);
        await refreshBookkeeping();
      }
    });
  }

  /* =========================
        Snapshot (optional)
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

  /* =========================
   TTM Accordion (by month)
   - Shows ALL months (history)
   - Accordion behavior: only one open
   - Newest month opens by default
   - Table initializes on-demand
   ========================= */

function monthKey(dateStr) {
  // payment_date expected "YYYY-MM-DD"
  return String(dateStr || "").slice(0, 7); // "YYYY-MM"
}

function monthLabel(yyyyMm) {
  // "2026-02" -> "Feb 2026"
  const [y, m] = String(yyyyMm).split("-");
  const d = new Date(Number(y), Number(m) - 1, 1);
  return d.toLocaleString(undefined, { month: "short", year: "numeric" });
}

function sum(arr, pick) {
  return arr.reduce((acc, x) => acc + (Number(pick(x)) || 0), 0);
}

async function initTTMAccordion(user) {
  const host = document.getElementById("ttmAccordion");
  if (!host) return; // not on TTM page

  // Pull ALL payments (history) for this user
  const { data, error } = await window.sb
    .from("apl_payments")
    .select("*")
    .eq("user_id", user.id)
    .order("payment_date", { ascending: false });

  if (error) {
    host.innerHTML = `<div class="tiny muted">Error: ${error.message}</div>`;
    return;
  }

  const rows = data || [];
  if (!rows.length) {
    host.innerHTML = `<div class="tiny muted">No payments yet.</div>`;
    return;
  }

  // Group by month (newest -> oldest because rows are ordered desc)
  const groups = new Map();
  for (const r of rows) {
    const k = monthKey(r.payment_date);
    if (!k) continue;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }

  host.innerHTML = "";

  let firstDetails = null;

  for (const [k, monthRows] of groups.entries()) {
    const gross = sum(monthRows, (x) => x.amount_gross);
    const net = sum(monthRows, (x) => x.amount_net);
    const pension = sum(monthRows, (x) => x.pension_deduction);
    const tax = sum(monthRows, (x) => x.tax_deduction);
    const ss = sum(monthRows, (x) => x.ss_deduction);

    const tableId = `ttmTable_${k.replace("-", "")}`;

    const details = document.createElement("details");
    details.className = "ttm-item";
    details.dataset.tableId = tableId;

    details.innerHTML = `
      <summary>
        <div class="ttm-left">
          <div class="ttm-month">${monthLabel(k)}</div>
          <div class="ttm-meta">${monthRows.length} payment(s)</div>
        </div>
        <div class="ttm-right">
          <div class="ttm-total">${fmtMoney(gross)}</div>
          <div class="ttm-meta">Gross</div>
        </div>
      </summary>

      <div class="ttm-body">
        <div class="ttm-kpis">
          <div class="ttm-kpi">
            <div class="label">Total Gross</div>
            <div class="val">${fmtMoney(gross)}</div>
          </div>
          <div class="ttm-kpi">
            <div class="label">Total Net (Deposits)</div>
            <div class="val">${fmtMoney(net)}</div>
          </div>
          <div class="ttm-kpi">
            <div class="label">Total Deductions</div>
            <div class="val">${fmtMoney(pension + tax + ss)}</div>
          </div>
        </div>

        <div class="table-wrap">
          <table id="${tableId}" class="display" style="width:100%">
            <thead>
              <tr>
                <th>Date</th>
                <th>Gross</th>
                <th>Net</th>
                <th>Pension</th>
                <th>Income Tax</th>
                <th>SS/Med</th>
                <th>Status</th>
                <th>Memo</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    `;

    host.appendChild(details);
    if (!firstDetails) firstDetails = details;

    // Accordion behavior + lazy table init
    details.addEventListener("toggle", () => {
      if (!details.open) return;

      // close others
      host.querySelectorAll("details.ttm-item").forEach((d) => {
        if (d !== details) d.open = false;
      });

      // init table once
      if (details.dataset.inited === "1") return;
      details.dataset.inited = "1";

      const tableEl = document.getElementById(tableId);
      if (!tableEl || !hasDT()) return;

      new DataTable(tableEl, {
        data: monthRows,
        order: [[0, "desc"]],
        scrollX: true,
        autoWidth: false,
        columns: [
          { data: "payment_date" },
          { data: "amount_gross", render: fmtMoney },
          { data: "amount_net", render: fmtMoney },
          { data: "pension_deduction", render: fmtMoney },
          { data: "tax_deduction", render: fmtMoney },
          { data: "ss_deduction", render: fmtMoney },
          { data: "status" },
          { data: "memo" },
        ],
      });
    });
  }

  // Open newest month by default
  if (firstDetails) firstDetails.open = true;
}

  /* =========================
        Init
     ========================= */
  document.addEventListener("DOMContentLoaded", async () => {
    ensureSupabaseClient();
    await handleMagicLinkReturn();

    // UI bindings
    safeOn(sendLinkBtn, "click", sendMagicLink);
    safeOn(unlockBtn, "click", () => {
      setPinAuthed();
      showApp();
    });
    safeOn(logoutBtn, "click", signOut);

    safeOn(addPaymentBtn, "click", () => openPaymentModal());
    safeOn(snapshotMonthBtn, "click", () =>
      snapshotSelectedMonth().catch((err) => alert(err?.message || String(err))),
    );

    safeOn(el("paymentModalSave"), "click", () =>
      savePayment().catch((err) => alert(err?.message || String(err))),
    );
    safeOn(el("paymentModalCancel"), "click", closePaymentModal);
    safeOn(el("paymentModalClose"), "click", closePaymentModal);

    safeOn(el("paymentModal"), "click", (e) => {
      if (e.target.id === "paymentModal") closePaymentModal();
    });

    // When user types gross/net, auto-recalc everything
    ["m_amount_gross", "m_amount_net"].forEach((id) =>
      safeOn(el(id), "input", () => {
        lastTyped = id.includes("gross") ? "gross" : "net";
        recalcModal();
      }),
    );

    // Mode chooser still works (gross-first / net-first / auto)
    safeOn(el("pay_mode"), "change", (e) => {
      modalMode = e.target.value;
      recalcModal();
    });

    // If you adjust override rates, recompute immediately
    ["rate_pension", "rate_tax", "rate_ss"].forEach((id) =>
      safeOn(el(id), "input", () => recalcModal()),
    );

    const user = await getUser();

    // Init TTM accordion if this page has it
    if (user) {
      initTTMAccordion(user).catch((err) =>
        console.error("TTM init failed:", err),
      );
    }

    if (user) showApp();
    else if (isPinAuthed()) showApp();
    else showLock();

    // Init payments table
    if (user && hasDT() && el("paymentsTable")) {
      const start = new Date();
      start.setDate(1);
      start.setHours(0, 0, 0, 0);

      const end = new Date(start);
      end.setMonth(end.getMonth() + 1);

      const startISO = start.toISOString().slice(0, 10);
      const endISO = end.toISOString().slice(0, 10);

      const { data, error } = await window.sb
        .from("apl_payments")
        .select("*")
        .eq("user_id", user.id)
        .gte("payment_date", startISO)
        .lt("payment_date", endISO)
        .order("payment_date", { ascending: false });

      if (error) console.error(error);
      initPaymentsTable(data || []);
    }

    // Auth changes
    if (hasSB()) {
      window.sb.auth.onAuthStateChange((_e, s) => {
        if (s?.user) showApp();
        else showLock();
      });
    }
  });
})();
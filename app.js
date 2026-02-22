/* =========================================================
   BLKPVNTHR Dashboard + Bookkeeping (Stipends)
   - Supabase auth (magic link)
   - Optional PIN privacy screen
   - DataTables UI
   - Payment modal: gross / net / auto
   - AUTO-CALC (San Antonio, TX – Stipends):
       • Suggested withholding (default 18%)
       • Retirement (renamed from pension; default 0%)
       • Budget allocations (of NET after withholding/retirement):
           - Housing 34%
           - Transportation 23%
           - Living 29%
           - Buffer 13%
       • NEW: Half of Buffer auto-moves into Retirement
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

  // Stipends: no payroll withholding by employer — we show a suggested amount to set aside.
  // Default suggested withholding = 18% (overrideable).
  // Retirement = renamed from pension; default 0% (overrideable).
  // Budget allocations apply to NET AFTER withholding + retirement set-asides.
  // NEW: Half of Buffer auto-moves into Retirement each time.
  const DEFAULT_RATES = {
    retirement: 0.0, // base retirement % (default 0)

    // Suggested tax withholding to set aside from each stipend
    withhold: 0.18, // 18%

    // Budget allocations of net-after-set-asides (pre buffer-split)
    allocHousing: 0.34,   // 34%
    allocTransport: 0.29, // 32%
    allocLiving: 0.23,    // 20%
    allocBuffer: 0.13,    // 13% (then half transfers to retirement)

    // Keep legacy field if your table still has ss_deduction column:
    payroll: 0.0, // 0 for stipend budgeting (not modeled as payroll taxes)
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

  // Robust session read
  async function getUser() {
    ensureSupabaseClient();
    if (!hasSB()) return null;

    try {
      const { data, error } = await window.sb.auth.getSession();
      if (error) {
        console.warn("getSession error:", error);
        return null;
      }
      return data?.session?.user || null;
    } catch (e) {
      console.warn("getUser exception:", e);
      return null;
    }
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
        Stipend math (AUTO)
        - Suggested withholding (default 18%)
        - Retirement (default 0%)
        - Allocations computed from remainder
        - NEW: Half of Buffer transfers to Retirement
     ========================= */

  let modalMode = "auto"; // gross | net | auto
  let lastTyped = "gross"; // enter stipend amount as gross

  function clampRate(r) {
    return Math.max(0, Math.min(0.95, r));
  }

  function clampAlloc(r) {
    return Math.max(0, Math.min(1.0, r));
  }

  function round2(n) {
    return Math.round((Number(n) || 0) * 100) / 100;
  }

  // ---- Field ID helpers (so you can rename HTML gradually) ----
  const getRetirementEl = () => el("m_retirement") || el("m_pension"); // fallback
  const getRateRetirementEl = () => el("rate_retirement") || el("rate_pension"); // fallback
  const getWithholdEl = () => el("m_tax_ded"); // keep same id, repurpose label in HTML to "Suggested Withholding"
  const getRateWithholdEl = () => el("rate_withhold") || el("rate_tax"); // fallback (old rate_tax)

  const getHousingEl = () => el("m_housing");
  const getTransportEl = () => el("m_transport");
  const getLivingEl = () => el("m_living");
  const getBufferEl = () => el("m_buffer");

  function getRates() {
    const rRet = getRateRetirementEl()?.value;
    const rWith = getRateWithholdEl()?.value;

    const ah = el("rate_alloc_housing")?.value;
    const at = el("rate_alloc_transport")?.value;
    const al = el("rate_alloc_living")?.value;
    const ab = el("rate_alloc_buffer")?.value;

    const retirement = clampRate(rRet ? num(rRet) / 100 : DEFAULT_RATES.retirement);
    const withhold = clampRate(rWith ? num(rWith) / 100 : DEFAULT_RATES.withhold);

    let allocHousing = clampAlloc(ah ? num(ah) / 100 : DEFAULT_RATES.allocHousing);
    let allocTransport = clampAlloc(at ? num(at) / 100 : DEFAULT_RATES.allocTransport);
    let allocLiving = clampAlloc(al ? num(al) / 100 : DEFAULT_RATES.allocLiving);
    let allocBuffer = clampAlloc(ab ? num(ab) / 100 : DEFAULT_RATES.allocBuffer);

    const allocSum = allocHousing + allocTransport + allocLiving + allocBuffer;
    if (allocSum > 0 && Math.abs(1 - allocSum) > 0.02) {
      allocHousing /= allocSum;
      allocTransport /= allocSum;
      allocLiving /= allocSum;
      allocBuffer /= allocSum;
    }

    return {
      retirement,
      withhold,
      allocHousing,
      allocTransport,
      allocLiving,
      allocBuffer,
    };
  }

  /**
   * NEW behavior:
   * - compute base retirement % from gross (often 0)
   * - compute suggested withholding from gross
   * - compute remainder1 = gross - withholding - baseRetirement
   * - compute housing/transport/living from remainder1
   * - bufferRaw = leftover remainder1 - (housing+transport+living)
   * - transfer 50% of bufferRaw into retirement
   * - bufferFinal = bufferRaw - transfer
   * - retirementTotal = baseRetirement + transfer
   * - netToBudget = gross - withholding - retirementTotal
   *   and equals housing+transport+living+bufferFinal
   */
  function calcFromGross(g) {
    if (!g) return null;

    const r = getRates();

    const suggestedWithhold = round2(g * r.withhold);
    const baseRetirement = round2(g * r.retirement);

    const remainder1 = round2(g - suggestedWithhold - baseRetirement);
    if (remainder1 < 0) return null;

    const housing = round2(remainder1 * r.allocHousing);
    const transport = round2(remainder1 * r.allocTransport);
    const living = round2(remainder1 * r.allocLiving);

    // buffer as leftover to avoid drift
    let bufferRaw = round2(remainder1 - housing - transport - living);
    if (bufferRaw < 0) bufferRaw = 0;

    const retirementFromBuffer = round2(bufferRaw * 0.5);
    const buffer = round2(bufferRaw - retirementFromBuffer);
    const retirement = round2(baseRetirement + retirementFromBuffer);

    const net = round2(g - suggestedWithhold - retirement);

    const overallWithholdPct = g ? suggestedWithhold / g : 0;

    return {
      gross: round2(g),
      net,

      suggestedWithhold,
      retirement, // ✅ includes half-buffer transfer

      // extra transparency:
      baseRetirement,
      bufferRaw,
      retirementFromBuffer,

      // allocations that sum to net
      housing,
      transport,
      living,
      buffer,

      overallWithholdPct,
      rates: r,
    };
  }

  // If user types "net you want to have after set-asides", compute required gross.
  // We approximate using: net = gross*(1-withhold-retRate) * (1 - 0.5*bufferShare)
  function calcFromNet(n) {
    if (!n) return null;

    const r = getRates();
    const denom = (1 - r.withhold - r.retirement) * (1 - 0.5 * r.allocBuffer);
    if (denom <= 0.05) return null;

    const gross = n / denom;
    return calcFromGross(gross);
  }

  function setReadOnlyCalculatedFields() {
    const ids = [
      "m_retirement",
      "m_pension", // fallback
      "m_tax_ded",
      "m_housing",
      "m_transport",
      "m_living",
      "m_buffer",
      "m_ss", // legacy
    ];

    ids.forEach((id) => {
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

    const pct = (c.overallWithholdPct * 100).toFixed(2);

    msg.textContent =
      `Suggested Withhold ${fmtMoney(c.suggestedWithhold)} (${pct}% of stipend) · ` +
      `Retirement ${fmtMoney(c.retirement)} (includes Buffer→Ret ${fmtMoney(c.retirementFromBuffer)}) · ` +
      `Net to budget ${fmtMoney(c.net)} · ` +
      `Housing ${fmtMoney(c.housing)} · Transport ${fmtMoney(c.transport)} · ` +
      `Living ${fmtMoney(c.living)} · Buffer ${fmtMoney(c.buffer)}`;

    box.classList.remove("neutral", "good", "bad");
    box.classList.add("good");
  }

  function recalcModal() {
    const gEl = el("m_amount_gross");
    const nEl = el("m_amount_net");

    const retEl = getRetirementEl();
    const wEl = getWithholdEl();

    const hEl = getHousingEl();
    const tEl = getTransportEl();
    const lEl = getLivingEl();
    const bEl = getBufferEl();

    if (!gEl || !nEl || !retEl || !wEl) return;

    const g = num(gEl.value);
    const n = num(nEl.value);

    let mode = modalMode;
    if (mode === "auto") {
      mode = lastTyped || (g ? "gross" : "net");
    }

    const c = mode === "gross" ? calcFromGross(g) : calcFromNet(n);
    if (!c) return;

    gEl.value = c.gross.toFixed(2);
    nEl.value = c.net.toFixed(2);

    // Set-asides
    retEl.value = c.retirement.toFixed(2);        // ✅ retirement includes half-buffer transfer
    wEl.value = c.suggestedWithhold.toFixed(2);   // ✅ suggested withholding

    // Allocations
    if (hEl) hEl.value = c.housing.toFixed(2);
    if (tEl) tEl.value = c.transport.toFixed(2);
    if (lEl) lEl.value = c.living.toFixed(2);
    if (bEl) bEl.value = c.buffer.toFixed(2);    // ✅ buffer is remaining half

    // Legacy field: keep 0 for stipends
    const ssEl = el("m_ss");
    if (ssEl) ssEl.value = (0).toFixed(2);

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

    el("m_amount_net").value = row?.amount_net ?? "";
    el("m_amount_gross").value = row?.amount_gross ?? "";

    modalMode = "auto";
    lastTyped = row?.amount_gross ? "gross" : (row?.amount_net ? "net" : "gross");

    setReadOnlyCalculatedFields();
    recalcModal();

    el("paymentModal")?.classList.remove("hidden");
  }

  function closePaymentModal() {
    el("paymentModal")?.classList.add("hidden");
  }

  async function savePayment() {
    const user = await getUser();
    if (!user) {
      showLock();
      setText(authMsg, "Please sign in with the magic link to save payments.");
      return;
    }

    recalcModal();

    const g = num(el("m_amount_gross")?.value);
    const c = calcFromGross(g);
    if (!c) return alert("Invalid payment amount.");

    const payload = {
      user_id: user.id,
      payment_date: el("m_payment_date").value,
      status: el("m_status").value,
      memo: el("m_memo").value || null,

      amount_gross: c.gross,
      amount_net: c.net, // ✅ net-to-budget after buffer->retirement transfer

      // keep schema usage:
      retirement_deduction: c.retirement,          // ✅ total retirement (base + half-buffer)
      tax_deduction: c.suggestedWithhold,          // ✅ suggested withholding set-aside
      ss_deduction: 0,                             // stipends: 0

      housing_alloc: c.housing,
      transport_alloc: c.transport,
      living_alloc: c.living,
      buffer_alloc: c.buffer,                      // ✅ remaining half-buffer

      health_insurance_paid: el("m_health").checked,
      education_paid: el("m_edu").checked,
    };

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
        { targets: 0, className: "dt-body-nowrap" },
        { targets: -1, orderable: false },
      ],
      columns: [
        { data: "payment_date" },
        { data: "amount_gross", render: fmtMoney }, // Stipend
        { data: "tax_deduction", render: fmtMoney, title: "Suggested Withhold" },
        { data: "retirement_deduction", render: fmtMoney, title: "Retirement" },
        { data: "amount_net", render: fmtMoney, title: "Net to Budget" },
        { data: "housing_alloc", render: fmtMoney, title: "Housing" },
        { data: "transport_alloc", render: fmtMoney, title: "Transport" },
        { data: "living_alloc", render: fmtMoney, title: "Living" },
        { data: "buffer_alloc", render: fmtMoney, title: "Buffer" },
        { data: "health_insurance_paid", render: (d) => (d ? "✓" : "") },
        { data: "education_paid", render: (d) => (d ? "✓" : "") },
        { data: "status" },
        { data: "memo" },
        {
          data: null,
          title: "Actions",
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
      if (!user) {
        showLock();
        setText(authMsg, "Please sign in to edit or delete payments.");
        return;
      }

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
   ========================= */

  function monthKey(dateStr) {
    return String(dateStr || "").slice(0, 7);
  }

  function monthLabel(yyyyMm) {
    const [y, m] = String(yyyyMm).split("-");
    const d = new Date(Number(y), Number(m) - 1, 1);
    return d.toLocaleString(undefined, { month: "short", year: "numeric" });
  }

  function sum(arr, pick) {
    return arr.reduce((acc, x) => acc + (Number(pick(x)) || 0), 0);
  }

  async function initTTMAccordion(user) {
    const host = document.getElementById("ttmAccordion");
    if (!host) return;

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
      const withhold = sum(monthRows, (x) => x.tax_deduction);
      const retirement = sum(monthRows, (x) => x.retirement_deduction);

      const housing = sum(monthRows, (x) => x.housing_alloc);
      const transport = sum(monthRows, (x) => x.transport_alloc);
      const living = sum(monthRows, (x) => x.living_alloc);
      const buffer = sum(monthRows, (x) => x.buffer_alloc);

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
            <div class="ttm-meta">Stipends</div>
          </div>
        </summary>

        <div class="ttm-body">
          <div class="ttm-kpis">
            <div class="ttm-kpi"><div class="label">Total Stipends</div><div class="val">${fmtMoney(gross)}</div></div>
            <div class="ttm-kpi"><div class="label">Suggested Withhold</div><div class="val">${fmtMoney(withhold)}</div></div>
            <div class="ttm-kpi"><div class="label">Retirement</div><div class="val">${fmtMoney(retirement)}</div></div>
            <div class="ttm-kpi"><div class="label">Net to Budget</div><div class="val">${fmtMoney(net)}</div></div>
          </div>

          <div class="ttm-kpis" style="margin-top:10px">
            <div class="ttm-kpi"><div class="label">Housing</div><div class="val">${fmtMoney(housing)}</div></div>
            <div class="ttm-kpi"><div class="label">Transport</div><div class="val">${fmtMoney(transport)}</div></div>
            <div class="ttm-kpi"><div class="label">Living</div><div class="val">${fmtMoney(living)}</div></div>
            <div class="ttm-kpi"><div class="label">Buffer</div><div class="val">${fmtMoney(buffer)}</div></div>
          </div>

          <div class="table-wrap">
            <table id="${tableId}" class="display" style="width:100%">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Stipend</th>
                  <th>Suggested Withhold</th>
                  <th>Retirement</th>
                  <th>Net to Budget</th>
                  <th>Housing</th>
                  <th>Transport</th>
                  <th>Living</th>
                  <th>Buffer</th>
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

      details.addEventListener("toggle", () => {
        if (!details.open) return;

        host.querySelectorAll("details.ttm-item").forEach((d) => {
          if (d !== details) d.open = false;
        });

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
            { data: "tax_deduction", render: fmtMoney },
            { data: "retirement_deduction", render: fmtMoney },
            { data: "amount_net", render: fmtMoney },
            { data: "housing_alloc", render: fmtMoney },
            { data: "transport_alloc", render: fmtMoney },
            { data: "living_alloc", render: fmtMoney },
            { data: "buffer_alloc", render: fmtMoney },
            { data: "status" },
            { data: "memo" },
          ],
        });
      });
    }

    if (firstDetails) firstDetails.open = true;
  }

  /* =========================
        Init
     ========================= */
  document.addEventListener("DOMContentLoaded", async () => {
    ensureSupabaseClient();
    await handleMagicLinkReturn();

    safeOn(sendLinkBtn, "click", sendMagicLink);

    // Unlock requires a real session
    safeOn(unlockBtn, "click", async () => {
      ensureSupabaseClient();
      const user = await getUser();
      if (!user) {
        setText(authMsg, "Please sign in with the magic link first.");
        showLock();
        return;
      }
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

    ["m_amount_gross", "m_amount_net"].forEach((id) =>
      safeOn(el(id), "input", () => {
        lastTyped = id.includes("gross") ? "gross" : "net";
        recalcModal();
      }),
    );

    safeOn(el("pay_mode"), "change", (e) => {
      modalMode = e.target.value;
      recalcModal();
    });

    [
      "rate_retirement",
      "rate_pension", // fallback
      "rate_withhold",
      "rate_tax", // fallback
      "rate_alloc_housing",
      "rate_alloc_transport",
      "rate_alloc_living",
      "rate_alloc_buffer",
    ].forEach((id) => safeOn(el(id), "input", () => recalcModal()));

    const user = await getUser();

    if (user) {
      initTTMAccordion(user).catch((err) =>
        console.error("TTM init failed:", err),
      );
      showApp();
    } else {
      showLock();
    }

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

    if (hasSB()) {
      window.sb.auth.onAuthStateChange((_e, s) => {
        if (s?.user) showApp();
        else showLock();
      });
    }
  });
})();
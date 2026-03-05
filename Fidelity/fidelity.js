/* =========================================================
   fidelity.js
   - Reads account/balance/performance/holdings from Supabase
   - Renders:
     * Accounts list + totals
     * Performance chart (portfolio or selected account)
     * Top/Bottom movers from latest holdings snapshot
   ========================================================= */

(() => {
  "use strict";

  // ---------- Helpers ----------
  const el = (id) => document.getElementById(id);

  const money = (n) => {
    const v = Number(n || 0);
    return "$" + v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  const pct = (n) => {
    const v = Number(n || 0);
    const sign = v > 0 ? "+" : "";
    return sign + v.toFixed(2) + "%";
  };

  const classifyDelta = (n) => {
    const v = Number(n || 0);
    if (Math.abs(v) < 0.005) return "flat";
    return v > 0 ? "up" : "down";
  };

  const parseRangeDays = (rangeStr) => {
    const d = Number(rangeStr);
    return Number.isFinite(d) ? d : 30;
  };

  const dateISO = (d) => {
    const x = new Date(d);
    const yyyy = x.getFullYear();
    const mm = String(x.getMonth() + 1).padStart(2, "0");
    const dd = String(x.getDate()).padStart(2, "0");
    return `${yyyy}-${mm}-${dd}`;
  };

  const daysAgoISO = (days) => {
    const d = new Date();
    d.setDate(d.getDate() - days);
    return dateISO(d);
  };

  // ---------- Supabase ----------
  const sb = () => window.sb;

  async function getUser() {
    const { data, error } = await sb().auth.getSession();
    if (error) throw error;
    return data?.session?.user || null;
  }

  // ---------- State ----------
  let user = null;
  let accounts = [];
  let selectedAccountId = null; // null = portfolio
  let rangeDays = 30;

  let chart = null;

  // ---------- Chart ----------
  function ensureChart() {
    const canvas = el("perfChart");
    if (!canvas) return null;

    if (chart) return chart;

    chart = new Chart(canvas, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Equity",
            data: [],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { mode: "index", intersect: false },
        },
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { ticks: { maxTicksLimit: 8 } },
          y: {
            ticks: {
              callback: (v) => money(v),
            },
          },
        },
      },
    });

    return chart;
  }

  function setStatus(msg) {
    const n = el("statusLine");
    if (n) n.textContent = msg;
  }

  // ---------- Data fetch ----------
  async function fetchAccounts() {
    const { data, error } = await sb()
      .from("accounts")
      .select("*")
      .order("created_at", { ascending: true });

    if (error) throw error;
    return data || [];
  }

  async function fetchLatestBalancesByAccount() {
    // We fetch the latest 2 records per account by grabbing recent rows then reducing
    const since = daysAgoISO(14); // enough to include last entries
    const { data, error } = await sb()
      .from("account_balances_daily")
      .select("account_id, as_of, balance, day_change")
      .gte("as_of", since)
      .order("as_of", { ascending: false });

    if (error) throw error;

    const latest = new Map(); // account_id -> row
    for (const r of data || []) {
      if (!latest.has(r.account_id)) latest.set(r.account_id, r);
    }
    return latest;
  }

  async function fetchPerformanceSeries(accountIdOrNull, days) {
    const since = daysAgoISO(days);

    if (accountIdOrNull) {
      const { data, error } = await sb()
        .from("account_performance_daily")
        .select("as_of, equity")
        .eq("account_id", accountIdOrNull)
        .gte("as_of", since)
        .order("as_of", { ascending: true });

      if (error) throw error;
      return (data || []).map((r) => ({ as_of: r.as_of, equity: Number(r.equity || 0) }));
    }

    // Portfolio series = sum across accounts by date
    const { data, error } = await sb()
      .from("account_performance_daily")
      .select("as_of, equity, account_id")
      .gte("as_of", since)
      .order("as_of", { ascending: true });

    if (error) throw error;

    const byDate = new Map(); // as_of -> total equity
    for (const r of data || []) {
      const k = r.as_of;
      const prev = byDate.get(k) || 0;
      byDate.set(k, prev + Number(r.equity || 0));
    }

    const out = [...byDate.entries()]
      .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
      .map(([as_of, equity]) => ({ as_of, equity }));

    return out;
  }

  async function fetchLatestHoldingsSnapshot(accountIdOrNull) {
    // Latest snapshot timestamp (portfolio or account)
    let q = sb()
      .from("holdings_snapshot")
      .select("as_of")
      .order("as_of", { ascending: false })
      .limit(1);

    if (accountIdOrNull) q = q.eq("account_id", accountIdOrNull);

    const { data: head, error: headErr } = await q;
    if (headErr) throw headErr;

    const latestAsOf = head?.[0]?.as_of;
    if (!latestAsOf) return [];

    // Fetch all rows with that timestamp (best-effort exact match)
    let q2 = sb()
      .from("holdings_snapshot")
      .select("symbol, qty, market_value, day_change_pct, account_id")
      .eq("as_of", latestAsOf);

    if (accountIdOrNull) q2 = q2.eq("account_id", accountIdOrNull);

    const { data, error } = await q2;
    if (error) throw error;

    return data || [];
  }

  // ---------- Render ----------
  function renderAccountsList(latestBalancesMap) {
    const host = el("accountsList");
    if (!host) return;

    host.innerHTML = "";

    let totalBal = 0;
    let totalChg = 0;

    for (const a of accounts) {
      const b = latestBalancesMap.get(a.id);
      const bal = Number(b?.balance || 0);
      const chg = Number(b?.day_change || 0);

      totalBal += bal;
      totalChg += chg;

      const card = document.createElement("div");
      card.className = "acct" + (selectedAccountId === a.id ? " active" : "");
      card.dataset.accountId = a.id;

      const deltaClass = classifyDelta(chg);
      const deltaText = (chg > 0 ? "+" : "") + money(chg).replace("$-", "-$");

      card.innerHTML = `
        <div class="acct-top">
          <div>
            <div class="acct-name">${escapeHtml(a.name)}</div>
            <div class="acct-meta">${escapeHtml(a.institution || "")} · ${escapeHtml(a.type || "")}</div>
          </div>
          <div style="text-align:right;">
            <div class="money"><strong>${money(bal)}</strong></div>
            <div class="delta ${deltaClass}">${deltaText}</div>
          </div>
        </div>
      `;

      card.addEventListener("click", () => {
        selectedAccountId = a.id;
        refreshAll().catch(console.error);
      });

      host.appendChild(card);
    }

    // totals
    const totalBalanceEl = el("totalBalance");
    const totalChangeEl = el("totalChange");
    if (totalBalanceEl) totalBalanceEl.textContent = money(totalBal);

    if (totalChangeEl) {
      const cls = classifyDelta(totalChg);
      totalChangeEl.textContent = (totalChg > 0 ? "+" : "") + money(totalChg).replace("$-", "-$");
      totalChangeEl.className = "money " + cls;
    }

    // as_of
    const asOfEl = el("asOf");
    if (asOfEl) asOfEl.textContent = "as of " + new Date().toLocaleString();
  }

  function renderChart(series, title) {
    const titleEl = el("chartTitle");
    if (titleEl) titleEl.textContent = title;

    const c = ensureChart();
    if (!c) return;

    const labels = series.map((r) => r.as_of);
    const values = series.map((r) => r.equity);

    c.data.labels = labels;
    c.data.datasets[0].data = values;
    c.update();

    // Period return
    const first = values.length ? values[0] : 0;
    const last = values.length ? values[values.length - 1] : 0;
    const ret = last - first;
    const retPct = first ? (ret / first) * 100 : 0;

    const pr = el("periodReturn");
    const pp = el("periodPct");
    if (pr) pr.textContent = (ret > 0 ? "+" : "") + money(ret).replace("$-", "-$");
    if (pp) pp.textContent = pct(retPct);

    const cls = classifyDelta(ret);
    if (pr) pr.className = "money " + cls;
    if (pp) pp.className = cls;
  }

  function renderMovers(rows) {
    // sort by day_change_pct
    const clean = rows
      .filter((r) => r.symbol && Number.isFinite(Number(r.day_change_pct)))
      .map((r) => ({
        symbol: String(r.symbol),
        pct: Number(r.day_change_pct),
        mv: Number(r.market_value || 0),
      }));

    clean.sort((a, b) => b.pct - a.pct);

    const top = clean.slice(0, 8);
    const bottom = clean.slice(-8).reverse();

    const topHost = el("topMovers");
    const botHost = el("bottomMovers");

    if (topHost) topHost.innerHTML = top.map(moverRowHtml).join("");
    if (botHost) botHost.innerHTML = bottom.map(moverRowHtml).join("");

    function moverRowHtml(x) {
      const cls = classifyDelta(x.pct);
      return `
        <div class="row">
          <span>${escapeHtml(x.symbol)}</span>
          <span class="highlight ${cls}">${pct(x.pct)}</span>
        </div>
      `;
    }
  }

  function highlightSelectedAccountCard() {
    const host = el("accountsList");
    if (!host) return;
    host.querySelectorAll(".acct").forEach((n) => {
      const id = n.dataset.accountId;
      n.classList.toggle("active", id === selectedAccountId);
    });
  }

  function setupRangeChips() {
    const chips = document.querySelectorAll(".chip[data-range]");
    chips.forEach((chip) => {
      chip.addEventListener("click", () => {
        chips.forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        rangeDays = parseRangeDays(chip.dataset.range);
        refreshAll().catch(console.error);
      });
    });
  }

  function setupButtons() {
    const reset = el("btnReset");
    const refresh = el("btnRefresh");

    if (reset) {
      reset.addEventListener("click", () => {
        selectedAccountId = null;
        refreshAll().catch(console.error);
      });
    }

    if (refresh) {
      refresh.addEventListener("click", () => {
        refreshAll().catch(console.error);
      });
    }
  }

  // ---------- Escape ----------
  function escapeHtml(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  // ---------- Main refresh ----------
  async function refreshAll() {
    if (!user) return;

    setStatus("Loading accounts…");
    accounts = await fetchAccounts();

    setStatus("Loading balances…");
    const latestBalances = await fetchLatestBalancesByAccount();
    renderAccountsList(latestBalances);
    highlightSelectedAccountCard();

    const selectedName =
      selectedAccountId
        ? accounts.find((a) => a.id === selectedAccountId)?.name || "Account"
        : "Portfolio";

    setStatus("Loading performance series…");
    const series = await fetchPerformanceSeries(selectedAccountId, rangeDays);

    renderChart(series, `${selectedName} Performance`);

    setStatus("Loading movers…");
    const holdings = await fetchLatestHoldingsSnapshot(selectedAccountId);
    renderMovers(holdings);

    setStatus("Loaded.");
  }

  // ---------- Boot ----------
  async function boot() {
    if (!window.sb) {
      console.error("window.sb not initialized. Ensure Supabase is created before fidelity.js.");
      return;
    }

    user = await getUser();
    if (!user) {
      setStatus("Sign in to load performance.");
      return;
    }

    setupRangeChips();
    setupButtons();

    await refreshAll();
  }

  document.addEventListener("DOMContentLoaded", () => {
    boot().catch((e) => {
      console.error(e);
      setStatus("Error: " + (e?.message || String(e)));
    });
  });
})();

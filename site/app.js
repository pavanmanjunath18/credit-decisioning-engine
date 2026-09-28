/* Credit decisioning site. All numbers come from data/*.json, written by src/export_site_data.py.
 *
 * Profit at any cost of funds is exact, not interpolated: it is linear in the rate, so each approval step
 * stores profit before cost of funds and total dollar-years, and
 *     profit(rate) = profit_before_cof - rate * dollar_years.
 * tests/test_site_data.py runs this same calculation in Python against src/strategy.py.
 */
(function () {
  "use strict";

  // ---------- palette & formatting ----------
  const css = getComputedStyle(document.documentElement);
  const C = {
    ink: css.getPropertyValue("--ink").trim(),
    ink2: css.getPropertyValue("--ink-2").trim(),
    muted: css.getPropertyValue("--muted").trim(),
    grid: css.getPropertyValue("--grid").trim(),
    accent: css.getPropertyValue("--accent").trim(),
    gray: css.getPropertyValue("--gray").trim(),
    gray2: css.getPropertyValue("--gray-2").trim(),
  };
  const SANS = "'IBM Plex Sans', -apple-system, sans-serif";

  const money = (v, d = 1) => (v < 0 ? "−" : "") + "$" + Math.abs(v / 1e6).toFixed(d) + "M";
  const signedMoney = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + "$" + Math.abs(v / 1e6).toFixed(1) + "M";
  const dollars = (v) => "$" + Math.round(Math.abs(v)).toLocaleString("en-US");
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const pts = (v) => (v * 100).toFixed(1);
  const $ = (id) => document.getElementById(id);
  const isSmall = () => window.matchMedia("(max-width: 720px)").matches;

  // ---------- Chart.js defaults: light grid, no legends (labels are drawn on the chart) ----------
  Chart.defaults.font.family = SANS;
  Chart.defaults.font.size = 12;
  Chart.defaults.color = C.muted;
  Chart.defaults.animation = false;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.display = false;
  Object.assign(Chart.defaults.plugins.tooltip, {
    backgroundColor: C.ink, titleFont: { family: SANS, weight: "600", size: 12 },
    bodyFont: { family: SANS, size: 12 }, padding: 10, cornerRadius: 6, displayColors: false,
  });
  const gridStyle = { color: C.grid, drawTicks: false };
  const borderOff = { display: false };

  // Direct labels: text placed next to marks, nudged apart vertically when they would overlap.
  Chart.register({
    id: "directLabels",
    afterDatasetsDraw(chart, _args, opts) {
      if (!opts || !opts.items) return;
      const items = opts.items(chart).filter(Boolean).sort((a, b) => a.y - b.y);
      const gap = opts.gap || 16;
      // Compare each label with every already-placed label nearby (not just the previous one in y order,
      // which may sit elsewhere on the chart), pushing it down until it clears them all.
      const placed = [];
      items.forEach((it) => {
        placed.forEach((p) => {
          if (Math.abs(p.x - it.x) < 120 && it.y < p.y + gap && it.y > p.y - gap) it.y = p.y + gap;
        });
        placed.push(it);
      });
      const ctx = chart.ctx;
      ctx.save();
      ctx.textBaseline = "middle";
      items.forEach((it) => {
        ctx.font = `${it.weight || 500} ${it.size || 12}px ${SANS}`;
        ctx.fillStyle = it.color;
        ctx.textAlign = it.align || "left";
        ctx.fillText(it.text, it.x, it.y);
      });
      ctx.restore();
    },
  });

  // Reference lines (a slider position or a threshold), drawn across the plot area.
  Chart.register({
    id: "refLines",
    afterDatasetsDraw(chart, _args, opts) {
      if (!opts || !opts.lines) return;
      const { ctx, chartArea: a } = chart;
      opts.lines(chart).forEach((ln) => {
        const scale = chart.scales[ln.axis];
        const p = scale.getPixelForValue(ln.value);
        ctx.save();
        ctx.strokeStyle = ln.color || C.ink2;
        ctx.lineWidth = ln.width || 1;
        ctx.setLineDash(ln.dash || [4, 4]);
        ctx.beginPath();
        if (ln.axis === "x") { ctx.moveTo(p, a.top); ctx.lineTo(p, a.bottom); }
        else { ctx.moveTo(a.left, p); ctx.lineTo(a.right, p); }
        ctx.stroke();
        if (ln.label) {
          ctx.setLineDash([]);
          ctx.font = `500 11px ${SANS}`;
          ctx.fillStyle = ln.color || C.ink2;
          ctx.textBaseline = "bottom";
          ctx.textAlign = ln.labelAlign || "left";
          const x = ln.axis === "x" ? p + (ln.labelAlign === "right" ? -5 : 5) : a.right;
          ctx.fillText(ln.label, x, a.top + 12);
        }
        ctx.restore();
      });
    },
  });

  // ---------- load ----------
  const files = ["key_numbers", "strategy", "vintages", "applicants", "fairness", "psi"];
  Promise.all(files.map((f) => fetch(`data/${f}.json`).then((r) => r.json())))
    .then(([K, S, V, A, F, P]) => {
      overview(K);
      strategy(K, S);
      vintages(V);
      applicants(K, A);
      fairness(K, F);
      monitoring(K, P);
      method(K);
      navHighlight();
    })
    .catch((err) => {
      document.querySelector("main").insertAdjacentHTML("afterbegin",
        `<p style="padding:24px 0;color:#8a2a2a">Could not load data (${err}). If you opened the file directly, serve the folder over HTTP instead.</p>`);
    });

  const metric = (K, model, split, key) => K.metrics.find((m) => m.model === model && m.split === split)[key];

  // ---------- 1. overview ----------
  function overview(K) {
    const fill = {
      n_loans: K.n_loans.toLocaleString("en-US"),
      hindsight_gain_4_pct: "+" + pct(K.hindsight_gain_4),
      hindsight_decline_pct: pct(1 - K.hindsight_approval_4, 0),
      model_vs_subgrade_84_m: money(K.model_vs_subgrade_84),
      auc_model: metric(K, "xgboost_with_lc", "test", "roc_auc").toFixed(3),
      auc_subgrade: metric(K, "lc_subgrade_lookup", "test", "roc_auc").toFixed(3),
    };
    document.querySelectorAll("[data-k]").forEach((el) => { el.textContent = fill[el.dataset.k]; });

    const n = (s) => `<span class="num">${s}</span>`;
    const findings = [
      `<strong>Most recent loans hadn’t finished yet.</strong> ${n(pct(K.share_unfinished_2018, 0))} of 2018’s 36-month loans were still running when the data ends, and the ones that had finished were mostly early payoffs. Only fully matured 2007–2015 loans are used.`,
      `<strong>The model adds value on top of LendingClub’s grade, not instead of it.</strong> Borrower data alone scores ${metric(K, "xgboost_no_lc", "test", "roc_auc").toFixed(3)} AUC vs ${fill.auc_subgrade} for the grade. Combined, it reaches ${fill.auc_model}, worth ${n(fill.model_vs_subgrade_84_m)} more than ranking by grade at 84% approval.`,
      `<strong>The cost of capital decides the policy.</strong> With free capital, approving almost everyone is best (${n("+" + pct(K.hindsight_gain_0))}), because LendingClub’s rates already price the risk. At a 4% cost of funds, declining the riskiest ${fill.hindsight_decline_pct} adds ${n(fill.hindsight_gain_4_pct)}.`,
      `<strong>A cutoff chosen on 2014 captured only ${n("+" + pct(K.chosen_gain_4))} in 2015.</strong> LendingClub cut grade B rates from ${K.grade_b_rate_2014.toFixed(1)}% to ${K.grade_b_rate_2015.toFixed(1)}% while defaults rose. The rate drift was visible at application time.`,
      `<strong>Income is the fairness pressure point.</strong> At a tighter 80% approval rate, applicants under $40K are approved at ${n(K.low_income_ratio_80.toFixed(2) + "×")} the best band’s rate, and the model overstates their risk gap (${pts(K.income_gap_predicted)} points predicted vs ${pts(K.income_gap_actual)} actual).`,
    ];
    $("findings").innerHTML = findings.map((f) => `<li>${f}</li>`).join("");
  }

  // ---------- 2. strategy ----------
  function strategy(K, S) {
    const grid = S.grid;
    const last = grid.length - 1;
    const state = { model: "xgboost_with_lc", idx: grid.indexOf(0.88), cof: 0.04 };
    const profitArr = (b, cof) => b.profit_before_cof.map((p, i) => p - cof * b.dollar_years[i]);
    const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);
    const at = (b, i, cof) => ({ approval_rate: b.approval_rate[i], n_approved: b.n_approved[i],
      default_rate: b.default_rate[i], dollar_loss: b.dollar_loss[i], profit: b.profit_before_cof[i] - cof * b.dollar_years[i] });
    const single = (b, cof) => ({ ...b, profit: b.profit_before_cof - cof * b.dollar_years });

    $("strategy-headline").textContent =
      `At a 4% cost of funds, declining the riskiest ${pct(1 - K.hindsight_approval_4, 0)} of applicants would have raised 2015 profit by ${pct(K.hindsight_gain_4)}.`;

    // ranking toggle
    const seg = $("ranking");
    Object.entries(S.rankings).forEach(([key, rk]) => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = rk.label; b.dataset.key = key;
      b.setAttribute("role", "radio");
      b.addEventListener("click", () => { state.model = key; render(); });
      seg.appendChild(b);
    });

    const colors = { xgboost_with_lc: C.accent, xgboost_no_lc: C.ink2, lc_subgrade_rank: C.gray };
    const dashes = { xgboost_with_lc: [], xgboost_no_lc: [6, 4], lc_subgrade_rank: [] };
    const keys = Object.keys(S.rankings);
    const chart = new Chart($("profit-chart"), {
      type: "line",
      data: {
        datasets: keys.map((k) => ({
          key: k, label: S.rankings[k].label, data: [],
          borderColor: colors[k], borderWidth: k === "xgboost_with_lc" ? 2.5 : 1.75,
          borderDash: dashes[k], pointRadius: 0, pointHoverRadius: 4, pointHitRadius: 8, tension: 0,
        })),
      },
      options: {
        layout: { padding: { right: 8, top: 18 } },
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { type: "linear", min: 50, max: 100, grid: gridStyle, border: borderOff,
               ticks: { callback: (v) => v + "%", stepSize: 10, padding: 8 } },
          y: { grid: gridStyle, border: borderOff, ticks: { callback: (v) => "$" + v + "M", padding: 8, maxTicksLimit: 6 } },
        },
        plugins: {
          tooltip: { callbacks: {
            title: (items) => `Approve ${items[0].parsed.x}%`,
            label: (it) => `${it.dataset.label}: ${money(it.parsed.y * 1e6)}`,
          } },
          directLabels: {
            gap: 16,
            // Labels at 80% approval, where the lines are apart: the top line labelled above itself,
            // the others below. On phones the lines are too close, so a key above the chart is used instead.
            items: (ch) => {
              if (isSmall()) return [];
              const idx = grid.indexOf(0.8);
              const pts = ch.data.datasets.map((ds, i) => ({ ds, p: ch.getDatasetMeta(i).data[idx] })).filter((o) => o.p);
              const topY = Math.min(...pts.map((o) => o.p.y));
              return pts.map(({ ds, p }) => ({
                x: p.x, y: p.y === topY ? p.y - 13 : p.y + 14, align: "center", text: ds.label,
                color: ds.key === "xgboost_with_lc" ? C.accent : C.ink2, weight: ds.key === "xgboost_with_lc" ? 600 : 500,
              }));
            },
          },
          refLines: { lines: () => [{ axis: "x", value: grid[state.idx] * 100, color: C.ink, dash: [3, 3],
                                      label: `you: ${Math.round(grid[state.idx] * 100)}%`, labelAlign: grid[state.idx] > 0.9 ? "right" : "left" }] },
        },
      },
    });

    $("profit-key").innerHTML = keys.map((k) => {
      const dash = dashes[k].length ? `stroke-dasharray="${dashes[k].join(" ")}"` : "";
      return `<span class="${k === "xgboost_with_lc" ? "model" : ""}"><svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="${colors[k]}" stroke-width="${k === "xgboost_with_lc" ? 2.5 : 1.75}" ${dash}/></svg>${S.rankings[k].label}</span>`;
    }).join("");

    function render() {
      const cof = state.cof;
      const rk = S.rankings[state.model];
      [...seg.children].forEach((b) => b.setAttribute("aria-checked", String(b.dataset.key === state.model)));
      $("rate-out").textContent = Math.round(grid[state.idx] * 100) + "%";
      $("cof-out").textContent = (cof * 100).toFixed(1) + "%";

      // readout for the slider position vs approving everyone
      const now = at(rk.test, state.idx, cof);
      const all = at(rk.test, last, cof);
      const lossChange = now.dollar_loss / all.dollar_loss - 1;
      $("readout").innerHTML = [
        ["Approved", pct(now.approval_rate, 0), `${now.n_approved.toLocaleString("en-US")} loans`],
        ["Default rate", pct(now.default_rate, 2), `${now.default_rate >= all.default_rate ? "+" : "−"}${Math.abs((now.default_rate - all.default_rate) * 100).toFixed(2)} pts vs all`],
        ["Dollar losses", money(now.dollar_loss), `${lossChange >= 0 ? "+" : "−"}${pct(Math.abs(lossChange))} vs all`],
        ["Net profit", money(now.profit), `${signedMoney(now.profit - all.profit)} vs all`],
      ].map(([l, v, d]) => `<div class="ro"><div class="ro-label">${l}</div><div class="ro-value">${v}</div><div class="ro-delta">${d}</div></div>`).join("");

      // chart
      chart.data.datasets.forEach((ds) => {
        const p = profitArr(S.rankings[ds.key].test, cof);
        ds.data = grid.map((g, i) => ({ x: Math.round(g * 100), y: p[i] / 1e6 }));
        ds.borderWidth = ds.key === state.model ? 2.75 : 1.5;
      });
      chart.update("none");

      // policy table: everything recomputed for this ranking and cost of funds
      const i14 = argmax(profitArr(rk.valid, cof));
      const i15 = argmax(profitArr(rk.test, cof));
      const rows = [
        { name: "Approve everyone", r: all },
        { name: "Decline grades E–\u2060G (LendingClub rule)", r: single(S.grade_rule, cof) },
        { name: `PD cutoff chosen on 2014 (PD ≤ ${pct(rk.valid.pd_cutoff[i14])})`, r: at(rk.pd_cutoff_on_test, i14, cof) },
        { name: `Chosen on 2014: approve safest ${pct(grid[i14], 0)}`, r: at(rk.test, i14, cof), cls: "chosen" },
        { name: `Best in hindsight: approve ${pct(grid[i15], 0)}`, r: at(rk.test, i15, cof), cls: "hindsight" },
        { name: `Your slider: approve ${pct(grid[state.idx], 0)}`, r: now, cls: "yours" },
      ];
      $("policy-table").querySelector("tbody").innerHTML = rows.map(({ name, r, cls }) => `
        <tr class="${cls || ""}"><td>${name}</td><td>${pct(r.approval_rate)}</td><td class="hide-sm">${pct(r.default_rate, 2)}</td>
        <td class="hide-sm">${money(r.dollar_loss)}</td><td>${money(r.profit)}</td>
        <td>${name === "Approve everyone" ? "—" : signedMoney(r.profit - all.profit)}</td></tr>`).join("");

      const chosen = at(rk.test, i14, cof);
      const best = at(rk.test, i15, cof);
      $("strategy-meaning").innerHTML = cof === 0
        ? `With free capital, the best 2015 cutoff approves ${pct(grid[i15], 0)} and adds only ${signedMoney(best.profit - all.profit)}: LendingClub’s interest rates already price the risk, so even risky loans pay for themselves. Raise the cost of funds to see the answer change.`
        : `LendingClub already priced risk into its rates, so declining a loan also gives up its interest. At a ${(cof * 100).toFixed(1)}% cost of funds the best 2015 cutoff approves ${pct(grid[i15], 0)} (${signedMoney(best.profit - all.profit)} vs approving everyone), but the cutoff chosen on 2014 approves ${pct(grid[i14], 0)} and captures ${signedMoney(chosen.profit - all.profit)}. The gap is 2015’s thinner margins, not the model: cutoffs need re-optimizing when pricing moves.`;
    }

    $("rate").addEventListener("input", (e) => { state.idx = grid.indexOf(+e.target.value / 100); render(); });
    $("cof").addEventListener("input", (e) => { state.cof = +e.target.value / 100; render(); });
    render();
  }

  // ---------- 3. vintages ----------
  function vintages(V) {
    const years = Object.keys(V).sort();
    const at = (y, m) => V[y].rate[m - 1];
    const worsening = ["2013", "2014", "2015", "2016"].map((y) => `${y}: ${pct(at(y, 24))}`).join(", ");
    $("vintage-headline").textContent = "Every vintage since 2013 defaulted faster than the one before it.";
    $("vintage-meaning").innerHTML =
      `Share charged off within 24 months, by issue year: ${worsening}. A model trained on older vintages under-predicts a worsening one, which is exactly what happened in 2015. Lines that stop early are unfinished vintages (the data ends March 2019). That is why only 2007–2015 loans are used for modeling.`;

    const style = (y) => {
      if (y === "2015") return { borderColor: C.accent, borderWidth: 2.75, borderDash: [] };
      if (+y >= 2016) return { borderColor: C.ink2, borderWidth: 1.5, borderDash: [5, 4] };
      return { borderColor: C.gray2, borderWidth: 1.5, borderDash: [] };
    };
    new Chart($("vintage-chart"), {
      type: "line",
      data: { datasets: years.map((y) => ({
        label: y, data: V[y].rate.map((v, i) => ({ x: i + 1, y: v * 100 })),
        pointRadius: 0, pointHitRadius: 6, tension: 0, ...style(y),
      })) },
      options: {
        layout: { padding: { right: 40, top: 10 } },
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "linear", min: 0, max: 60, grid: gridStyle, border: borderOff,
               ticks: { stepSize: 12, callback: (v) => v + " mo", padding: 8 } },
          y: { min: 0, grid: gridStyle, border: borderOff, ticks: { callback: (v) => v + "%", padding: 8 } },
        },
        plugins: {
          tooltip: { callbacks: {
            title: (items) => `${items[0].dataset.label} vintage, month ${items[0].parsed.x}`,
            label: (it) => `${it.parsed.y.toFixed(2)}% charged off`,
          } },
          directLabels: { gap: 15, items: (ch) => ch.data.datasets.map((ds, i) => {
            const pts = ch.getDatasetMeta(i).data;
            const p = pts[pts.length - 1];
            if (!p || ds.data.length < 6) return null; // 2018 has only 3 months: too short to label
            const key = ds.label === "2015";
            const base = { text: ds.label, color: key ? C.accent : C.ink2, weight: key ? 600 : 500, size: 11 };
            // Finished vintages (48+ months): label in the right margin so labels never sit on lines.
            // On phones six labels can't fit beside their lines, so one group label replaces them.
            if (ds.data.length >= 48) {
              if (!isSmall()) return { ...base, x: ch.chartArea.right + 5, y: p.y };
              if (ds.label !== "2012") return null;
              const ends = ch.data.datasets.map((d, j) => d.data.length >= 48 ? ch.getDatasetMeta(j).data.at(-1).y : null).filter((y) => y !== null);
              // Placed above the cluster, in the empty space right of where the 2015 line ends.
              return { ...base, text: "2009–14", align: "center", x: ch.scales.x.getPixelForValue(51), y: Math.min(...ends) - 17 };
            }
            if (key) return { ...base, x: p.x + 5, y: p.y };
            // Unfinished vintages: label above-left of the line end, clear of the 2015 line.
            return { ...base, x: p.x - 4, y: p.y - 10, align: "right", text: ds.label === "2016" ? "2016 (unfinished)" : ds.label };
          }) },
        },
      },
    });
  }

  // ---------- 4. applicants ----------
  function applicants(K, A) {
    const cutoff = K.chosen_policy.pd_cutoff;
    const MAX = 0.6;
    const declined = A.map((a, i) => (a.approved ? -1 : i)).filter((i) => i >= 0);
    $("pd-cutoff").style.left = (cutoff / MAX) * 100 + "%";
    $("pd-cutoff-label").textContent = `cutoff ${pct(cutoff)}`;
    $("decline-share").textContent = pct(1 - K.chosen_policy.approval_rate, 0);

    // Round the percentile down: the riskiest applicant is riskier than 99.6% of others, not "100%".
    const riskierThan = (p) => Math.floor(p * 100) + "%";

    function show(i) {
      const a = A[i];
      $("risk").value = i;
      $("risk-out").textContent = `riskier than ${riskierThan(a.percentile)} of 2015 applicants`;
      const v = $("verdict");
      v.textContent = a.approved ? "Approved" : "Declined";
      v.className = "verdict " + (a.approved ? "approved" : "declined");
      $("verdict-sub").innerHTML = `Probability of default <strong>${pct(a.pd)}</strong> · riskier than ${riskierThan(a.percentile)} of 2015 applicants`;
      const w = Math.min(a.pd / MAX, 1) * 100;
      $("pd-fill").style.width = w + "%";
      $("pd-marker").style.left = w + "%";

      $("reasons-title").textContent = a.approved ? "What raised this applicant’s risk most" : "Reasons for decline";
      $("reasons").innerHTML = a.reasons.length
        ? a.reasons.map((r) => `<li>${r}</li>`).join("")
        : "<li>No factor raised risk above the average applicant.</li>";

      const emp = a.emp_length === null ? "not provided" : a.emp_length >= 10 ? "10+ years" : `${a.emp_length} year${a.emp_length === 1 ? "" : "s"}`;
      const facts = [
        ["Loan", `${dollars(a.loan_amnt)} · ${a.purpose}`],
        ["LendingClub grade", `${a.sub_grade} at ${a.int_rate.toFixed(2)}%`],
        ["Income", dollars(a.annual_inc)],
        ["FICO", a.fico],
        ["Debt-to-income", a.dti === null ? "n/a" : a.dti.toFixed(1) + "%"],
        ["Home", a.home_ownership],
        ["Employment", emp],
      ];
      $("facts").innerHTML = facts.map(([k, val]) => `<dt>${k}</dt><dd>${val}</dd>`).join("");

      const made = a.net_profit >= 0;
      $("outcome").innerHTML = `<strong>What actually happened:</strong> the loan was ${a.outcome === "Charged off" ? "charged off" : "paid in full"}. ` +
        `The lender ${made ? "made" : "lost"} ${dollars(a.net_profit)} on ${dollars(a.funded)} funded (before cost of funds).`;
    }

    $("risk").addEventListener("input", (e) => show(+e.target.value));
    $("btn-random").addEventListener("click", () => show(Math.floor(Math.random() * A.length)));
    $("btn-declined").addEventListener("click", () => show(declined[Math.floor(Math.random() * declined.length)]));
    show(Math.floor(A.length / 2));
  }

  // ---------- 5. fairness ----------
  function fairness(K, F) {
    $("fair-headline").textContent =
      `At tighter cutoffs, applicants earning under $40K are approved at about half the rate of the best income band.`;
    $("fair-meaning").innerHTML =
      `Income is the most common decline reason and a legitimate ability-to-pay factor. But among approved applicants the model predicts a ${pts(K.income_gap_predicted)}-point default gap between the under-$40K and $80–100K bands when the actual gap is ${pts(K.income_gap_actual)} points, so it over-penalizes lower incomes. That is an accuracy problem to fix first (for example, recalibrating by income band), and a flag for a proper fair-lending review.`;

    const ops = { stress_80: "Stress test: approve 80%", chosen: `Chosen policy: approve ${pct(K.chosen_policy.approval_rate, 0)}` };
    let op = "stress_80";
    const seg = $("fair-op");
    Object.entries(ops).forEach(([key, label]) => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label; b.dataset.key = key; b.setAttribute("role", "radio");
      b.addEventListener("click", () => { op = key; render(); });
      seg.appendChild(b);
    });

    const valueLabels = (fmt, suffixFor) => ({
      gap: 0,
      items: (ch) => ch.data.datasets.flatMap((ds, i) => ch.getDatasetMeta(i).data.map((bar, j) => ({
        x: bar.x + 6, y: bar.y, text: fmt(ds.data[j]) + (suffixFor ? suffixFor(i, j) : ""),
        color: ds.labelColor ? ds.labelColor[j] : C.ink2, size: 11,
      }))),
    });
    const barOpts = (xMax, xFmt, extra) => ({
      indexAxis: "y",
      layout: { padding: { right: 70 } },
      scales: {
        x: { min: 0, max: xMax, grid: gridStyle, border: borderOff, ticks: { callback: xFmt, padding: 6, stepSize: xMax / 5 } },
        y: { grid: { display: false }, border: borderOff, ticks: { color: C.ink2, padding: 6 } },
      },
      plugins: { tooltip: { enabled: false }, ...extra },
    });

    const incomeChart = new Chart($("fair-income-chart"), {
      type: "bar",
      data: { labels: [], datasets: [{ data: [], backgroundColor: [], labelColor: [], barThickness: 16, borderRadius: 3 }] },
      options: barOpts(1.0, (v) => v.toFixed(1), {
        directLabels: valueLabels((v) => v.toFixed(2)),
        refLines: { lines: () => [{ axis: "x", value: 0.8, color: C.ink2, label: "0.80 screen" }] },
      }),
    });
    const calibChart = new Chart($("fair-calib-chart"), {
      type: "bar",
      data: { labels: [], datasets: [
        { data: [], backgroundColor: C.accent, barThickness: 9, borderRadius: 2 },
        { data: [], backgroundColor: C.gray2, barThickness: 9, borderRadius: 2 },
      ] },
      options: barOpts(15, (v) => v + "%", {
        directLabels: valueLabels((v) => v.toFixed(1) + "%", (i, j) => (j === 0 ? (i === 0 ? " predicted" : " actual") : "")),
      }),
    });

    function render() {
      [...seg.children].forEach((b) => b.setAttribute("aria-checked", String(b.dataset.key === op)));
      const g = F[op].groups;
      const inc = g.filter((x) => x.dimension === "income_band");
      const labels = inc.map((x) => x.group.replace("-", "–"));
      incomeChart.data.labels = labels;
      const ds = incomeChart.data.datasets[0];
      ds.data = inc.map((x) => x.approval_ratio);
      ds.backgroundColor = inc.map((x) => (x.approval_ratio < 0.8 ? C.accent : C.gray2));
      ds.labelColor = inc.map((x) => (x.approval_ratio < 0.8 ? C.accent : C.ink2));
      incomeChart.update("none");

      calibChart.data.labels = labels;
      calibChart.data.datasets[0].data = inc.map((x) => x.mean_pd_approved * 100);
      calibChart.data.datasets[1].data = inc.map((x) => x.default_rate_approved * 100);
      calibChart.update("none");

      const home = Object.fromEntries(g.filter((x) => x.dimension === "home_ownership").map((x) => [x.group, x]));
      const states = g.filter((x) => x.dimension === "addr_state" && !x.small).sort((a, b) => a.approval_ratio - b.approval_ratio);
      $("fair-side").innerHTML =
        `<strong>Home ownership:</strong> renters are approved at ${home.RENT.approval_ratio.toFixed(2)}× and outright owners at ${home.OWN.approval_ratio.toFixed(2)}× the rate of mortgage holders. ` +
        `<strong>State</strong> (not a model input): ratios run from ${states[0].approval_ratio.toFixed(2)} (${states[0].group}) to 1.00 across ${states.length} states with 1,000+ applicants. Neither falls below 0.80.`;
    }
    render();
  }

  // ---------- 5b. monitoring ----------
  function monitoring(K, P) {
    const names = {
      score: "Model score", sub_grade_num: "LendingClub sub-grade", int_rate: "Interest rate", annual_inc: "Income",
      payment_to_income: "Payment / income", revol_bal: "Revolving balance", credit_history_months: "Credit history length",
      dti: "Debt-to-income", fico_mid: "FICO", purpose: "Loan purpose", revol_util: "Revolving utilization",
      mort_acc: "Mortgage accounts", verification_status: "Income verification", emp_length_yrs: "Employment length",
      loan_to_income: "Loan / income", inq_last_6mths: "Recent inquiries",
    };
    const score = P.find((p) => p.feature === "score");
    const feats = P.filter((p) => p.feature !== "score" && (p.gain_share >= 0.03 || p.psi_2015 > 0.1));
    const rows = [score, ...feats];
    const CAP = 0.3;
    $("psi-headline").textContent = "Interest-rate drift flagged thinner 2015 margins from application data alone, months before the defaults showed it.";
    $("psi-meaning").innerHTML =
      `The model score barely moved (PSI ${K.psi_score.toFixed(3)}), but interest rates shifted (${K.psi_int_rate.toFixed(2)}): LendingClub cut rates within grades, with grade B going from ${K.grade_b_rate_2014.toFixed(1)}% to ${K.grade_b_rate_2015.toFixed(1)}%. That is the margin squeeze that made the 2014 cutoff too loose, and PSI sees it the day applications arrive. PSI can’t see outcomes, so it has to be paired with predicted-vs-actual default tracking. The mortgage-accounts shift is a data artifact: LendingClub didn’t collect it before 2012.`;

    new Chart($("psi-chart"), {
      type: "bar",
      data: {
        labels: rows.map((r) => names[r.feature] || r.feature),
        datasets: [{
          data: rows.map((r) => Math.min(r.psi_2015, CAP)),
          backgroundColor: rows.map((r) => (r.psi_2015 > 0.1 ? C.accent : C.gray2)),
          barThickness: 14, borderRadius: 3,
        }],
      },
      options: {
        indexAxis: "y",
        layout: { padding: { right: 96, top: 14 } },
        scales: {
          x: { min: 0, max: CAP, grid: gridStyle, border: borderOff,
               ticks: { stepSize: isSmall() ? 0.1 : 0.05, padding: 6, maxRotation: 0, callback: (v) => v.toFixed(2) } },
          y: { grid: { display: false }, border: borderOff, ticks: { color: C.ink2, padding: 6, autoSkip: false } },
        },
        plugins: {
          tooltip: { enabled: false },
          refLines: { lines: () => [
            { axis: "x", value: 0.1, color: C.ink2, label: "0.10", labelAlign: "left" },
            { axis: "x", value: 0.25, color: C.ink2, label: "0.25", labelAlign: "left" },
          ] },
          directLabels: { gap: 0, items: (ch) => ch.getDatasetMeta(0).data.map((bar, j) => {
            const v = rows[j].psi_2015;
            const off = v > CAP;
            return { x: bar.x + 6, y: bar.y, text: off ? `${v.toFixed(2)} (off scale)` : v.toFixed(3),
                     color: v > 0.1 ? C.accent : C.ink2, size: 11, weight: v > 0.1 ? 600 : 500,
                     align: off && isSmall() ? "right" : "left", ...(off && isSmall() ? { x: bar.x - 6, color: "#ffffff" } : {}) };
          }) },
        },
      },
    });
  }

  // ---------- 6. method ----------
  function method(K) {
    $("method-data").textContent =
      `${K.n_loans.toLocaleString("en-US")} LendingClub 36-month loans issued 2007–2015 that were paid in full or charged off. Newer loans are excluded: ${pct(K.share_unfinished_2018, 0)} of 2018 loans were still running when the data ends, and keeping only the finished ones biases default rates down.`;
    $("method-models").textContent =
      `Logistic regression and XGBoost, each trained with and without LendingClub’s grade and rate. The decision model is XGBoost with the grade: 2015 ROC-AUC ${metric(K, "xgboost_with_lc", "test", "roc_auc").toFixed(3)} vs ${metric(K, "lc_subgrade_lookup", "test", "roc_auc").toFixed(3)} for the grade alone. State was dropped from the model (AUC change ${K.state_drop_auc_change < 0 ? "−" : "+"}${Math.abs(K.state_drop_auc_change).toFixed(4)}) so every decline reason is defensible.`;
    $("limit-calib").textContent =
      `The model under-predicts risk in 2015 (average predicted default ${pct(metric(K, "xgboost_with_lc", "test", "mean_pd"))} vs ${pct(metric(K, "xgboost_with_lc", "test", "actual_default_rate"))} actual), so decisions use the model’s ranking and actual dollar outcomes, not its probabilities.`;
  }

  // ---------- nav: highlight the section in view ----------
  function navHighlight() {
    const links = [...document.querySelectorAll(".navlinks a")];
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === "#" + e.target.id));
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("main > section, main > header").forEach((s) => obs.observe(s));
  }
})();

"use strict";
/* G1 Dance Studio — frontend engine. Plain framework-free JS, no build step.
   Wires the dark "Creative Studio" design to the real FastAPI backend. */

// ---------- tiny helpers ----------
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtDur = (s) => { s = Math.round(s || 0); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const vnd = (n) => n == null ? "—" : (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "K" : Math.round(n)) + " ₫";

// engine-down aware fetch (audit: show a banner, don't silently show stale)
let ENGINE_DOWN = false;
async function api(path, opts) {
  try {
    const r = await fetch(path, opts);
    setEngineDown(false);
    if (!r.ok) { let msg; try { msg = (await r.json()).detail; } catch { msg = r.statusText; } const e = new Error(msg || ("HTTP " + r.status)); e.status = r.status; throw e; }
    const ct = r.headers.get("content-type") || "";
    return ct.includes("json") ? r.json() : r.text();
  } catch (e) {
    if (e instanceof TypeError) setEngineDown(true); // network / server gone
    throw e;
  }
}
function setEngineDown(down) {
  if (down === ENGINE_DOWN) return; ENGINE_DOWN = down;
  $("#banner").innerHTML = down ? `<div class="banner"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/></svg>The app engine isn't responding — data below may be stale. Retrying…</div>` : "";
}

// toasts
function toast(msg, kind = "info") {
  const icons = { ok: '<path d="m5 12 4 4L19 6"/>', err: '<path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/>', info: '<path d="M12 8v5M12 16h.01"/><circle cx="12" cy="12" r="9"/>' };
  const t = el(`<div class="toast ${kind}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">${icons[kind] || icons.info}</svg><div>${esc(msg)}</div></div>`);
  $("#toasts").appendChild(t); setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.remove(), 300); }, 4200);
}

// confirm/prompt modal → Promise<string|null>
function modal({ title, body, confirmLabel = "Confirm", danger = false, input = null }) {
  return new Promise((resolve) => {
    const bg = el(`<div class="modal-bg"><div class="modal"><h3>${esc(title)}</h3><p>${body || ""}</p>${input != null ? `<input class="field" id="mIn" placeholder="${esc(input)}" autocomplete="off">` : ""}<div class="row"><button class="btn btn-ghost" id="mNo">Cancel</button><button class="btn ${danger ? "btn-danger" : "btn-primary"}" id="mYes">${esc(confirmLabel)}</button></div></div></div>`);
    $("#modalRoot").appendChild(bg);
    const done = (v) => { bg.remove(); resolve(v); };
    $("#mNo", bg).onclick = () => done(null);
    $("#mYes", bg).onclick = () => done(input != null ? ($("#mIn", bg).value || "") : "ok");
    bg.onclick = (e) => { if (e.target === bg) done(null); };
    const inp = $("#mIn", bg); if (inp) inp.focus();
  });
}

const ICON = { play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>', check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m5 12 4 4L19 6"/></svg>' };
const THUMBS = ["thumb-1", "thumb-2", "thumb-3", "thumb-4", "thumb-5"];
const thumbFor = (id) => THUMBS[[...String(id)].reduce((a, c) => a + c.charCodeAt(0), 0) % THUMBS.length];
const STATUS_BADGE = {
  "show-ready": '<span class="badge b-ready"><span class="dot" style="background:var(--ok)"></span>Show-Ready</span>',
  "sim-verified": '<span class="badge b-verified">Sim-Verified</span>',
  "draft": '<span class="badge b-draft">Draft</span>',
};

// ---------- global state ----------
const S = { system: null, dances: [], jobs: [], stageOrder: ["extract", "retarget", "train", "verify", "export"], selJob: null, cur: "dashboard", danceFilter: "all", search: "" };

// ---------- navigation ----------
const TITLES = { dashboard: ["Dashboard", "Overview of your studio"], library: ["Library", "Your trained dances"], studio: ["Create", "Video → balanced robot dance"], show: ["Show Mode", "Operator console"], system: ["System", "Cloud GPU & training"], settings: ["Settings", "Connections & safety"] };
function go(id) {
  if (!TITLES[id]) return;
  S.cur = id;
  $$(".screen").forEach(s => s.classList.toggle("active", s.id === id));
  $$(".nav-item").forEach(n => n.classList.toggle("active", n.dataset.nav === id));
  const [t, c] = TITLES[id]; $("#pageTitle").textContent = t; $("#pageCrumb").textContent = c;
  $(".main").scrollTo(0, 0);
  RENDER[id] && RENDER[id]();
}
document.addEventListener("click", (e) => { const n = e.target.closest("[data-nav]"); if (n) go(n.dataset.nav); });

// ---------- data refresh ----------
async function refreshSystem() {
  try { S.system = await api("/api/system"); renderFooter(); if (S.cur === "dashboard" || S.cur === "system") RENDER[S.cur](); }
  catch { renderFooter(); }
}
async function refreshDances() { try { S.dances = await api("/api/dances"); const n = S.dances.length; const b = $("#navDanceCount"); b.textContent = n; b.classList.toggle("hidden", !n); } catch { } }
async function refreshJobs() { try { S.jobs = await api("/api/jobs"); } catch { } }

function renderFooter() {
  const sys = S.system, g = sys && sys.gpu, cost = sys && sys.cost;
  const dot = $("#footGpu .dot"), txt = $("#footGpu .sc-txt");
  if (sys && sys.reachable && g) {
    dot.className = "dot live"; dot.style.background = "var(--ok)";
    txt.innerHTML = `<b>GPU · RTX 4090</b><span>${g.utilization_pct ?? "?"}% · ${g.temperature_c ?? "?"}°C · ${g.busy ? "training" : "idle"}</span>`;
  } else {
    dot.className = "dot"; dot.style.background = "var(--text-faint)";
    txt.innerHTML = `<b>GPU box</b><span>${sys ? "unreachable" : "checking…"}</span>`;
  }
  if (cost) { $("#footCost").textContent = `${vnd(cost.accrued_vnd)} / ${vnd(cost.cap_vnd)}`; $("#footCostBar").style.width = Math.min(100, (cost.cap_fraction || 0) * 100) + "%"; }
}

// ---------- screen renderers ----------
const RENDER = {};

RENDER.dashboard = function () {
  const sys = S.system, cost = sys && sys.cost;
  const byStatus = { "show-ready": 0, "sim-verified": 0, "draft": 0 };
  S.dances.forEach(d => byStatus[d.status] = (byStatus[d.status] || 0) + 1);
  const allJobs = (sys && sys.jobs || []);
  const isLive = j => (j.live ?? j.running) === true;
  const active = allJobs.filter(isLive)[0];  // only a genuinely-running job is "active"
  const capPct = cost ? Math.round((cost.cap_fraction || 0) * 1000) / 10 : 0;
  const progRows = [...allJobs].sort((a, b) => (isLive(b) ? 1 : 0) - (isLive(a) ? 1 : 0)).map(j => {
    const live = isLive(j);
    const pct = j.max_iteration ? Math.round(100 * j.iteration / j.max_iteration) : 0;
    const tag = live
      ? `<span class="badge b-train">${esc(shortJob(j.name))}</span>`
      : `<span class="badge" style="opacity:.6">${esc(shortJob(j.name))} · ${esc(j.state || "finished")}</span>`;
    return `<div class="prog-row"${live ? "" : ' style="opacity:.5"'}><div class="nm">${tag}</div><div class="bar"><i style="width:${pct}%"></i></div><div class="pct">${j.iteration ? (j.iteration / 1000).toFixed(1) + "K" : "—"}${j.max_iteration ? "/" + (j.max_iteration / 1000).toFixed(0) + "K" : ""}</div></div>`;
  }).join("") || `<p class="muted" style="font-size:12.5px">No training job is running right now.</p>`;

  $("#dashboard").innerHTML = `
    <div class="grid g-4" style="margin-bottom:18px">
      <div class="stat"><div class="lbl">${svg('<path d="M4 4v16h16"/><path d="M4 16l5-5 4 3 6-7"/>')}Active Training</div><div class="big">${active ? esc(shortJob(active.name)) : "Idle"}</div><div class="sub">${active ? `reward ${fmtNum(active.mean_reward)} · ${epPct(active)}` : "no job running"}</div></div>
      <div class="stat"><div class="lbl">${svg('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18"/>')}Dances</div><div class="big">${S.dances.length}</div><div class="sub">${byStatus["show-ready"]} show-ready · ${byStatus["sim-verified"]} verified</div></div>
      <div class="stat"><div class="lbl">${svg('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>')}GPU Spend</div><div class="big">${cost ? vnd(cost.accrued_vnd) : "—"}</div><div class="sub">${cost ? `of ${vnd(cost.cap_vnd)} · ~$${cost.accrued_usd}` : ""}</div></div>
      <div class="stat"><div class="lbl">${svg('<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>')}Show-Ready</div><div class="big">${byStatus["show-ready"]}</div><div class="sub">${byStatus["show-ready"] ? "ready to perform" : "none yet"}</div></div>
    </div>
    <div class="grid g-2" style="margin-bottom:18px">
      <div class="card">
        <div class="card-h"><div class="ico">${svg('<path d="M4 4v16h16"/><path d="M4 15l4-4 3 2 5-6 4 3"/>')}</div><h3>Training Progress</h3>${active ? '<span class="badge b-train"><span class="dot" style="background:var(--violet)"></span>Live</span>' : ""}</div>
        ${progRows}
        <p style="font-size:12px;color:var(--text-faint);margin-top:12px">${sys && sys.reachable ? "Reading the GPU box directly over SSH." : "Box unreachable — training may still be running; showing last-known."}</p>
      </div>
      <div class="card">
        <div class="card-h"><div class="ico">${svg('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>')}</div><h3>Budget</h3></div>
        <div class="gauge" style="--v:${capPct}"><div class="gv"><b>${capPct}%</b><span>of cap used</span></div></div>
        <div style="display:flex;justify-content:space-between;margin-top:14px;font-size:12.5px"><span style="color:var(--text-dim)">Spent <b style="color:var(--text)">${cost ? vnd(cost.accrued_vnd) : "—"}</b></span><span style="color:var(--text-dim)">Cap <b style="color:var(--text)">${cost ? vnd(cost.cap_vnd) : "—"}</b></span></div>
        <div class="hint">${svg('<path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/>')}Billing runs until the box is deleted — not just stopped.</div>
      </div>
    </div>
    <div class="grid g-2">
      <div class="card">
        <div class="card-h"><div class="ico">${svg('<path d="M12 8v4l3 2"/><circle cx="12" cy="12" r="9"/>')}</div><h3>Recent Dances</h3><span class="more" data-nav="library" style="cursor:pointer">View all →</span></div>
        ${S.dances.slice(0, 4).map(d => `<div class="act" data-open-dance="${esc(d.id)}" style="cursor:pointer"><div class="ai">${ICON.play}</div><div class="at"><b>${esc(d.name)}</b> · ${d.duration_s ? fmtDur(d.duration_s) : "—"}</div><div class="aw">${STATUS_BADGE[d.status] || esc(d.status)}</div></div>`).join("") || `<p class="muted" style="font-size:12.5px">No dances yet — create one from a video.</p>`}
      </div>
      <div class="card">
        <div class="card-h"><div class="ico">${svg('<path d="M13 2 3 14h8l-1 8 10-12h-8l1-8Z"/>')}</div><h3>Next Steps</h3></div>
        ${nextSteps(byStatus)}
      </div>
    </div>`;
};
function nextSteps(byStatus) {
  const steps = [];
  if (byStatus["sim-verified"] || byStatus["draft"]) steps.push(["1", "Verify a trained dance", "Run the signed sim-exam to reach show-ready", "show"]);
  if (byStatus["show-ready"]) steps.push(["2", "Schedule robot day", "Gantry test · health check · 30 min", "show"]);
  steps.push(["3", "Back up your library", `${S.dances.length} dances · export a .tar.gz`, "settings"]);
  return `<div style="display:flex;flex-direction:column;gap:11px">` + steps.map(([n, t, d, nav]) =>
    `<div data-nav="${nav}" style="display:flex;align-items:center;gap:12px;padding:13px;border-radius:var(--r);background:var(--bg-2);border:1px solid var(--border);cursor:pointer"><span class="badge b-verified">${n}</span><div style="flex:1"><b style="font-size:13px">${t}</b><div style="font-size:11.5px;color:var(--text-faint)">${d}</div></div><span style="color:var(--text-faint)">→</span></div>`).join("") + `</div>`;
}

RENDER.library = function () {
  const filters = [["all", "All Dances"], ["show-ready", "Show-Ready"], ["sim-verified", "Sim-Verified"], ["draft", "Draft"]];
  let list = S.dances.slice();
  if (S.danceFilter !== "all") list = list.filter(d => d.status === S.danceFilter);
  if (S.search) list = list.filter(d => d.name.toLowerCase().includes(S.search));
  const addCard = `<div class="dcard" style="border-style:dashed;display:grid;place-items:center;min-height:214px" data-nav="studio"><div style="text-align:center;color:var(--text-faint)"><div style="width:44px;height:44px;border-radius:12px;background:var(--bg-3);display:grid;place-items:center;margin:0 auto 10px">${svg('<path d="M12 5v14M5 12h14"/>', "var(--accent)")}</div><b style="font-size:13px;color:var(--text-dim)">New Dance</b><div style="font-size:11.5px">Upload a reference video</div></div></div>`;
  $("#library").innerHTML = `
    <div class="pills">${filters.map(([k, l]) => `<span class="pill ${S.danceFilter === k ? "on" : ""}" data-filter="${k}">${l}</span>`).join("")}</div>
    <div class="dance-grid">${list.map(danceCard).join("")}${addCard}</div>`;
};
function danceCard(d) {
  const meta = d.status === "show-ready" ? `${(d.repeatability && d.repeatability.consecutive_clean) || 0}/${d.repeatability_target} clean runs`
    : d.policy_path ? "policy attached" : d.motion_csv ? "motion vetted" : "in progress";
  return `<div class="dcard" data-open-dance="${esc(d.id)}"><div class="dthumb ${thumbFor(d.id)}"><div class="play">${ICON.play}</div>${d.duration_s ? `<span class="dur">${fmtDur(d.duration_s)}</span>` : ""}</div><div class="dbody"><div class="dn">${esc(d.name)} ${STATUS_BADGE[d.status] || ""}</div><div class="dm">${d.duration_s ? fmtDur(d.duration_s) + " · " : ""}${esc(meta)}</div></div></div>`;
}

async function openDance(id) {
  let d; try { d = await api("/api/dances/" + id); } catch (e) { return toast(e.message, "err"); }
  const prev = d.preview ? (d.preview.startsWith("/") ? d.preview : "/previews/" + d.preview.split("/").pop()) : null;
  const vetRows = d.vet && d.vet.checks ? Object.entries(d.vet.checks).map(([k, v]) => `<tr><td>${esc(k)}</td><td style="text-align:right">${v.pass === false ? '<span class="badge b-warn">CHECK</span>' : '<span class="badge b-ready">PASS</span>'}</td></tr>`).join("") : "";
  const bg = el(`<div class="modal-bg"><div class="modal" style="max-width:560px">
    <h3>${esc(d.name)} ${STATUS_BADGE[d.status] || ""}</h3>
    <p>${d.duration_s ? fmtDur(d.duration_s) + " · " : ""}${d.policy_path ? "policy attached" : "no policy yet"} · ${(d.repeatability && d.repeatability.consecutive_clean) || 0}/${d.repeatability_target} clean sim runs</p>
    ${prev ? `<video class="preview" src="${esc(prev)}" controls muted onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'empty',innerHTML:'preview unavailable'}))"></video>` : `<div class="empty" style="padding:24px">No preview rendered yet</div>`}
    ${vetRows ? `<div class="section-title">Vetting</div><table>${vetRows}</table>` : ""}
    <div class="row"><button class="btn btn-ghost" id="dClose">Close</button>${d.status === "draft" && !d.policy_path ? `<button class="btn btn-ghost" id="dAttach">Attach policy…</button>` : ""}</div>
  </div></div>`);
  $("#modalRoot").appendChild(bg);
  $("#dClose", bg).onclick = () => bg.remove();
  bg.onclick = (e) => { if (e.target === bg) bg.remove(); };
  const at = $("#dAttach", bg);
  if (at) at.onclick = async () => {
    const p = await modal({ title: "Attach a trained policy", body: "Path to the exported policy file (project-relative). Attaching resets verification — the sim-exam must run again.", input: "data/policies/…/policy.onnx" });
    if (!p) return;
    try { await api("/api/dances/" + id + "/policy", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ policy_path: p }) }); toast("Policy attached — re-run the sim-exam", "ok"); bg.remove(); await refreshDances(); RENDER[S.cur](); }
    catch (e) { toast(e.message, "err"); }
  };
}
document.addEventListener("click", (e) => {
  const dc = e.target.closest("[data-open-dance]"); if (dc) openDance(dc.dataset.openDance);
  const pl = e.target.closest("[data-filter]"); if (pl) { S.danceFilter = pl.dataset.filter; RENDER.library(); }
});

RENDER.studio = async function () {
  await refreshJobs();
  const jobsList = S.jobs.map(jobRow).join("") || `<p class="muted" style="font-size:12.5px">No jobs yet.</p>`;
  const detail = S.selJob ? jobDetail(S.jobs.find(j => j.id === S.selJob)) : `<div class="empty"><div class="ei">${svg('<path d="M4 4v16h16"/><path d="M4 15l5-5 4 3 5-6"/>')}</div><h3>Select or create a job</h3><p>Pick a job on the left, or drop a video above to start a new dance.</p></div>`;
  $("#studio").innerHTML = `
    <div class="dropzone" id="dropzone">
      <div class="dz-ic">${svg('<path d="M12 15V3M7 8l5-5 5 5"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>')}</div>
      <h3>Drop a reference dance video</h3>
      <p>One dancer · whole body visible · camera still · 15s–4min · no floorwork · stay on one spot</p>
      <button class="btn btn-primary" id="pickBtn" style="margin-top:16px">Choose file…</button>
    </div>
    <div class="grid g-2" style="align-items:start">
      <div class="card"><div class="card-h"><div class="ico">${svg('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18"/>')}</div><h3>Jobs</h3></div>${jobsList}</div>
      <div class="card" id="jobDetail">${detail}</div>
    </div>`;
  $("#pickBtn").onclick = () => $("#filePicker").click();
  const dz = $("#dropzone");
  dz.ondragover = (e) => { e.preventDefault(); dz.style.borderColor = "var(--accent)"; };
  dz.ondragleave = () => dz.style.borderColor = "";
  dz.ondrop = (e) => { e.preventDefault(); dz.style.borderColor = ""; if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); };
  if (S.selJob) loadJobLog(S.selJob);
};
function jobRow(j) {
  const st = j.stages[j.current_stage] || {};
  const state = st.state || "done";
  const label = j.current_stage ? `${state} at ${j.current_stage}` : "complete";
  return `<div class="job-row ${S.selJob === j.id ? "sel" : ""}" data-job="${esc(j.id)}"><div style="flex:1"><div class="jn">${esc(j.name)}</div><div class="jm">${esc(label)}</div></div>${state === "running" ? '<span class="dot live" style="background:var(--violet)"></span>' : state === "blocked" ? '<span class="badge b-warn">blocked</span>' : state === "failed" ? '<span class="badge" style="background:var(--danger-dim);color:#fecaca">failed</span>' : ""}</div>`;
}
function jobDetail(j) {
  if (!j) return "";
  const steps = S.stageOrder.map((name, i) => {
    const st = j.stages[name] || {};
    const cls = st.state === "done" ? "done" : st.state === "running" ? "active" : st.state === "blocked" ? "blocked" : st.state === "failed" ? "failed" : "";
    const lbl = { extract: "Extract", retarget: "Retarget", train: "Train", verify: "Sim-Exam", export: "Export" }[name] || name;
    return `<div class="step ${cls}"><div class="sdot">${st.state === "done" ? ICON.check : (i + 1)}</div><div class="slbl">${lbl}</div><div class="sst">${esc(st.message || st.state || "—")}</div></div>`;
  }).join("");
  const cur = j.stages[j.current_stage] || {};
  const canRetry = ["failed", "blocked"].includes(cur.state);
  const prev = j.preview_url;
  const vet = j.vet && j.vet.checks ? Object.entries(j.vet.checks).map(([k, v]) => `<tr><td>${esc(k)}</td><td style="text-align:right">${v.pass === false ? '<span class="badge b-warn">CHECK</span>' : '<span class="badge b-ready">PASS</span>'}</td></tr>`).join("") : "";
  return `<div class="card-h"><div class="ico">${svg('<path d="M4 4v16h16"/><path d="M4 15l5-5 4 3 5-6"/>')}</div><h3>${esc(j.name)}</h3>${canRetry ? `<button class="btn btn-ghost btn-sm" id="retryBtn" style="margin-left:auto">Retry</button>` : ""}</div>
    <div class="stepper">${steps}</div>
    ${cur.state === "blocked" ? `<div class="hint" style="color:var(--warn)">${svg('<path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/>')}${esc(cur.message || "waiting on the cloud GPU")}</div>` : ""}
    ${prev ? `<div class="section-title">Preview</div><video class="preview" src="${esc(prev)}" controls muted onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'empty',innerHTML:'preview unavailable'}))"></video>` : ""}
    ${vet ? `<div class="section-title">Vetting Report</div><table>${vet}</table>` : ""}
    <div class="section-title">Log</div><div class="stage-log" id="jobLog">loading…</div>`;
}
document.addEventListener("click", async (e) => {
  const jr = e.target.closest("[data-job]");
  if (jr) { S.selJob = jr.dataset.job; RENDER.studio(); }
  if (e.target.id === "retryBtn") { e.target.disabled = true; try { await api(`/api/jobs/${S.selJob}/retry`, { method: "POST" }); toast("Stage re-queued", "ok"); await refreshJobs(); RENDER.studio(); } catch (err) { toast(err.message, "err"); e.target.disabled = false; } }
});
async function loadJobLog(id) { try { const d = await api("/api/jobs/" + id); const l = $("#jobLog"); if (l) l.textContent = (d.log_tail || []).join("\n") || "(no log yet)"; } catch { } }

// file upload (audit: button disabled during request; size cap enforced server-side)
$("#filePicker") && ($("#filePicker").onchange = (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); e.target.value = ""; });
async function uploadFile(file) {
  toast(`Uploading ${file.name}…`, "info");
  const btn = $("#pickBtn"); if (btn) btn.disabled = true;
  const fd = new FormData(); fd.append("video", file);
  try { const j = await api("/api/jobs/upload", { method: "POST", body: fd }); toast("Job created — pipeline started", "ok"); S.selJob = j.id; await refreshJobs(); if (S.cur === "studio") RENDER.studio(); else go("studio"); }
  catch (e) { toast(e.message || "upload failed", "err"); if (btn) btn.disabled = false; }
}

RENDER.show = async function () {
  let shows = []; try { shows = await api("/api/shows"); } catch { }
  const ready = S.dances.filter(d => d.status === "show-ready");
  const hist = shows.filter(s => s.closed).slice(0, 8);
  const target = (S.dances[0] && S.dances[0].repeatability_target) || 3;
  $("#show").innerHTML = `
    <div class="grid g-2" style="align-items:start">
      <div>
        <div class="section-title" style="margin-top:0">Show-ready dances</div>
        ${ready.length ? ready.map(d => `<div class="dcard" style="cursor:pointer" data-start-show="${esc(d.id)}"><div class="dthumb ${thumbFor(d.id)}" style="height:120px"><div class="play">${ICON.play}</div>${d.duration_s ? `<span class="dur">${fmtDur(d.duration_s)}</span>` : ""}</div><div class="dbody"><div class="dn">${esc(d.name)} ${STATUS_BADGE[d.status]}</div><div class="dm">${(d.repeatability && d.repeatability.consecutive_clean) || 0}/${d.repeatability_target} clean runs · tap to start a show</div></div></div>`).join("")
      : `<div class="empty"><div class="ei">${svg('<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>')}</div><h3>No show-ready dances yet</h3><p>A dance becomes show-ready after it passes the signed sim-exam and ${target} clean runs.</p></div>`}
      </div>
      <div>
        <div class="section-title" style="margin-top:0">Show history</div>
        ${hist.length ? `<table><thead><tr><th>Dance</th><th>Operator</th><th>Result</th></tr></thead>${hist.map(s => `<tr><td><b>${esc(s.dance_name || "")}</b></td><td>${esc(s.operator || "")}</td><td>${outcomeBadge(s.outcome && s.outcome.result)}</td></tr>`).join("")}</table>` : `<p class="muted" style="font-size:12.5px">No performances recorded yet.</p>`}
        <div class="hint" style="margin-top:16px">${svg('<path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/>')}This robot has no hardware torque-cut e-stop — only the damping remote and the power switch. Keep the remote in hand for every performance.</div>
      </div>
    </div>`;
};
function outcomeBadge(r) { return r === "clean" ? '<span class="badge b-ready">Clean</span>' : r === "aborted" ? '<span class="badge b-warn">Aborted</span>' : r === "incident" ? '<span class="badge" style="background:var(--danger-dim);color:#fecaca">Incident</span>' : "—"; }

document.addEventListener("click", async (e) => {
  const ss = e.target.closest("[data-start-show]"); if (!ss) return;
  const operator = await modal({ title: "Start a show", body: "Enter the operator name running this performance.", input: "Operator name", confirmLabel: "Begin checklist" });
  if (!operator) return;
  try { const show = await api("/api/shows", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ dance_id: ss.dataset.startShow, operator }) }); runChecklist(show); }
  catch (err) { toast(err.message, "err"); }
});

// pre-show checklist wizard (server-enforced order; deploy stays typed-DEPLOY record-only)
function runChecklist(show) {
  const spec = show.checklist_spec;
  const bg = el(`<div class="modal-bg"><div class="modal" style="max-width:560px"><h3>Pre-show checklist — ${esc(show.dance_name)}</h3><p>Operator: ${esc(show.operator)}. Complete every step in order to unlock deploy.</p><div id="clBody"></div></div></div>`);
  $("#modalRoot").appendChild(bg);
  bg.onclick = (e) => { if (e.target === bg) { bg.remove(); RENDER.show(); } };
  const draw = (sh) => {
    const done = sh.steps || {};
    $("#clBody", bg).innerHTML = `<div class="checklist">` + spec.map(step => {
      const isDone = !!done[step.key];
      const val = isDone && step.kind === "number" ? ` — ${done[step.key].value}%` : "";
      return `<div class="chk-item ${isDone ? "done" : ""}"><div class="cb">${isDone ? ICON.check : ""}</div><div class="ct"><b>${esc(step.title)}${val}</b><span>${esc(step.detail)}</span></div><div class="cx">${isDone ? "" : step.kind === "number" ? `<input class="field" style="width:70px" placeholder="%" data-step="${step.key}"><button class="btn btn-ghost btn-sm" data-do="${step.key}">Save</button>` : `<button class="btn btn-ghost btn-sm" data-do="${step.key}">Confirm</button>`}</div></div>`;
    }).join("") + `</div>` + deployBlock(sh);
    $$("[data-do]", bg).forEach(b => b.onclick = async () => {
      b.disabled = true;
      const key = b.dataset.do, spc = spec.find(s => s.key === key);
      let value = true;
      if (spc.kind === "number") { const inp = $(`input[data-step="${key}"]`, bg); value = parseFloat(inp.value); if (isNaN(value)) { b.disabled = false; return toast("Enter a number", "err"); } }
      try { draw(await api(`/api/shows/${sh.id}/steps/${key}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ value }) })); }
      catch (err) { toast(err.message, "err"); b.disabled = false; }
    });
    const dep = $("#deployBtn", bg);
    if (dep) dep.onclick = async () => {
      const phrase = await modal({ title: "Deploy to robot", body: "This is <b>record-only</b> — nothing is sent to the robot. Type DEPLOY to record the authorization.", input: "type DEPLOY", confirmLabel: "Deploy", danger: true });
      if (phrase == null) return;
      try { await api(`/api/shows/${sh.id}/deploy`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ confirm_phrase: phrase }) }); toast("Deploy recorded (no robot contact)", "ok"); bg.remove(); RENDER.show(); }
      catch (err) { toast(err.message, "err"); }
    };
  };
  draw(show);
}
function deployBlock(sh) {
  if (!sh.checklist_complete) return `<div class="deploy-lock locked"><div class="dl-ic">${svg('<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>', "var(--warn)")}</div><b style="font-size:14px">Complete the checklist to unlock</b><p style="font-size:12px;color:var(--text-faint);margin:6px 0 0">Next: ${esc(sh.next_step || "")} · deploy requires typing DEPLOY</p></div>`;
  return `<div class="deploy-lock"><div class="dl-ic" style="background:var(--ok-dim)">${svg('<path d="m5 12 4 4L19 6"/>', "var(--ok)")}</div><b style="font-size:14px">Checklist complete</b><p style="font-size:12px;color:var(--text-faint);margin:6px 0 12px">Type DEPLOY to record the performance authorization</p><button class="btn btn-danger" id="deployBtn">Deploy ${esc(sh.dance_name)} →</button></div>`;
}

RENDER.system = function () {
  const sys = S.system, g = sys && sys.gpu, cost = sys && sys.cost;
  const note = `<div class="hint" style="margin-bottom:16px;background:var(--bg-1);border:1px solid var(--border);padding:11px 14px;border-radius:var(--r)">${svg('<path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/>')}The GreenNode console shows instance <b>state</b>, not live load. This panel reads the hardware directly over SSH${sys && sys.stale ? " — <b>last reading is stale</b> (box uplink flaky)." : "."}</div>`;
  if (!sys || !sys.reachable || !g) {
    $("#system").innerHTML = note + `<div class="empty"><div class="ei">${svg('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6v6H9z"/>')}</div><h3>GPU box unreachable</h3><p>${esc((sys && sys.detail) || "Can't reach the cloud box right now. Training may still be running — check back shortly.")}</p></div>`;
    return;
  }
  const jobRows = (sys.jobs || []).map(j => `<tr><td><b>${esc(j.name)}</b></td><td class="mono">${j.iteration != null ? j.iteration.toLocaleString() : "—"}${j.max_iteration ? " / " + j.max_iteration.toLocaleString() : ""}</td><td class="mono" style="color:var(--ok)">${fmtNum(j.mean_reward)}</td><td class="mono">${j.mean_episode_length != null ? Math.round(j.mean_episode_length) : "—"}</td><td style="text-align:right">${j.wandb_url ? `<a href="${esc(j.wandb_url)}" target="_blank">W&amp;B ↗</a>` : ""}</td></tr>`).join("") || `<tr><td colspan="5" class="muted">no active training jobs</td></tr>`;
  $("#system").innerHTML = note + `
    <div class="grid g-4" style="margin-bottom:18px">
      <div class="stat"><div class="lbl">GPU Utilization</div><div class="big">${g.utilization_pct ?? "?"}%</div><div class="bar" style="margin-top:8px"><i style="width:${g.utilization_pct || 0}%"></i></div></div>
      <div class="stat"><div class="lbl">VRAM</div><div class="big">${g.memory_used_mib ? (g.memory_used_mib / 1024).toFixed(1) : "?"}<span style="font-size:15px;color:var(--text-faint)"> / ${g.memory_total_mib ? Math.round(g.memory_total_mib / 1024) : "?"} GB</span></div><div class="bar" style="margin-top:8px"><i style="width:${g.memory_util_pct || 0}%"></i></div></div>
      <div class="stat"><div class="lbl">Power</div><div class="big">${g.power_w ?? "?"}<span style="font-size:15px;color:var(--text-faint)"> W</span></div><div class="sub">of 450W TDP</div></div>
      <div class="stat"><div class="lbl">Temp</div><div class="big">${g.temperature_c ?? "?"}<span style="font-size:15px;color:var(--text-faint)">°C</span></div><div class="sub">${g.temperature_c > 85 ? "hot" : "nominal"}</div></div>
    </div>
    <div class="card" style="margin-bottom:18px"><div class="card-h"><div class="ico">${svg('<path d="M4 4v16h16"/><path d="M4 14l5-5 4 3"/>')}</div><h3>Training Jobs</h3></div>
      <table><thead><tr><th>Job</th><th>Iteration</th><th>Reward</th><th>Ep-Length</th><th></th></tr></thead>${jobRows}</table></div>
    <div class="card" style="max-width:520px"><div class="card-h"><div class="ico">${svg('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>')}</div><h3>Cost this run</h3></div>
      <div style="display:flex;align-items:baseline;gap:8px"><div style="font-size:30px;font-weight:700">${vnd(cost && cost.accrued_vnd)}</div><span style="color:var(--text-faint);font-size:13px">≈ $${cost && cost.accrued_usd}</span></div>
      <div class="bar" style="margin:14px 0 8px;height:8px"><i style="width:${Math.min(100, ((cost && cost.cap_fraction) || 0) * 100)}%"></i></div>
      <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-faint)"><span>${(((cost && cost.cap_fraction) || 0) * 100).toFixed(1)}% of ${vnd(cost && cost.cap_vnd)} cap</span><span>${vnd(cost && cost.rate_vnd_per_hour)}/hr · ${cost && cost.hours}h</span></div>
      <div class="hint">${svg('<path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/>')}Billing runs creation→deletion. Delete the box (not just stop) to end charges.</div></div>`;
};

RENDER.settings = async function () {
  let cloud = {}, bm = {};
  try { cloud = await api("/api/cloud"); } catch { }
  try { bm = await api("/api/bodymodels"); } catch { }
  const cfg = cloud.config || {};
  const conn = cloud.last_test && cloud.last_test.ok;
  $("#settings").innerHTML = `
    <div class="card" style="max-width:720px">
      <div class="section-title" style="margin-top:0">Cloud Connection</div>
      <div class="set-row"><div class="si">${svg('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6v6H9z"/>')}</div><div class="st"><b>GreenNode GPU box</b><span>${esc(cfg.transport || "not configured")}${cfg.host ? " · " + esc(cfg.host) : ""}</span></div>${conn ? '<span class="badge b-ready">Connected</span>' : '<span class="badge b-draft">Untested</span>'}</div>
      <div style="display:flex;gap:12px;margin:8px 0 4px;flex-wrap:wrap">
        <input class="field cloudf" data-k="host" placeholder="host" value="${esc(cfg.host || "")}" style="flex:1;min-width:150px">
        <input class="field cloudf" data-k="port" placeholder="port" value="${esc(cfg.port || "")}" style="width:90px">
        <input class="field cloudf" data-k="user" placeholder="user" value="${esc(cfg.user || "")}" style="width:120px">
        <button class="btn btn-ghost btn-sm" id="cloudSave">Save</button>
        <button class="btn btn-primary btn-sm" id="cloudTest">Test</button>
      </div>
      <div class="section-title">Assets</div>
      <div class="set-row"><div class="si">${svg('<path d="M12 2 3 7v10l9 5 9-5V7z"/>')}</div><div class="st"><b>Body models</b><span>${bm.ready ? "SMPL + SMPL-X installed" : "not installed"}</span></div><span class="badge ${bm.ready ? "b-ready" : "b-draft"}">${bm.ready ? "Installed" : "Missing"}</span></div>
      <div class="set-row"><div class="si">${svg('<path d="M12 3v12M7 10l5 5 5-5"/><path d="M5 21h14"/>')}</div><div class="st"><b>Library backup</b><span>Download all trained dances as one portable archive</span></div><a class="btn btn-ghost btn-sm" href="/api/library/export">Export .tar.gz</a></div>
      <div class="section-title">Safety</div>
      <div class="set-row"><div class="si">${svg('<path d="M12 2 4 6v6c0 5 3.5 8 8 10 4.5-2 8-5 8-10V6z"/><path d="m9 12 2 2 4-4"/>')}</div><div class="st"><b>Signed sim-exam verdicts</b><span>Only cryptographically-signed exam results can authorize a robot deploy</span></div><span class="badge b-ready">Enforced</span></div>
      <div class="set-row"><div class="si">${svg('<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>')}</div><div class="st"><b>Repeatability target</b><span>Consecutive clean sim runs required for Show-Ready</span></div><span class="badge b-verified">${(S.dances[0] && S.dances[0].repeatability_target) || 3}</span></div>
    </div>`;
  $("#cloudSave").onclick = async (ev) => { ev.target.disabled = true; try { const patch = {}; $$(".cloudf").forEach(i => { if (i.value) patch[i.dataset.k] = i.dataset.k === "port" ? parseInt(i.value) : i.value; }); await api("/api/cloud/config", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(patch) }); toast("Cloud config saved", "ok"); } catch (e) { toast(e.message, "err"); } ev.target.disabled = false; };
  $("#cloudTest").onclick = async (ev) => { ev.target.disabled = true; ev.target.textContent = "Testing…"; try { const r = await api("/api/cloud/test", { method: "POST" }); toast(r.ok ? "Connected: " + (r.detail || "ok") : "Failed: " + (r.detail || "no"), r.ok ? "ok" : "err"); RENDER.settings(); } catch (e) { toast(e.message, "err"); ev.target.disabled = false; ev.target.textContent = "Test"; } };
};

// ---------- small render utils ----------
function svg(inner, stroke) { return `<svg viewBox="0 0 24 24" fill="none" stroke="${stroke || "currentColor"}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${inner}</svg>`; }
function fmtNum(n) { return n == null ? "—" : (Math.round(n * 10) / 10).toString(); }
function shortJob(n) { return String(n || "").replace(/^train-/, "").replace(/-a\d+$/, ""); }
function epPct(j) { return j.mean_episode_length ? Math.round(100 * j.mean_episode_length / 500) + "% survival" : "training"; }

// ---------- search ----------
$("#search") && ($("#search").oninput = (e) => { S.search = e.target.value.toLowerCase(); if (S.cur === "library") RENDER.library(); });

// ---------- boot ----------
async function boot() {
  try { const h = await api("/api/health"); if (h.stage_order) S.stageOrder = h.stage_order; } catch { }
  await Promise.all([refreshSystem(), refreshDances()]);
  RENDER.dashboard();
  setInterval(refreshSystem, 20000);
  setInterval(async () => { await refreshDances(); if (S.cur === "studio") await refreshJobs(); }, 30000);
}
boot();

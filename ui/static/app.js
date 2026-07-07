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
const S = { system: null, dances: [], jobs: [], setlists: [], stageOrder: ["extract", "retarget", "train", "verify", "export"], selJob: null, cur: "dashboard", danceFilter: "all", search: "", showTab: "perform", showMode: "live", selSetlist: null };

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
async function refreshSetlists() { try { S.setlists = await api("/api/setlists"); } catch { } }

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
  return `<div class="dcard" data-open-dance="${esc(d.id)}"><div class="dthumb ${thumbFor(d.id)}"><div class="play">${ICON.play}</div>${d.duration_s ? `<span class="dur">${fmtDur(d.duration_s)}</span>` : ""}${d.audio ? '<span class="aud-badge" title="has music">♪</span>' : ""}</div><div class="dbody"><div class="dn">${esc(d.name)} ${STATUS_BADGE[d.status] || ""}</div><div class="dm">${d.duration_s ? fmtDur(d.duration_s) + " · " : ""}${esc(meta)}</div></div></div>`;
}

async function openDance(id) {
  let d; try { d = await api("/api/dances/" + id); } catch (e) { return toast(e.message, "err"); }
  // prefer the music-muxed preview when the dance has audio
  const muxed = d.audio && d.audio.muxed_preview;
  const prev = muxed || (d.preview ? (d.preview.startsWith("/") ? d.preview : "/previews/" + d.preview.split("/").pop()) : null);
  const vetRows = d.vet && d.vet.checks ? Object.entries(d.vet.checks).map(([k, v]) => `<tr><td>${esc(k)}</td><td style="text-align:right">${v.pass === false ? '<span class="badge b-warn">CHECK</span>' : '<span class="badge b-ready">PASS</span>'}</td></tr>`).join("") : "";
  const audioRow = d.audio
    ? `<div class="music-cue"><div style="flex:1">${svg('<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>')} <b>${d.audio.source === "placeholder_click_track" ? "Placeholder track" : "Music attached"}</b> · delay ${d.audio.align ? d.audio.align.audio_delay_s : "?"}s</div><audio controls src="/api/dances/${esc(d.id)}/audio-file" style="height:30px"></audio><button class="btn btn-ghost btn-sm" id="dMusic">Replace</button><button class="btn btn-ghost btn-sm" id="dMusicX">Remove</button></div>`
    : `<div class="row" style="margin:8px 0"><button class="btn btn-ghost btn-sm" id="dMusic">${svg('<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>')} Attach music…</button><span class="muted" style="font-size:11.5px;align-self:center">silent — add a track for shows</span></div>`;
  const bg = el(`<div class="modal-bg"><div class="modal" style="max-width:560px">
    <h3>${esc(d.name)} ${STATUS_BADGE[d.status] || ""} ${d.audio ? '<span class="aud-inline">♪</span>' : ""}</h3>
    <p>${d.duration_s ? fmtDur(d.duration_s) + " · " : ""}${d.policy_path ? "policy attached" : "no policy yet"} · ${(d.repeatability && d.repeatability.consecutive_clean) || 0}/${d.repeatability_target} clean sim runs</p>
    ${prev ? `<video class="preview" src="${esc(prev)}" controls ${muxed ? "" : "muted"} onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'empty',innerHTML:'preview unavailable'}))"></video>` : `<div class="empty" style="padding:24px">No preview rendered yet</div>`}
    ${audioRow}
    ${vetRows ? `<div class="section-title">Vetting</div><table>${vetRows}</table>` : ""}
    ${d.policy_path ? `<div class="section-title">Policy versions</div><div id="dVersions" class="muted" style="font-size:12px">loading…</div>` : ""}
    <div class="row"><button class="btn btn-ghost" id="dClose">Close</button>${d.status === "draft" && !d.policy_path ? `<button class="btn btn-ghost" id="dAttach">Attach policy…</button>` : ""}${d.status === "sim-verified" ? `<button class="btn btn-primary" id="dPromote">Promote to Show-Ready</button>` : ""}</div>
    <div id="dPromoteErr" class="hint" style="color:var(--danger);margin-top:8px;display:none"></div>
  </div></div>`);
  $("#modalRoot").appendChild(bg);
  $("#dClose", bg).onclick = () => bg.remove();
  bg.onclick = (e) => { if (e.target === bg) bg.remove(); };
  // policy version history + rollback (populated async)
  if (d.policy_path) loadVersions(d.id, bg);
  const pBtn = $("#dPromote", bg);
  if (pBtn) pBtn.onclick = async () => {
    const errEl = $("#dPromoteErr", bg);
    errEl.style.display = "none";
    pBtn.disabled = true;
    try {
      await api(`/api/dances/${esc(d.id)}/promote`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ status: "show-ready" }) });
      toast("Promoted to Show-Ready", "ok");
      bg.remove(); await refreshDances(); if (RENDER[S.cur]) RENDER[S.cur]();
    } catch (e) {
      // server explains why (e.g. "2/3 consecutive clean sim runs" or a policy swap)
      errEl.textContent = e.message; errEl.style.display = "block"; pBtn.disabled = false;
    }
  };
  const reopen = async () => { bg.remove(); await refreshDances(); if (RENDER[S.cur]) RENDER[S.cur](); openDance(id); };
  const mBtn = $("#dMusic", bg);
  if (mBtn) mBtn.onclick = async () => {
    const choice = await modal({ title: "Attach music", body: "Path to a music file (project-relative or absolute), or leave blank to generate a placeholder click track to preview sync.", input: "data/audio/…/song.mp3 (or blank)", confirmLabel: "Attach" });
    if (choice == null) return;
    const body = choice.trim() ? { source_path: choice.trim() } : { bpm: 118 };
    try { await api("/api/dances/" + id + "/audio", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }); toast("Music attached", "ok"); reopen(); }
    catch (e) { toast(e.message, "err"); }
  };
  const mX = $("#dMusicX", bg);
  if (mX) mX.onclick = async () => { try { await api("/api/dances/" + id + "/audio", { method: "DELETE" }); toast("Music removed", "info"); reopen(); } catch (e) { toast(e.message, "err"); } };
  const at = $("#dAttach", bg);
  if (at) at.onclick = async () => {
    const p = await modal({ title: "Attach a trained policy", body: "Path to the exported policy file (project-relative). Attaching resets verification — the sim-exam must run again.", input: "data/policies/…/policy.onnx" });
    if (!p) return;
    try { await api("/api/dances/" + id + "/policy", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ policy_path: p }) }); toast("Policy attached — re-run the sim-exam", "ok"); bg.remove(); await refreshDances(); RENDER[S.cur](); }
    catch (e) { toast(e.message, "err"); }
  };
}
// policy version history + rollback (safety net across retrains)
async function loadVersions(id, bg) {
  const box = $("#dVersions", bg); if (!box) return;
  let vs; try { vs = (await api("/api/dances/" + id + "/versions")).versions || []; }
  catch { box.textContent = "unavailable"; return; }
  if (!vs.length) { box.textContent = "no snapshots yet — a version is saved each time this dance is promoted to show-ready."; return; }
  box.innerHTML = vs.map(v => {
    const when = v.at_epoch ? new Date(v.at_epoch * 1000).toLocaleString() : "";
    return `<div class="row" style="justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)"><div style="min-width:0"><b style="font-family:monospace">${esc((v.version_id || "").slice(0, 10))}</b> <span class="muted" style="font-size:11.5px">${esc(v.note || "")}${when ? " · " + esc(when) : ""}</span></div><button class="btn btn-ghost btn-sm" data-rollback="${esc(v.version_id)}">Roll back</button></div>`;
  }).join("");
  box.querySelectorAll("[data-rollback]").forEach(btn => {
    btn.onclick = async () => {
      const ok = await modal({ title: "Roll back policy?", body: "Restores this version's files and RESETS the dance to draft — you re-run the sim exam to re-promote. The show-ready gate is never bypassed.", confirmLabel: "Roll back", danger: true });
      if (!ok) return;
      try { await api("/api/dances/" + id + "/rollback", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ version_id: btn.dataset.rollback }) }); toast("Rolled back — re-run the sim exam to re-promote", "ok"); bg.remove(); await refreshDances(); if (RENDER[S.cur]) RENDER[S.cur](); openDance(id); }
      catch (e) { toast(e.message, "err"); }
    };
  });
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
  // Human gate: train blocked waiting for the operator's preview sign-off.
  const needsApproval = j.current_stage === "train" && cur.state === "blocked" && /approv/i.test(cur.message || "");
  const prev = j.preview_url;
  const vet = j.vet && j.vet.checks ? Object.entries(j.vet.checks).map(([k, v]) => `<tr><td>${esc(k)}</td><td style="text-align:right">${v.pass === false ? '<span class="badge b-warn">CHECK</span>' : '<span class="badge b-ready">PASS</span>'}</td></tr>`).join("") : "";
  return `<div class="card-h"><div class="ico">${svg('<path d="M4 4v16h16"/><path d="M4 15l5-5 4 3 5-6"/>')}</div><h3>${esc(j.name)}</h3>${needsApproval ? `<button class="btn btn-sm" id="approveTrainBtn" style="margin-left:auto">Approve training</button>` : canRetry ? `<button class="btn btn-ghost btn-sm" id="retryBtn" style="margin-left:auto">Retry</button>` : ""}</div>
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
  if (e.target.id === "approveTrainBtn") {
    if (!confirm("Approve training? This starts a ~2-3 h GPU run on the cloud box. Only approve after reviewing the motion preview.")) return;
    e.target.disabled = true;
    try { await api(`/api/jobs/${S.selJob}/approve-train`, { method: "POST" }); toast("Training approved — job queued", "ok"); await refreshJobs(); RENDER.studio(); }
    catch (err) { toast(err.message, "err"); e.target.disabled = false; }
  }
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
  await Promise.all([refreshSetlists(), (async () => { try { S.showsCache = await api("/api/shows"); } catch { S.showsCache = []; } })()]);
  const tabs = [["perform", "Perform"], ["setlists", "Set Lists"], ["timeline", "Timeline"]];
  const modeToggle = `<div class="mode-toggle ${S.showMode}">
    <button class="mt-btn ${S.showMode === "live" ? "on" : ""}" data-showmode="live">● Live</button>
    <button class="mt-btn ${S.showMode === "rehearsal" ? "on" : ""}" data-showmode="rehearsal">▷ Rehearsal</button></div>`;
  const banner = S.showMode === "rehearsal"
    ? `<div class="rehearsal-banner">${svg('<path d="M8 5v14l11-7z"/>')}<b>REHEARSAL MODE</b> — dry run. Outcomes are logged separately and never change a dance's show-ready status.</div>` : "";
  $("#show").innerHTML = `
    <div class="show-head">
      <div class="subtabs">${tabs.map(([k, l]) => `<button class="subtab ${S.showTab === k ? "on" : ""}" data-showtab="${k}">${l}</button>`).join("")}</div>
      ${modeToggle}
    </div>
    ${banner}
    <div id="showBody"></div>`;
  ({ perform: showPerform, setlists: showSetlists, timeline: showTimeline }[S.showTab] || showPerform)();
};
function outcomeBadge(r) { return r === "clean" ? '<span class="badge b-ready">Clean</span>' : r === "aborted" ? '<span class="badge b-warn">Aborted</span>' : r === "incident" ? '<span class="badge" style="background:var(--danger-dim);color:#fecaca">Incident</span>' : "—"; }
const esE_STOP_HINT = `<div class="hint" style="margin-top:16px">${svg('<path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/>')}This robot has no hardware torque-cut e-stop — only the damping remote and the power switch. Keep the remote in hand for every performance.</div>`;

function showPerform() {
  const ready = S.dances.filter(d => d.status === "show-ready");
  const hist = (S.showsCache || []).filter(s => s.closed).slice(0, 10);
  const target = (S.dances[0] && S.dances[0].repeatability_target) || 3;
  $("#showBody").innerHTML = `
    <div id="venueBar" class="muted" style="font-size:12px;margin-bottom:12px">loading venue…</div>
    <div class="grid g-2" style="align-items:start">
      <div>
        <div class="section-title" style="margin-top:0">Show-ready dances</div>
        ${ready.length ? ready.map(d => `<div class="dcard" style="cursor:pointer" data-start-show="${esc(d.id)}"><div class="dthumb ${thumbFor(d.id)}" style="height:120px"><div class="play">${ICON.play}</div>${d.duration_s ? `<span class="dur">${fmtDur(d.duration_s)}</span>` : ""}${d.audio ? '<span class="aud-badge" title="has music">♪</span>' : ""}</div><div class="dbody"><div class="dn">${esc(d.name)} ${STATUS_BADGE[d.status]}</div><div class="dm">${(d.repeatability && d.repeatability.consecutive_clean) || 0}/${d.repeatability_target} clean runs · tap for the pre-show checklist</div><div class="row" style="margin-top:9px">${d.audio ? `<button class="btn btn-danger btn-sm" data-run-show="${esc(d.id)}">▶ RUN SHOW</button>` : `<button class="btn btn-ghost btn-sm" disabled title="attach music before running a show">▶ RUN SHOW</button>`}</div></div></div>`).join("")
      : `<div class="empty"><div class="ei">${svg('<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>')}</div><h3>No show-ready dances yet</h3><p>A dance becomes show-ready after it passes the signed sim-exam and ${target} clean runs.</p></div>`}
      </div>
      <div>
        <div class="section-title" style="margin-top:0">Show history</div>
        ${hist.length ? `<table><thead><tr><th>Dance</th><th>Operator</th><th>Mode</th><th>Result</th></tr></thead>${hist.map(s => `<tr><td><b>${esc(s.dance_name || "")}</b></td><td>${esc(s.operator || "")}</td><td>${s.mode === "rehearsal" ? '<span class="badge b-draft">Rehearsal</span>' : '<span class="badge b-verified">Live</span>'}</td><td>${outcomeBadge(s.outcome && s.outcome.result)}</td></tr>`).join("")}</table>` : `<p class="muted" style="font-size:12.5px">No performances recorded yet.</p>`}
        ${esE_STOP_HINT}
      </div>
    </div>`;
  loadVenueBar();
}
// venue selector (drives the vet excursion limit) + the who-controls-when show-phase strip
async function loadVenueBar() {
  const bar = $("#venueBar"); if (!bar) return;
  let vdata, phases;
  try { vdata = await api("/api/venues"); } catch { bar.textContent = "venue unavailable"; return; }
  try { phases = (await api("/api/show-phases")).phases || []; } catch { phases = []; }
  const a = vdata.active || {};
  const opts = (vdata.venues || []).map(v => `<option value="${esc(v.id)}" ${v.id === a.id ? "selected" : ""}>${esc(v.name)} — ${v.max_excursion_m}m</option>`).join("");
  const strip = phases.length ? `<div class="row" style="gap:6px;flex-wrap:wrap;margin-top:8px">` + phases.map((p, i) =>
    `<span class="badge ${p.owner.includes("policy") ? "b-verified" : "b-draft"}" title="${esc(p.note)}">${esc(p.phase)} · ${esc(p.owner)}</span>${i < phases.length - 1 ? '<span style="color:var(--text-faint)">→</span>' : ""}`).join("") + `</div>` : "";
  bar.innerHTML = `<div class="row" style="align-items:center;gap:10px"><b>Venue:</b>
      <select id="venueSel" class="field" style="width:auto;padding:5px 8px">${opts}</select>
      <span class="muted">excursion limit <b>${a.max_excursion_m}m</b> (radius ${a.radius_m} − margin ${a.margin_m})</span>
      <button class="btn btn-ghost btn-sm" id="venueAdd">+ Add venue</button></div>${strip}`;
  $("#venueSel", bar).onchange = async (e) => {
    try { await api("/api/venues/active", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ key: e.target.value }) }); toast("Active venue set — the vet gate now uses its limit", "ok"); loadVenueBar(); }
    catch (err) { toast(err.message, "err"); }
  };
  $("#venueAdd", bar).onclick = async () => {
    const name = await modal({ title: "Add venue", body: "Name this venue (e.g. 'Client stage 3×3m').", input: "Venue name", confirmLabel: "Next" });
    if (!name) return;
    const radius = await modal({ title: "Dance-area radius (m)", body: "Half the smallest floor dimension the robot can use. Excursion limit = radius − 0.5m margin.", input: "e.g. 2.0", confirmLabel: "Add venue" });
    if (!radius) return;
    try { await api("/api/venues", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, radius_m: parseFloat(radius), make_active: true }) }); toast("Venue added + set active", "ok"); loadVenueBar(); }
    catch (err) { toast(err.message, "err"); }
  };
}

function showSetlists() {
  const sel = S.selSetlist && S.setlists.find(s => s.id === S.selSetlist);
  $("#showBody").innerHTML = `
    <div class="grid" style="grid-template-columns:280px 1fr;gap:18px;align-items:start">
      <div class="card">
        <div class="card-h"><h3>Set Lists</h3><button class="btn btn-primary btn-sm" id="newSetlist" style="margin-left:auto">+ New</button></div>
        ${S.setlists.length ? S.setlists.map(sl => `<div class="setlist-row ${S.selSetlist === sl.id ? "sel" : ""}" data-setlist="${esc(sl.id)}"><div style="flex:1"><b>${esc(sl.name)}</b><div class="jm">${sl.count} dance${sl.count === 1 ? "" : "s"} · ${fmtDur(sl.total_runtime_s)}</div></div>${sl.show_ready ? '<span class="badge b-ready">Ready</span>' : sl.count ? '<span class="badge b-warn">Blocked</span>' : '<span class="badge b-draft">Empty</span>'}</div>`).join("") : `<p class="muted" style="font-size:12.5px">No set-lists yet. Create one to sequence a show.</p>`}
      </div>
      <div id="setlistEditor">${sel ? setlistEditor(sel) : `<div class="empty"><div class="ei">${svg('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M9 4v16"/>')}</div><h3>Select or create a set-list</h3><p>Arrange several show-ready dances into one performance with gaps and music.</p></div>`}</div>
    </div>`;
}

function setlistEditor(sl) {
  const dmap = Object.fromEntries(S.dances.map(d => [d.id, d]));
  const rows = sl.items.map((it, i) => `
    <div class="sl-item ${it.show_ready ? "" : "blocked"}">
      <div class="sl-ord">${i + 1}</div>
      <div class="sl-main"><b>${esc(it.name)}</b> ${it.present ? (STATUS_BADGE[it.status] || "") : '<span class="badge b-warn">missing</span>'} ${it.has_audio ? '<span class="aud-inline">♪</span>' : ""}<div class="jm">${it.duration_s ? fmtDur(it.duration_s) : "—"}${it.note ? " · " + esc(it.note) : ""}</div></div>
      <div class="sl-gap">${i < sl.items.length - 1 ? `<input class="field sl-gapf" style="width:56px" value="${it.gap_after_s}" data-idx="${i}" title="gap after (s)"><span class="jm">s gap</span>` : ""}</div>
      <div class="sl-ctl">
        <button class="btn btn-ghost btn-sm" data-slmove="${i}:-1" ${i === 0 ? "disabled" : ""}>↑</button>
        <button class="btn btn-ghost btn-sm" data-slmove="${i}:1" ${i === sl.items.length - 1 ? "disabled" : ""}>↓</button>
        <button class="btn btn-ghost btn-sm" data-slremove="${i}">✕</button>
      </div></div>`).join("") || `<p class="muted" style="font-size:12.5px">Empty — add dances below.</p>`;
  const addable = S.dances.filter(d => !sl.items.some(it => it.dance_id === d.id));
  return `
    <div class="card">
      <div class="card-h"><h3>${esc(sl.name)}</h3>
        <span class="badge ${sl.show_ready ? "b-ready" : sl.count ? "b-warn" : "b-draft"}" style="margin-left:auto">${sl.show_ready ? "Show-ready" : sl.count ? sl.blockers.length + " not ready" : "empty"}</span>
        <button class="btn btn-ghost btn-sm" id="renameSetlist">Rename</button>
        <button class="btn btn-ghost btn-sm" id="deleteSetlist">Delete</button>
      </div>
      <div class="sl-list">${rows}</div>
      <div class="sl-total">Total runtime <b>${fmtDur(sl.total_runtime_s)}</b> · ${sl.count} number${sl.count === 1 ? "" : "s"}${sl.blockers.length ? ` · <span style="color:var(--warn)">${sl.blockers.map(b => esc(b.name) + " (" + esc(b.reason) + ")").join(", ")}</span>` : ""}</div>
      <div class="row" style="margin-top:12px">
        ${addable.length ? `<select class="field" id="addDanceSel" style="flex:1"><option value="">+ Add a dance…</option>${addable.map(d => `<option value="${esc(d.id)}">${esc(d.name)} (${d.status})</option>`).join("")}</select>` : `<span class="muted" style="font-size:12px">All dances added.</span>`}
        <button class="btn btn-primary" data-run-setlist="${esc(sl.id)}" ${sl.show_ready ? "" : "disabled title='every dance must be show-ready'"}>${S.showMode === "rehearsal" ? "Rehearse set →" : "Run set →"}</button>
      </div>
    </div>`;
}

function showTimeline() {
  const sel = S.selSetlist && S.setlists.find(s => s.id === S.selSetlist);
  if (!sel) { $("#showBody").innerHTML = `<div class="empty"><div class="ei">${svg('<path d="M3 12h18M3 6h18M3 18h18"/>')}</div><h3>Pick a set-list</h3><p>Go to Set Lists, choose one, then view its timeline here.</p></div>`; return; }
  const total = sel.total_runtime_s || 1;
  const blocks = sel.items.map((it, i) => {
    const w = Math.max(4, 100 * (it.duration_s || 0) / total);
    const gapw = i < sel.items.length - 1 ? 100 * (it.gap_after_s || 0) / total : 0;
    return `<div class="tl-block ${it.show_ready ? "" : "blocked"}" style="width:${w}%" title="${esc(it.name)} · ${fmtDur(it.duration_s)}"><b>${esc(it.name)}</b><span>${fmtDur(it.duration_s)}${it.has_audio ? " ♪" : ""}</span></div>${gapw ? `<div class="tl-gap" style="width:${gapw}%" title="${it.gap_after_s}s gap"></div>` : ""}`;
  }).join("");
  $("#showBody").innerHTML = `
    <div class="card">
      <div class="card-h"><div class="ico">${svg('<path d="M3 12h18M3 6h18M3 18h18"/>')}</div><h3>${esc(sel.name)} — timeline</h3><span class="badge b-verified" style="margin-left:auto">${fmtDur(sel.total_runtime_s)} total</span></div>
      ${sel.count ? `<div class="timeline">${blocks}</div><div class="tl-legend"><span><i class="tl-sw"></i>dance</span><span><i class="tl-sw gap"></i>gap/transition</span><span>♪ = has music</span></div>` : `<p class="muted" style="font-size:12.5px">This set-list is empty.</p>`}
      ${sel.blockers.length ? `<div class="hint" style="color:var(--warn);margin-top:14px">${svg('<path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/>')}Not runnable yet: ${sel.blockers.map(b => esc(b.name)).join(", ")} ${sel.blockers.length === 1 ? "is" : "are"} not show-ready.</div>` : ""}
    </div>`;
}

// ---- show-mode event handlers ----
document.addEventListener("click", async (e) => {
  const tb = e.target.closest("[data-showtab]"); if (tb) { S.showTab = tb.dataset.showtab; RENDER.show(); return; }
  const mm = e.target.closest("[data-showmode]"); if (mm) { S.showMode = mm.dataset.showmode; RENDER.show(); return; }
  const slr = e.target.closest("[data-setlist]"); if (slr) { S.selSetlist = slr.dataset.setlist; showSetlists(); return; }
  if (e.target.id === "newSetlist") {
    const name = await modal({ title: "New set-list", body: "Name this show.", input: "e.g. Friday Night Set", confirmLabel: "Create" });
    if (!name) return;
    try { const sl = await api("/api/setlists", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name }) }); S.selSetlist = sl.id; await refreshSetlists(); showSetlists(); } catch (err) { toast(err.message, "err"); }
    return;
  }
  const mv = e.target.closest("[data-slmove]"); if (mv) { const [i, dir] = mv.dataset.slmove.split(":").map(Number); return reorderSetlist(i, dir); }
  const rm = e.target.closest("[data-slremove]"); if (rm) return reorderSetlist(+rm.dataset.slremove, 0, true);
  if (e.target.id === "renameSetlist") { const n = await modal({ title: "Rename set-list", input: "New name" }); if (n) { try { await api("/api/setlists/" + S.selSetlist, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: n }) }); await refreshSetlists(); showSetlists(); } catch (err) { toast(err.message, "err"); } } return; }
  if (e.target.id === "deleteSetlist") { const ok = await modal({ title: "Delete set-list?", body: "This removes the set-list (the dances themselves are untouched).", confirmLabel: "Delete", danger: true }); if (ok) { try { await api("/api/setlists/" + S.selSetlist, { method: "DELETE" }); S.selSetlist = null; await refreshSetlists(); showSetlists(); } catch (err) { toast(err.message, "err"); } } return; }
  const rs = e.target.closest("[data-run-setlist]"); if (rs) return runSetlist(rs.dataset.runSetlist);
  // RUN SHOW is the REAL live run (spawns show_run.sh); it sits inside the show-ready
  // card, so handle + return before the card's data-start-show checklist branch.
  const rsw = e.target.closest("[data-run-show]"); if (rsw) { openRunShow(rsw.dataset.runShow); return; }
  const ss = e.target.closest("[data-start-show]");
  if (ss) {
    const operator = await modal({ title: `Start ${S.showMode === "rehearsal" ? "a rehearsal" : "a show"}`, body: "Enter the operator name.", input: "Operator name", confirmLabel: "Begin checklist" });
    if (!operator) return;
    try { const show = await api("/api/shows", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ dance_id: ss.dataset.startShow, operator, mode: S.showMode }) }); runChecklist(show); }
    catch (err) { toast(err.message, "err"); }
  }
});
document.addEventListener("change", async (e) => {
  if (e.target.id === "addDanceSel" && e.target.value) {
    const sl = S.setlists.find(s => s.id === S.selSetlist); if (!sl) return;
    const items = sl.items.map(it => ({ dance_id: it.dance_id, gap_after_s: it.gap_after_s, note: it.note })).concat([{ dance_id: e.target.value, gap_after_s: 8 }]);
    await saveSetlistItems(items);
  }
  if (e.target.classList && e.target.classList.contains("sl-gapf")) {
    const sl = S.setlists.find(s => s.id === S.selSetlist); if (!sl) return;
    const idx = +e.target.dataset.idx, v = parseFloat(e.target.value);
    if (isNaN(v) || v < 0) { toast("Gap must be a non-negative number", "err"); return; }
    const items = sl.items.map((it, i) => ({ dance_id: it.dance_id, gap_after_s: i === idx ? v : it.gap_after_s, note: it.note }));
    await saveSetlistItems(items);
  }
});
async function reorderSetlist(idx, dir, remove = false) {
  const sl = S.setlists.find(s => s.id === S.selSetlist); if (!sl) return;
  let items = sl.items.map(it => ({ dance_id: it.dance_id, gap_after_s: it.gap_after_s, note: it.note }));
  if (remove) items.splice(idx, 1);
  else { const j = idx + dir; if (j < 0 || j >= items.length) return;[items[idx], items[j]] = [items[j], items[idx]]; }
  await saveSetlistItems(items);
}
async function saveSetlistItems(items) {
  try { await api("/api/setlists/" + S.selSetlist, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ items }) }); await refreshSetlists(); showSetlists(); }
  catch (err) { toast(err.message, "err"); }
}

// sequential set-list runner: walk each number through its own checklist
async function runSetlist(id) {
  const sl = S.setlists.find(s => s.id === id); if (!sl || !sl.show_ready) return toast("Every dance must be show-ready", "err");
  const operator = await modal({ title: `${S.showMode === "rehearsal" ? "Rehearse" : "Run"} “${sl.name}”`, body: `${sl.count} numbers · ${fmtDur(sl.total_runtime_s)}. You'll run each number's pre-show checklist in order.`, input: "Operator name", confirmLabel: "Start set" });
  if (!operator) return;
  for (let i = 0; i < sl.items.length; i++) {
    const it = sl.items[i];
    const go2 = await modal({ title: `Number ${i + 1}/${sl.count}: ${it.name}`, body: `${fmtDur(it.duration_s)}${it.has_audio ? " · ♪ music" : ""}. ${i > 0 ? "Previous number done. " : ""}Start this number's checklist?`, confirmLabel: "Checklist →" });
    if (go2 == null) { toast("Set stopped by operator", "info"); break; }
    try {
      const show = await api("/api/shows", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ dance_id: it.dance_id, operator, mode: S.showMode, setlist_id: id }) });
      await new Promise(res => runChecklist(show, res));
    } catch (err) { toast(err.message, "err"); break; }
  }
  RENDER.show();
}

// pre-show checklist wizard (server-enforced order; deploy stays typed-DEPLOY record-only)
function runChecklist(show, onDone) {
  const spec = show.checklist_spec;
  const dance = S.dances.find(d => d.id === show.dance_id);
  const rehearsal = show.mode === "rehearsal";
  const finish = (bg) => { bg.remove(); if (onDone) onDone(); else RENDER.show(); };
  const head = rehearsal ? `<div class="rehearsal-tag">▷ REHEARSAL — dry run, not a live performance</div>` : "";
  const music = dance && dance.audio ? `<div class="music-cue">${svg('<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>')}<div style="flex:1"><b>Music ready</b> — starts ${dance.audio.align ? dance.audio.align.audio_delay_s + "s" : ""} into the performance, on the go signal.</div><audio controls src="/api/dances/${esc(dance.id)}/audio-file" style="height:30px"></audio></div>` : "";
  const bg = el(`<div class="modal-bg"><div class="modal ${rehearsal ? "rehearsal" : ""}" style="max-width:560px">${head}<h3>Pre-show checklist — ${esc(show.dance_name)}</h3><p>Operator: ${esc(show.operator)}. Complete every step in order to unlock deploy.</p>${music}<div id="clBody"></div></div></div>`);
  $("#modalRoot").appendChild(bg);
  bg.onclick = (e) => { if (e.target === bg) finish(bg); };
  const drawOutcome = (sh) => {
    $("#clBody", bg).innerHTML = `<div class="deploy-lock"><div class="dl-ic" style="background:var(--ok-dim)">${svg('<path d="m5 12 4 4L19 6"/>', "var(--ok)")}</div><b style="font-size:14px">${rehearsal ? "Rehearsal" : "Performance"} authorized (recorded)</b><p style="font-size:12px;color:var(--text-faint);margin:6px 0 12px">Record how it went to close this ${rehearsal ? "rehearsal" : "show"}.</p><div class="row"><button class="btn btn-ghost" data-outcome="clean">Clean ✓</button><button class="btn btn-ghost" data-outcome="aborted">Aborted</button><button class="btn btn-danger" data-outcome="incident">Incident</button></div></div>`;
    $$("[data-outcome]", bg).forEach(b => b.onclick = async () => {
      try { await api(`/api/shows/${sh.id}/outcome`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ result: b.dataset.outcome }) }); toast(`Outcome recorded: ${b.dataset.outcome}${rehearsal ? " (rehearsal)" : ""}`, b.dataset.outcome === "clean" ? "ok" : "info"); await refreshDances(); finish(bg); }
      catch (err) { toast(err.message, "err"); }
    });
  };
  const draw = (sh) => {
    if (sh.deploy) return drawOutcome(sh);
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
      const phrase = await modal({ title: rehearsal ? "Rehearsal cue" : "Deploy to robot", body: "This is <b>record-only</b> — nothing is sent to the robot. Type DEPLOY to record the authorization.", input: "type DEPLOY", confirmLabel: rehearsal ? "Cue" : "Deploy", danger: !rehearsal });
      if (phrase == null) return;
      try { const r = await api(`/api/shows/${sh.id}/deploy`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ confirm_phrase: phrase }) }); toast(rehearsal ? "Rehearsal cued (no robot contact)" : "Deploy recorded (no robot contact)", "ok"); drawOutcome(r.show || sh); }
      catch (err) { toast(err.message, "err"); }
    };
  };
  draw(show);
}
function deployBlock(sh) {
  if (!sh.checklist_complete) return `<div class="deploy-lock locked"><div class="dl-ic">${svg('<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>', "var(--warn)")}</div><b style="font-size:14px">Complete the checklist to unlock</b><p style="font-size:12px;color:var(--text-faint);margin:6px 0 0">Next: ${esc(sh.next_step || "")} · deploy requires typing DEPLOY</p></div>`;
  return `<div class="deploy-lock"><div class="dl-ic" style="background:var(--ok-dim)">${svg('<path d="m5 12 4 4L19 6"/>', "var(--ok)")}</div><b style="font-size:14px">Checklist complete</b><p style="font-size:12px;color:var(--text-faint);margin:6px 0 12px">Type DEPLOY to record the performance authorization</p><button class="btn btn-danger" id="deployBtn">Deploy ${esc(sh.dance_name)} →</button></div>`;
}

// ---- one-button LIVE show (POST /api/shows/{id}/run spawns tools/show_run.sh) ----
// This is distinct from the record-only checklist deploy above: it actually launches
// the robot. The typed phrase below + the operator's hand-held damping remote are the
// deploy human-confirmation; the remote is the ONLY stop (no hardware e-stop).
const RUN_PHRASE = "I AM PRESENT WITH THE DAMPING REMOTE";

function openRunShow(danceId) {
  const d = S.dances.find(x => x.id === danceId);
  if (!d) return toast("Dance not found — refresh", "err");
  const lbl = "display:block;font-size:12px;color:var(--text-dim);margin:12px 0 4px";
  const bg = el(`<div class="modal-bg"><div class="modal" style="max-width:520px">
    <h3>Run show — ${esc(d.name)}</h3>
    <p style="font-size:12.5px;color:var(--text-faint);margin-top:4px">This launches the real robot show (full dance + music). Keep the damping remote in your hand — it is the ONLY stop.</p>
    <label style="${lbl}">Mode</label>
    <div class="mode-toggle rehearsal" id="runModeTog">
      <button class="mt-btn on" data-runmode="rehearsal" type="button">▷ Rehearsal</button>
      <button class="mt-btn" data-runmode="live" type="button">● Live</button>
    </div>
    <label style="${lbl}">Operator</label>
    <input class="field" id="runOp" value="alois" autocomplete="off">
    <label style="display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--text-dim);margin-top:12px" id="runStandWrap"><input type="checkbox" id="runStand"> Stand at end <span class="muted" style="font-size:11.5px">(experimental · unvalidated on hardware · rehearsal only)</span></label>
    <label style="${lbl}">Confirm you are present with the damping remote</label>
    <input class="field" id="runPhrase" placeholder="${esc(RUN_PHRASE)}" autocomplete="off">
    <div class="hint" style="color:var(--danger);margin-top:6px">Type exactly, character for character: <b>${esc(RUN_PHRASE)}</b></div>
    <div id="runErr" class="hint" style="color:var(--danger);margin-top:8px;display:none"></div>
    <div class="row" style="margin-top:14px"><button class="btn btn-ghost" id="runCancel">Cancel</button><button class="btn btn-danger" id="runStart" disabled>▶ Start show</button></div>
  </div></div>`);
  $("#modalRoot").appendChild(bg);
  let mode = "rehearsal";
  const startBtn = $("#runStart", bg), phraseInp = $("#runPhrase", bg),
    standCb = $("#runStand", bg), standWrap = $("#runStandWrap", bg);
  // Start unlocks ONLY on an exact phrase match (mirrors the server's 403 guard).
  const sync = () => { startBtn.disabled = phraseInp.value !== RUN_PHRASE; };
  phraseInp.oninput = sync;
  $$("[data-runmode]", bg).forEach(b => b.onclick = () => {
    mode = b.dataset.runmode;
    $$("[data-runmode]", bg).forEach(x => x.classList.toggle("on", x === b));
    $("#runModeTog", bg).className = "mode-toggle " + mode;
    const live = mode === "live";                 // stand-at-end is rehearsal-only
    standCb.disabled = live; if (live) standCb.checked = false;
    standWrap.style.opacity = live ? "0.45" : "";
  });
  $("#runCancel", bg).onclick = () => bg.remove();
  bg.onclick = (e) => { if (e.target === bg) bg.remove(); };
  startBtn.onclick = async () => {
    startBtn.disabled = true;
    const body = {
      operator: ($("#runOp", bg).value || "").trim() || "alois",
      mode, confirmation: phraseInp.value,
      exit_stand: standCb.checked && mode === "rehearsal",
    };
    try {
      const r = await api(`/api/shows/${esc(danceId)}/run`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      bg.remove();
      openRunMonitor(r.show, r.run);
    } catch (e) {
      const errEl = $("#runErr", bg); errEl.textContent = e.message; errEl.style.display = "block"; sync();
    }
  };
  phraseInp.focus();
}

// Live status panel: poll /api/shows/runs/current for phase + last lines; when the
// process exits, surface the outcome capture (reuses the /outcome endpoint).
function openRunMonitor(show, initialRun) {
  const bg = el(`<div class="modal-bg"><div class="modal" style="max-width:620px">
    <h3>${esc(show.dance_name)} — <span id="runPhase">${esc((initialRun && initialRun.phase) || "launching")}</span></h3>
    <div style="background:var(--danger);color:#fff;font-weight:800;text-align:center;padding:12px;border-radius:var(--r);letter-spacing:.5px;margin:10px 0;font-size:14px">⏹ REMOTE = ONLY STOP</div>
    <div class="muted" style="font-size:12px;margin-bottom:8px">Operator ${esc(show.operator)} · ${show.mode === "rehearsal" ? "rehearsal (never demotes the dance)" : "LIVE performance"} · <span id="runState">${initialRun && initialRun.running ? "running" : "starting…"}</span></div>
    <pre id="runLog" class="mono" style="background:var(--bg-1);border:1px solid var(--border);border-radius:var(--r);padding:10px;max-height:240px;overflow:auto;font-size:11.5px;white-space:pre-wrap;margin:0"></pre>
    <div id="runOutcome"></div>
  </div></div>`);
  $("#modalRoot").appendChild(bg);
  const phaseEl = $("#runPhase", bg), stateEl = $("#runState", bg),
    logEl = $("#runLog", bg), outEl = $("#runOutcome", bg);
  let timer = null, ended = false;
  const drawOutcome = () => {
    ended = true;
    // Reuse the existing outcome machinery (POST /api/shows/{id}/outcome ->
    // shows.record_outcome). An unresolved show blocks the next run.
    outEl.innerHTML = `<div class="deploy-lock" style="margin-top:14px"><b style="font-size:14px">Show ended — record how it went</b><p style="font-size:12px;color:var(--text-faint);margin:6px 0 12px">Required before the next run: an unresolved show is blocked.</p><div class="row"><button class="btn btn-ghost" data-run-outcome="clean">Clean ✓</button><button class="btn btn-ghost" data-run-outcome="aborted">Aborted</button><button class="btn btn-danger" data-run-outcome="incident">Incident</button></div></div>`;
    $$("[data-run-outcome]", bg).forEach(b => b.onclick = async () => {
      const result = b.dataset.runOutcome;
      try {
        await api(`/api/shows/${show.id}/outcome`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ result }) });
        toast(`Outcome recorded: ${result}${show.mode === "rehearsal" ? " (rehearsal)" : ""}`, result === "clean" ? "ok" : "info");
        bg.remove(); await refreshDances(); if (S.cur === "show") RENDER.show();
      } catch (err) { toast(err.message, "err"); }
    });
  };
  const poll = async () => {
    let st; try { st = await api("/api/shows/runs/current"); } catch { return; }
    phaseEl.textContent = st.phase || "";
    stateEl.textContent = st.running ? "running" : "exited";
    logEl.textContent = (st.last_lines || []).join("\n") || "(waiting for output…)";
    logEl.scrollTop = logEl.scrollHeight;
    if (!st.running && !ended) { if (timer) { clearInterval(timer); timer = null; } drawOutcome(); }
  };
  timer = setInterval(poll, 1000);
  poll();
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

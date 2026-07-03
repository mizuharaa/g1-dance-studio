// G1 Dance Studio frontend — talks to ui/server.py. No framework, no build step.
"use strict";

const $ = (sel) => document.querySelector(sel);
const api = async (path, opts) => {
  let r;
  try {
    r = await fetch(path, opts);
  } catch (netErr) {
    markEngine(false);   // server/process down — surface it, don't fail silently
    throw new Error("app engine not reachable");
  }
  markEngine(true);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

// Engine-down banner (audit MEDIUM): a background poll that hits a dead server used
// to fail silently, leaving stale data on screen. Show a clear offline bar instead.
function markEngine(ok) {
  let bar = $("#engine-offline");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "engine-offline";
    bar.textContent = "⚠ App engine not reachable — data may be stale. Retrying…";
    bar.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:9999;" +
      "background:#c0392b;color:#fff;padding:6px 12px;font:600 13px sans-serif;" +
      "text-align:center;display:none";
    document.body.appendChild(bar);
  }
  bar.style.display = ok ? "none" : "block";
}

// Disable a button for the duration of an async action so a double-click can't
// double-submit a job / start two shows / fire the deploy gate twice (audit MEDIUM).
async function withBusy(btn, fn) {
  if (!btn) return fn();
  if (btn.disabled) return;            // already in flight — ignore the repeat
  btn.disabled = true;
  try { return await fn(); }
  finally { btn.disabled = false; }
}

let selectedJob = null;

// ---- jobs -----------------------------------------------------------------

async function refreshJobs() {
  const jobs = await api("/api/jobs");
  const ul = $("#job-list");
  ul.innerHTML = "";
  if (!jobs.length) ul.innerHTML = '<li class="empty">No jobs yet</li>';
  for (const j of jobs) {
    const li = document.createElement("li");
    const stage = j.current_stage ? `waiting at: ${j.current_stage}` : "complete";
    li.innerHTML = `${j.name}<span class="sub">${stage} · ${j.id}</span>`;
    if (j.id === selectedJob) li.classList.add("selected");
    li.onclick = () => { selectedJob = j.id; refreshJobs(); showJob(j.id); };
    ul.appendChild(li);
  }
}

async function showJob(id) {
  const j = await api(`/api/jobs/${id}`);
  $("#job-title").textContent = `— ${j.name}`;
  const box = $("#stages");
  box.innerHTML = "";
  for (const [name, st] of Object.entries(j.stages)) {
    const div = document.createElement("div");
    div.className = "stage";
    div.innerHTML = `
      <span class="name">${name}</span>
      <div class="bar"><div style="width:${Math.round(st.progress * 100)}%"></div></div>
      <span class="state ${st.state}">${st.state}${st.message ? " · " + st.message : ""}</span>`;
    box.appendChild(div);
  }
  const cur = j.current_stage && j.stages[j.current_stage];
  const retry = $("#retry-btn");
  retry.hidden = !(cur && (cur.state === "failed" || cur.state === "blocked"));
  retry.onclick = async () => {
    try { await api(`/api/jobs/${id}/retry`, { method: "POST" }); showJob(id); }
    catch (err) { alert("Retry refused: " + err.message); }
  };
  $("#job-vet").innerHTML = j.vet ? vetHtml(j.vet) : "";
  if (j.preview_url && $("#preview-video").src !== location.origin + j.preview_url) {
    playPreview(j.preview_url);
  }
  const log = $("#job-log");
  log.hidden = !j.log_tail.length;
  log.textContent = j.log_tail.join("\n");
  $("#deploy-btn").disabled = false; // gate itself is enforced server-side
}

$("#new-job-form").onsubmit = async (e) => {
  e.preventDefault();
  const file = $("#video-file").files[0];
  if (!file) return alert("Pick a video file first.");
  const fd = new FormData();
  fd.append("video", file);
  await withBusy(e.submitter || $("#new-job-form button[type=submit]"), async () => {
    try {
      const j = await api("/api/jobs/upload", { method: "POST", body: fd });
      selectedJob = j.id;
      $("#video-file").value = "";
      await refreshJobs();
      await showJob(j.id);
    } catch (err) { alert("Could not create job: " + err.message); }
  });
};

// ---- vetting ---------------------------------------------------------------

async function loadMotions() {
  const motions = await api("/api/motions");
  const opts = motions.map((m) => `<option value="${m.path}">${m.name}</option>`).join("");
  $("#vet-select").innerHTML = opts;
  $("#csv-select").innerHTML = opts;
}

$("#csv-run").onclick = (e) => withBusy(e.currentTarget, async () => {
  const csv = $("#csv-select").value;
  if (!csv) return;
  try {
    const j = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input_path: csv }),
    });
    selectedJob = j.id;
    await refreshJobs();
    await showJob(j.id);
  } catch (err) { alert("Could not create job: " + err.message); }
});

function vetHtml(r) {
  const row = (name, c, hard) => {
    const ok = hard ? c.pass : c.ok;
    const badge = hard
      ? `<span class="badge ${ok ? "pass" : "fail"}">${ok ? "PASS" : "FAIL"}</span>`
      : `<span class="badge ${ok ? "pass" : "warn"}">${ok ? "ok" : "WARN"}</span>`;
    const detail = Object.entries(c).filter(([k]) => k !== "pass" && k !== "ok")
      .map(([k, v]) => `${k}: ${v}`).join(", ");
    return `<tr><td>${badge}</td><td>${name}</td><td>${detail}</td></tr>`;
  };
  return `
    <p class="hint">${r.file.split("/").pop()} — ${r.frames} frames, ${r.seconds.toFixed(1)} s</p>
    <table class="vet">
      <tr><th></th><th>check</th><th>details</th></tr>
      ${Object.entries(r.hard).map(([n, c]) => row(n, c, true)).join("")}
      ${Object.entries(r.advisory).map(([n, c]) => row(n, c, false)).join("")}
    </table>
    <p class="verdict">${r.pass
      ? '<span class="badge pass">DEPLOYABLE MOTION</span>'
      : '<span class="badge fail">REJECTED</span>'}</p>`;
}

function playPreview(url) {
  const v = $("#preview-video");
  v.src = url;
  v.hidden = false;
  $("#preview-hint").hidden = true;
  // audit MEDIUM: a dangling/unserved preview used to fail silently (blank box).
  v.onerror = () => {
    v.hidden = true;
    const hint = $("#preview-hint");
    hint.hidden = false;
    hint.textContent = "Preview unavailable — the render may still be in progress "
      + "or the file is missing.";
  };
}

$("#vet-run").onclick = async () => {
  const csv = $("#vet-select").value;
  if (!csv) return;
  const out = $("#vet-report");
  out.innerHTML = '<p class="hint">Running checks (loads physics model)&hellip;</p>';
  try {
    const r = await api(`/api/vet?csv=${encodeURIComponent(csv)}`);
    out.innerHTML = vetHtml(r);
  } catch (err) { out.innerHTML = `<p class="hint">Vet failed: ${err.message}</p>`; }
};

// ---- previews ---------------------------------------------------------------

async function loadPreviews() {
  const previews = await api("/api/previews");
  const ul = $("#preview-list");
  ul.innerHTML = previews.length ? "" : '<li class="empty">No previews yet</li>';
  for (const p of previews) {
    const li = document.createElement("li");
    li.innerHTML = `${p.name}<span class="sub">${(p.size / 1e6).toFixed(1)} MB</span>`;
    li.onclick = () => { playPreview(p.url); $("#preview-video").play(); };
    ul.appendChild(li);
  }
}

// ---- deploy gate (placeholder) ----------------------------------------------

$("#deploy-btn").onclick = () => {
  if (!selectedJob) return;
  delete $("#deploy-confirm").dataset.target;  // this is a STUDIO deploy (audit MEDIUM
  $("#deploy-phrase").value = "";              // leak: a cancelled show-deploy left
  $("#deploy-dialog").showModal();             // target='show' and misrouted this one)
};
$("#deploy-cancel").onclick = () => {
  delete $("#deploy-confirm").dataset.target;  // clear target on cancel too
  $("#deploy-dialog").close();
};
$("#deploy-confirm").onclick = async () => {
  try {
    const r = await api(`/api/jobs/${selectedJob}/deploy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm_phrase: $("#deploy-phrase").value }),
    });
    $("#deploy-result").textContent = r.note;
  } catch (err) {
    $("#deploy-result").textContent = "Refused: " + err.message;
  }
  $("#deploy-dialog").close();
};

// ---- cloud GPU ---------------------------------------------------------------

function cloudFieldsFor(transport) {
  $("#fields-ssh").hidden = transport !== "ssh";
  $("#fields-jupyter").hidden = transport !== "jupyter";
}

async function refreshCloud(test = false) {
  const info = await api("/api/cloud");
  const cfg = info.config;
  // audit MEDIUM: the 30s auto-refresh used to overwrite fields mid-typing. Only
  // (re)populate when the operator isn't editing the cloud form.
  const editing = $("#cloud-form").contains(document.activeElement);
  if (!editing) {
    if (cfg.transport) {
      document.querySelector(`input[name=transport][value=${cfg.transport}]`).checked = true;
      cloudFieldsFor(cfg.transport);
    }
    $("#ssh-host").value = cfg.ssh.host || "";
    $("#ssh-port").value = cfg.ssh.port || "";
    $("#ssh-user").value = cfg.ssh.user || "";
    $("#ssh-key").value = cfg.ssh.key_path || "";
    $("#jup-url").value = cfg.jupyter.url || "";
  }
  const t = test ? await api("/api/cloud/test", { method: "POST" }) : info.last_test;
  const dot = $("#cloud-dot"), status = $("#cloud-status");
  if (!cfg.transport) {
    dot.className = "dot off";
    status.textContent = "not configured — waiting for GreenNode setup";
  } else if (!t) {
    dot.className = "dot off";
    status.textContent = "configured, not tested yet";
  } else if (!t.connected) {
    dot.className = "dot bad";
    status.textContent = "disconnected: " + t.detail;
  } else {
    dot.className = t.busy ? "dot busy" : "dot ok";
    status.textContent = (t.busy ? "GPU busy — " : "connected — ") + t.detail;
  }
}

for (const r of document.querySelectorAll("input[name=transport]"))
  r.onchange = () => cloudFieldsFor(r.value);

$("#cloud-form").onsubmit = async (e) => {
  e.preventDefault();
  const transport = (document.querySelector("input[name=transport]:checked") || {}).value;
  if (!transport) return alert("Pick SSH or Jupyter first.");
  const payload = {
    transport,
    ssh: { host: $("#ssh-host").value.trim(), port: $("#ssh-port").value.trim(),
           user: $("#ssh-user").value.trim(), key_path: $("#ssh-key").value.trim(),
           password: $("#ssh-pass").value },
    jupyter: { url: $("#jup-url").value.trim(), token: $("#jup-token").value },
  };
  try {
    await api("/api/cloud/config", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    $("#cloud-status").textContent = "testing…";
    await refreshCloud(true);
  } catch (err) { alert("Cloud config failed: " + err.message); }
};

// ---- body models ---------------------------------------------------------------

async function refreshBodyModels() {
  const s = await api("/api/bodymodels");
  const btn = $("#bm-install");
  if (s.ready) {
    $("#bm-status").innerHTML = '<span class="badge pass">installed</span> SMPL + SMPL-X ready';
    btn.hidden = true;
  } else {
    const zips = s.zips.filter((z) => z.detected !== "unrecognized");
    $("#bm-status").textContent = zips.length
      ? `${zips.length} model zip(s) found — ready to install`
      : (s.hint || "missing");
    btn.hidden = !zips.length;
  }
}

$("#bm-install").onclick = async () => {
  $("#bm-status").textContent = "installing…";
  try {
    await api("/api/bodymodels/install", { method: "POST" });
  } catch (err) { alert("Install failed: " + err.message); }
  refreshBodyModels();
};

// ============================ SHOW MODE ======================================
// Operator console: dance library -> pre-show checklist -> record-only deploy.

let currentDance = null;   // dance object shown in the run panel
let currentShow = null;    // active show (performance) object

function setMode(mode) {
  for (const m of ["studio", "show", "system"]) {
    $(`#${m}-main`).hidden = m !== mode;
    $(`#mode-${m}`).classList.toggle("active", m === mode);
  }
  localStorage.setItem("g1.mode", mode);
  if (mode === "show") { refreshDances(); refreshShowHistory(); }
  if (mode === "system") refreshSystem();
}
$("#mode-studio").onclick = () => setMode("studio");
$("#mode-show").onclick = () => setMode("show");
$("#mode-system").onclick = () => setMode("system");

// ---- system monitor ----------------------------------------------------------
function fmtVnd(n) { return Math.round(n).toLocaleString("en-US") + " VND"; }

function renderSystem(s) {
  const dot = $("#sys-dot"), fresh = $("#sys-freshness");
  const gpu = s.gpu;
  if (s.reachable && !s.stale) {
    dot.className = "dot " + (gpu && gpu.busy ? "busy" : "ok");
    fresh.textContent = "live · " + new Date(s.checked_at * 1000).toLocaleTimeString();
  } else if (s.stale) {
    dot.className = "dot bad";
    fresh.textContent = "stale — box unreachable, last seen " +
      (s.last_good_at ? new Date(s.last_good_at * 1000).toLocaleTimeString() : "—");
  } else {
    dot.className = "dot off";
    fresh.textContent = s.detail || "not connected";
  }

  const g = $("#sys-gpu");
  if (gpu) {
    g.innerHTML = `
      <div class="stat"><span class="k">GPU load</span><span class="v">${gpu.utilization_pct}%</span></div>
      <div class="stat"><span class="k">VRAM</span><span class="v">${Math.round(gpu.memory_used_mib)}/${Math.round(gpu.memory_total_mib)} MiB</span></div>
      <div class="stat"><span class="k">Power</span><span class="v">${Math.round(gpu.power_w)} W</span></div>
      <div class="stat"><span class="k">Temp</span><span class="v">${Math.round(gpu.temperature_c)}°C</span></div>`;
  } else {
    g.innerHTML = `<p class="empty">${s.reachable ? "No GPU visible." : (s.detail || "Box not reachable.")}</p>`;
  }

  const jb = $("#sys-jobs");
  if (s.jobs && s.jobs.length) {
    jb.innerHTML = s.jobs.map(j => {
      const pct = j.progress != null ? Math.round(j.progress * 100) : null;
      const run = j.running ? '<span class="badge pass">running</span>'
                            : `<span class="badge warn">${j.state || "idle"}</span>`;
      const bar = pct != null ? `<div class="bar"><div class="fill" style="width:${pct}%"></div></div>` : "";
      const iter = j.iteration != null ? `iter ${j.iteration}/${j.max_iteration}` : "";
      const rew = j.mean_reward != null ? ` · reward ${j.mean_reward.toFixed(2)}` : "";
      const ep = j.mean_episode_length != null ? ` · ep-len ${Math.round(j.mean_episode_length)}` : "";
      const wb = j.wandb_url ? ` · <a href="${j.wandb_url}" target="_blank">W&amp;B ↗</a>` : "";
      return `<div class="job-row"><div class="job-head"><strong>${j.name}</strong> ${run}</div>
        <div class="hint">${iter}${rew}${ep}${wb}</div>${bar}</div>`;
    }).join("");
  } else {
    jb.innerHTML = '<p class="empty">No training jobs detected on the box.</p>';
  }

  const c = s.cost || {};
  const frac = Math.min(1, c.cap_fraction || 0);
  const capCls = c.over_cap ? "fail" : frac > 0.8 ? "warn" : "pass";
  $("#sys-cost").innerHTML = `
    <div class="stat"><span class="k">Box uptime</span><span class="v">${c.hours} h</span></div>
    <div class="stat"><span class="k">Rate</span><span class="v">${fmtVnd(c.rate_vnd_per_hour)}/h</span></div>
    <div class="stat"><span class="k">Accrued</span><span class="v">${fmtVnd(c.accrued_vnd)} (~$${c.accrued_usd})</span></div>
    <div class="bar cap"><div class="fill ${capCls}" style="width:${Math.round(frac * 100)}%"></div></div>
    <p class="hint">${fmtVnd(c.accrued_vnd)} of ${fmtVnd(c.cap_vnd)} cap (${Math.round(frac * 100)}%)</p>`;
}

async function refreshSystem() {
  try { renderSystem(await api("/api/system")); }
  catch (e) { $("#sys-freshness").textContent = "error: " + e.message; }
}

function statusBadge(d) {
  const cls = { "draft": "warn", "sim-verified": "pass", "show-ready": "pass" };
  return `<span class="badge ${cls[d.status] || "warn"}">${d.status}</span>`;
}

function repeatBadge(d) {
  const r = d.repeatability || {};
  const n = r.consecutive_clean || 0, t = d.repeatability_target || 3;
  const cls = n >= t ? "pass" : n > 0 ? "warn" : "fail";
  return `<span class="badge ${cls}" title="consecutive clean sim runs">` +
         `${n}/${t} clean</span>`;
}

async function refreshDances() {
  const dances = await api("/api/dances");
  const box = $("#dance-cards");
  box.innerHTML = dances.length ? "" :
    '<p class="empty">No dances registered yet. Train one in Studio mode.</p>';
  for (const d of dances) {
    const card = document.createElement("div");
    card.className = "dance-card";
    card.innerHTML = `
      <h3>${d.name}</h3>
      <p class="sub">${d.duration_s ? d.duration_s.toFixed(1) + " s" : "—"}</p>
      <p>${statusBadge(d)} ${repeatBadge(d)}</p>
      <p class="sub">${d.policy_path ? "policy ready" : "policy pending"}</p>`;
    card.onclick = () => openDance(d);
    box.appendChild(card);
  }
}

function openDance(d) {
  currentDance = d;
  currentShow = null;
  $("#show-library").hidden = true;
  $("#show-history-panel").hidden = true;
  $("#show-run").hidden = false;
  $("#show-dance-name").innerHTML = `${d.name} ${statusBadge(d)} ${repeatBadge(d)}`;
  const v = $("#show-preview");
  if (d.preview) {
    v.src = d.preview;
    v.hidden = false;
    v.onerror = () => { v.hidden = true; };  // don't leave a broken player (audit MEDIUM)
  } else v.hidden = true;
  $("#show-start-box").hidden = false;
  $("#checklist-box").hidden = true;
  $("#show-deploy-result").textContent = "";
}

$("#show-back").onclick = () => {
  $("#show-run").hidden = true;
  $("#show-library").hidden = false;
  $("#show-history-panel").hidden = false;
  refreshDances(); refreshShowHistory();
};

$("#show-start").onclick = (e) => withBusy(e.currentTarget, async () => {
  const operator = $("#operator-name").value.trim();
  if (!operator) return alert("Enter the operator name first.");
  if (!currentDance) return;
  try {
    currentShow = await api("/api/shows", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dance_id: currentDance.id, operator }),
    });
    $("#show-start-box").hidden = true;
    $("#checklist-box").hidden = false;
    renderChecklist();
  } catch (err) { alert("Could not start show: " + err.message); }
});

function renderChecklist() {
  const s = currentShow;
  const spec = s.checklist_spec;
  const done = Object.keys(s.steps).length;
  $("#checklist-progress").innerHTML = spec.map((st) =>
    `<span class="chip ${st.key in s.steps ? "done" : st.key === s.next_step ? "now" : ""}">` +
    `${st.title}</span>`).join("");
  const stepBox = $("#checklist-step"), deployBox = $("#show-deploy-box");
  if (s.checklist_complete) {
    stepBox.hidden = true;
    deployBox.hidden = false;
    return;
  }
  deployBox.hidden = true;
  stepBox.hidden = false;
  const st = spec.find((x) => x.key === s.next_step);
  $("#step-title").textContent = `${done + 1}/${spec.length} — ${st.title}`;
  $("#step-detail").textContent = st.detail;
  $("#step-value").hidden = st.kind !== "number";
  $("#step-confirm").textContent = st.kind === "number" ? "Record" : "Confirm";
}

$("#step-confirm").onclick = async () => {
  const s = currentShow;
  const spec = s.checklist_spec.find((x) => x.key === s.next_step);
  const value = spec.kind === "number" ? $("#step-value").value : true;
  if (spec.kind === "number" && value === "") return alert("Enter the value first.");
  try {
    currentShow = await api(`/api/shows/${s.id}/steps/${spec.key}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value: spec.kind === "number" ? value : true }),
    });
    $("#step-value").value = "";
    renderChecklist();
  } catch (err) { alert("Step refused: " + err.message); }
};

$("#show-deploy").onclick = () => {
  if (!currentShow) return;
  $("#deploy-phrase").value = "";
  $("#deploy-dialog").showModal();
  // Reuse the studio dialog but reroute its confirm to the show endpoint.
  $("#deploy-confirm").dataset.target = "show";
};

const studioDeployConfirm = $("#deploy-confirm").onclick;
$("#deploy-confirm").onclick = async () => {
  if ($("#deploy-confirm").dataset.target !== "show") return studioDeployConfirm();
  delete $("#deploy-confirm").dataset.target;
  try {
    const r = await api(`/api/shows/${currentShow.id}/deploy`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm_phrase: $("#deploy-phrase").value }),
    });
    currentShow = r.show;
    $("#show-deploy-result").textContent = r.note;
  } catch (err) {
    $("#show-deploy-result").textContent = "Refused: " + err.message;
  }
  $("#deploy-dialog").close();
};

for (const btn of document.querySelectorAll("button.outcome")) {
  btn.onclick = () => withBusy(btn, async () => {
    if (!currentShow) return;
    const result = btn.dataset.result;
    // audit MEDIUM: incident/abort is irreversible (it demotes the dance and resets
    // its clean streak). Confirm before firing so one misclick can't do it.
    if (result === "incident" || result === "aborted") {
      if (!confirm(`Record this show as "${result}"? This demotes the dance from `
                   + "show-ready and resets its clean-run streak.")) return;
    }
    try {
      await api(`/api/shows/${currentShow.id}/outcome`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ result, notes: $("#outcome-notes").value }),
      });
      $("#show-back").click();
    } catch (err) { alert("Could not record outcome: " + err.message); }
  });
}

async function refreshShowHistory() {
  const list = await api("/api/shows");
  const ul = $("#show-history");
  ul.innerHTML = list.length ? "" : '<li class="empty">No shows yet</li>';
  for (const s of list) {
    const when = new Date(s.created_at * 1000).toLocaleString();
    const result = s.outcome ? s.outcome.result :
                   s.deploy ? "deployed (record-only)" :
                   s.checklist_complete ? "checklist done" : "checklist incomplete";
    const li = document.createElement("li");
    li.innerHTML = `${s.dance_name} — ${result}` +
      `<span class="sub">${when} · operator: ${s.operator}` +
      `${s.outcome && s.outcome.notes ? " · " + s.outcome.notes : ""}</span>`;
    ul.appendChild(li);
  }
}

// ---- boot --------------------------------------------------------------------

refreshJobs();
loadMotions();
loadPreviews();
refreshCloud();
refreshBodyModels();
setMode(localStorage.getItem("g1.mode") || "studio");
setInterval(() => {
  if (!$("#studio-main").hidden) { refreshJobs(); if (selectedJob) showJob(selectedJob); }
}, 2500);
setInterval(refreshCloud, 30000);
setInterval(() => { if (!$("#system-main").hidden) refreshSystem(); }, 20000);

// G1 Dance Studio frontend — talks to ui/server.py. No framework, no build step.
"use strict";

const $ = (sel) => document.querySelector(sel);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

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
  try {
    const j = await api("/api/jobs/upload", { method: "POST", body: fd });
    selectedJob = j.id;
    $("#video-file").value = "";
    await refreshJobs();
    await showJob(j.id);
  } catch (err) { alert("Could not create job: " + err.message); }
};

// ---- vetting ---------------------------------------------------------------

async function loadMotions() {
  const motions = await api("/api/motions");
  const opts = motions.map((m) => `<option value="${m.path}">${m.name}</option>`).join("");
  $("#vet-select").innerHTML = opts;
  $("#csv-select").innerHTML = opts;
}

$("#csv-run").onclick = async () => {
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
};

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
  $("#deploy-phrase").value = "";
  $("#deploy-dialog").showModal();
};
$("#deploy-cancel").onclick = () => $("#deploy-dialog").close();
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
  if (cfg.transport) {
    document.querySelector(`input[name=transport][value=${cfg.transport}]`).checked = true;
    cloudFieldsFor(cfg.transport);
  }
  $("#ssh-host").value = cfg.ssh.host || "";
  $("#ssh-port").value = cfg.ssh.port || "";
  $("#ssh-user").value = cfg.ssh.user || "";
  $("#ssh-key").value = cfg.ssh.key_path || "";
  $("#jup-url").value = cfg.jupyter.url || "";
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

// ---- boot --------------------------------------------------------------------

refreshJobs();
loadMotions();
loadPreviews();
refreshCloud();
refreshBodyModels();
setInterval(() => { refreshJobs(); if (selectedJob) showJob(selectedJob); }, 2500);
setInterval(refreshCloud, 30000);

"use strict";

const state = { tasks: [], task: null, sample: null, zoom: false, point: null, epochTimer: null };
const $ = (id) => document.getElementById(id);
const fileUrl = (p) => "/file?p=" + encodeURIComponent(p);
const thumbUrl = (p) => "/thumb?p=" + encodeURIComponent(p);

const GROUPS = [
  ["Restoration", (t) => t.kind === "restore"],
  ["Super-resolution x6", (t) => t.kind === "sr"],
  ["Text to image", (t) => t.kind === "prompt"],
  ["Image-to-image translation", (t) => t.kind === "epochs"],
];

function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const c of kids) if (c != null) n.append(c);
  return n;
}

/* ---------- theme ---------- */
function applyTheme(t) {
  if (t) document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
}
function currentTheme() {
  const set = document.documentElement.getAttribute("data-theme");
  if (set) return set;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
try { applyTheme(localStorage.getItem("gvl-theme")); } catch (e) { /* storage blocked */ }
$("theme").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("gvl-theme", next); } catch (e) { /* storage blocked */ }
});

/* ---------- task list ---------- */
function renderTaskList() {
  const list = $("taskList");
  const sel = $("taskSelect");
  list.replaceChildren();
  sel.replaceChildren();
  for (const [name, test] of GROUPS) {
    const items = state.tasks.filter(test);
    if (!items.length) continue;
    const g = el("div", { class: "group" }, el("h4", { text: name }));
    const og = el("optgroup", { label: name });
    for (const t of items) {
      g.append(el("button", { class: "task", type: "button", "data-id": t.id, text: t.title,
        onclick: () => selectTask(t.id) }));
      og.append(el("option", { value: t.id, text: t.title }));
    }
    list.append(g);
    sel.append(og);
  }
  sel.addEventListener("change", () => selectTask(sel.value));
}

function selectTask(id) {
  const t = state.tasks.find((x) => x.id === id);
  if (!t) return;
  state.task = t;
  state.point = null;
  stopEpochs();
  for (const b of document.querySelectorAll(".task")) b.classList.toggle("active", b.dataset.id === id);
  $("taskSelect").value = id;
  $("taskTitle").textContent = t.title;
  $("taskModel").textContent = t.model || "see the project folder";
  $("taskData").textContent = t.data || "see the project folder";
  renderScores(t);
  if (history.replaceState) history.replaceState(null, "", "#" + id);
  if (t.kind === "epochs") return renderEpochs(t);
  renderPicker(t);
  selectSample(t.samples[0] && t.samples[0].id);
}

function renderScores(t) {
  const wrap = $("scoreWrap");
  wrap.replaceChildren();
  if (!t.scores.length) {
    wrap.append(el("p", { class: "note", text: "This run has no single scalar metric; judge it on the images below." }));
    return;
  }
  const body = el("tbody");
  for (const r of t.scores) {
    body.append(el("tr", {}, el("td", { text: r.split }), el("td", { text: r.metric }),
      el("td", { text: String(r.value) })));
  }
  wrap.append(el("table", {}, el("thead", {}, el("tr", {},
    el("th", { text: "Split" }), el("th", { text: "Metric" }), el("th", { text: "Value (results.json)" }))), body));
}

/* ---------- samples ---------- */
function thumbSource(t, s) { return s.input || s.after; }

function renderPicker(t) {
  const p = $("picker");
  p.replaceChildren();
  p.hidden = false;
  t.samples.forEach((s, i) => {
    const label = t.kind === "prompt" ? "Prompt " + (i + 1) : "Sample " + (i + 1);
    p.append(el("button", { class: "thumb", type: "button", "data-id": s.id, role: "listitem",
      "aria-label": label, title: s.prompt || s.id, onclick: () => selectSample(s.id) },
      el("img", { src: thumbUrl(thumbSource(t, s)), alt: "" }),
      el("span", { text: label })));
  });
}

function columnsFor(t, s) {
  const cols = [];
  if (t.kind === "prompt") {
    cols.push({ key: "before", name: t.before, src: s.before });
    cols.push({ key: "after", name: "Final model", src: s.after });
    return cols;
  }
  cols.push({ key: "input", name: t.kind === "sr" ? "Low-res input (enlarged)" : "Input", src: s.input });
  if (s.bicubic) cols.push({ key: "bicubic", name: "Bicubic", src: s.bicubic });
  cols.push({ key: "before", name: t.before, src: s.before });
  cols.push({ key: "after", name: "Fine-tuned model", src: s.after });
  cols.push({ key: "truth", name: "Ground truth", src: s.truth });
  return cols;
}

async function selectSample(id) {
  const t = state.task;
  const s = t.samples.find((x) => x.id === id);
  if (!s) return;
  state.sample = s;
  for (const b of document.querySelectorAll(".thumb")) b.classList.toggle("active", b.dataset.id === id);
  renderControls(t);
  const idx = t.samples.indexOf(s) + 1;
  $("viewerTitle").textContent = (t.kind === "prompt" ? "Prompt " : "Sample ") + idx + " of " + t.samples.length;
  const stage = $("stage");
  stage.replaceChildren();
  if (t.kind === "prompt") stage.append(el("p", { class: "prompt" }, "Prompt: ", el("q", { text: s.prompt })));
  const cols = columnsFor(t, s);
  const grid = el("div", { class: "cols" });
  for (const c of cols) {
    const frame = el("div", { class: "frame" + (state.zoom ? " zoomable" : "") });
    const img = el("img", { src: fileUrl(c.src), alt: c.name + ", " + (s.prompt || s.id) });
    frame.append(img);
    frame.addEventListener("click", (ev) => onFrameClick(ev, img));
    const score = el("div", { class: "score", "data-key": c.key, text: c.key === "truth" ? "reference" : "" });
    grid.append(el("div", { class: "col" }, el("figure", {}, frame,
      el("figcaption", {}, el("div", { class: "name", text: c.name }), score))));
    c.img = img;
    c.frame = frame;
  }
  stage.append(grid);
  state.cols = cols;
  $("note").className = "note" + (t.before_note && t.before !== "Before fine-tuning" && !t.before.startsWith("Published") ? " warn" : "");
  $("note").textContent = (t.before ? t.before + ": " + t.before_note : "");
  if (state.zoom && state.point) drawZoom();
  else if (state.zoom) {
    $("note").className = "note";
    $("note").textContent = "Zoom is on: click the spot you want to inspect on any image.";
  }
  if (s.truth) loadScores(t, s);
}

async function loadScores(t, s) {
  try {
    const r = await fetch("/api/scores?task=" + encodeURIComponent(t.id) + "&sample=" + encodeURIComponent(s.id));
    if (!r.ok) throw new Error("HTTP " + r.status);
    const scores = await r.json();
    if (state.sample !== s) return;
    const vals = Object.entries(scores).filter(([k]) => k !== "input");
    const best = vals.length ? Math.max(...vals.map(([, v]) => v)) : null;
    for (const node of document.querySelectorAll(".score[data-key]")) {
      const k = node.dataset.key;
      if (k in scores) {
        node.textContent = "PSNR " + scores[k].toFixed(2) + " dB";
        node.classList.toggle("best", k !== "input" && scores[k] === best);
      }
    }
  } catch (e) {
    for (const node of document.querySelectorAll(".score[data-key]")) {
      if (node.dataset.key !== "truth") { node.textContent = "score unavailable"; node.classList.add("err"); }
    }
  }
}

/* ---------- zoom ---------- */
function renderControls(t) {
  const c = $("controls");
  c.replaceChildren();
  if (t.kind === "epochs") return;
  const n = t.samples.length;
  const i = t.samples.indexOf(state.sample);
  c.append(el("button", { type: "button", text: "Previous", "aria-label": "Previous sample",
    onclick: () => selectSample(t.samples[(i - 1 + n) % n].id) }));
  c.append(el("button", { type: "button", text: "Next", "aria-label": "Next sample",
    onclick: () => selectSample(t.samples[(i + 1) % n].id) }));
  c.append(el("button", { type: "button", class: state.zoom ? "on" : "", "aria-pressed": String(state.zoom),
    text: state.zoom ? "Zoom on" : "Zoom", onclick: toggleZoom }));
}

function toggleZoom() {
  state.zoom = !state.zoom;
  state.point = null;
  selectSample(state.sample.id);
}

function onFrameClick(ev, img) {
  if (!state.zoom) return;
  const frame = ev.currentTarget;
  const canvas = frame.querySelector("canvas");
  const rect = frame.getBoundingClientRect();
  const fx = (ev.clientX - rect.left) / rect.width;
  const fy = (ev.clientY - rect.top) / rect.height;
  if (!state.point || !canvas) {
    // the image is letterboxed inside a square frame (object-fit: contain)
    const r = img.naturalWidth / img.naturalHeight || 1;
    const w = r >= 1 ? 1 : r, h = r >= 1 ? 1 / r : 1;
    const ix = (fx - (1 - w) / 2) / w, iy = (fy - (1 - h) / 2) / h;
    if (ix < 0 || ix > 1 || iy < 0 || iy > 1) return;
    state.point = { fx: ix, fy: iy };
  } else {
    // clicking inside a zoomed crop moves the centre within the crop
    const p = state.point;
    const span = state.span;
    state.point = { fx: p.fx + (fx - 0.5) * span.w, fy: p.fy + (fy - 0.5) * span.h };
  }
  drawZoom();
}

function drawZoom() {
  const cols = state.cols || [];
  const ref = cols.find((c) => c.key === "truth") || cols[cols.length - 1];
  if (!ref) return;
  const go = () => {
    const W = ref.img.naturalWidth, H = ref.img.naturalHeight;
    if (!W || !H) return;
    const size = Math.max(48, Math.round(Math.min(W, H) / (W > 1000 ? 8 : 4)));
    state.span = { w: size / W, h: size / H };
    const p = state.point;
    p.fx = Math.min(1 - state.span.w / 2, Math.max(state.span.w / 2, p.fx));
    p.fy = Math.min(1 - state.span.h / 2, Math.max(state.span.h / 2, p.fy));
    for (const c of cols) {
      const draw = () => {
        const w = c.img.naturalWidth, h = c.img.naturalHeight;
        const cw = Math.round(size * w / W), ch = Math.round(size * h / H);
        const canvas = c.frame.querySelector("canvas") || el("canvas", { width: 512, height: 512, "aria-label": c.name + " zoomed" });
        const ctx = canvas.getContext("2d");
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, 512, 512);
        ctx.drawImage(c.img, Math.round(p.fx * w - cw / 2), Math.round(p.fy * h - ch / 2), cw, ch, 0, 0, 512, 512);
        if (!canvas.parentNode) { c.img.style.display = "none"; c.frame.append(canvas); }
      };
      if (c.img.complete && c.img.naturalWidth) draw(); else c.img.addEventListener("load", draw, { once: true });
    }
    $("note").textContent = "Zoom: a " + size + " x " + size + " pixel region of the " + W + " x " + H +
      (ref.key === "truth" ? " ground truth" : " image") + ", same place in every column. Click a crop to move it; turn Zoom off for the whole image.";
    $("note").className = "note";
  };
  if (ref.img.complete && ref.img.naturalWidth) go(); else ref.img.addEventListener("load", go, { once: true });
}

/* ---------- epochs ---------- */
function stopEpochs() {
  if (state.epochTimer) clearInterval(state.epochTimer);
  state.epochTimer = null;
}

function renderEpochs(t) {
  $("picker").replaceChildren();
  $("picker").hidden = true;
  $("controls").replaceChildren();
  $("note").className = "note";
  $("note").textContent = "";
  const stage = $("stage");
  stage.replaceChildren(el("p", { class: "note", text: "Each frame is a grid the run saved during training: " +
    "the same held-out inputs, the generator's output at that epoch and, for the paired pix2pix runs, the ground truth. " +
    "Drag the slider or press Play to watch the outputs change over training." }));
  const n = t.samples.length;
  if (!n) { stage.append(el("p", { class: "err", text: "No saved grids found for this run." })); return; }
  $("viewerTitle").textContent = "Training progress, " + n + " saved epochs";
  const img = el("img", { class: "grid-img", alt: "" });
  const label = el("strong", {});
  const range = el("input", { type: "range", min: "0", max: String(n - 1), value: String(n - 1),
    "aria-label": "Epoch" });
  const play = el("button", { type: "button", text: "Play" });
  const show = (i) => {
    const s = t.samples[i];
    img.src = fileUrl(s.grid);
    img.alt = t.title + ", outputs at epoch " + s.epoch;
    label.textContent = "Epoch " + s.epoch;
    range.value = String(i);
  };
  range.addEventListener("input", () => { stopEpochs(); play.textContent = "Play"; show(Number(range.value)); });
  play.addEventListener("click", () => {
    if (state.epochTimer) { stopEpochs(); play.textContent = "Play"; return; }
    play.textContent = "Pause";
    let i = Number(range.value) >= n - 1 ? 0 : Number(range.value);
    show(i);
    state.epochTimer = setInterval(() => {
      i += 1;
      if (i >= n) { stopEpochs(); play.textContent = "Play"; return; }
      show(i);
    }, 700);
  });
  stage.append(el("div", { class: "epoch-row" }, play, range, label), img);
  show(n - 1);
}

/* ---------- boot ---------- */
async function boot() {
  try {
    const r = await fetch("/api/catalogue");
    if (!r.ok) throw new Error("HTTP " + r.status);
    state.tasks = (await r.json()).tasks;
  } catch (e) {
    $("taskTitle").textContent = "Could not load the catalogue";
    $("stage").append(el("p", { class: "err", text: String(e) }));
    return;
  }
  renderTaskList();
  const want = decodeURIComponent(location.hash.slice(1));
  selectTask(state.tasks.some((t) => t.id === want) ? want : state.tasks[0].id);
  window.addEventListener("hashchange", () => {
    const id = decodeURIComponent(location.hash.slice(1));
    if (state.task && id !== state.task.id && state.tasks.some((t) => t.id === id)) selectTask(id);
  });
}
boot();

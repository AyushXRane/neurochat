/* neurochat UI.
 *
 * Two rules shape this file:
 *
 *   R5 — anything computable without the LLM must be reachable without it. Clicking
 *   an atlas region, clicking an action-log row, dragging a display slider: all of
 *   these POST straight to /api/tool. The LLM-call counter in the top bar does not
 *   move, and a test asserts on it.
 *
 *   Latency — the viewer is driven by WebSocket commands the backend pushes as each
 *   tool call completes, not by the chat response. The crosshair moves while the
 *   prose is still streaming.
 */

const nv = new niivue.Niivue({
  backColor: [0, 0, 0, 1],
  show3Dcrosshair: true,
  crosshairColor: [0.35, 0.65, 0.96, 1],
  textHeight: 0.04,
  isColorbar: false,
});

const state = {
  layers: [],
  selected: null,
  atlasId: null,
  atlasSpace: null,
  regions: [],
  library: [],
  activeScan: null,
  lastRegion: null,
  loadedUrls: new Map(), // layer name -> url currently in Niivue
};

const $ = (id) => document.getElementById(id);
const fileUrl = (path) => `/api/file?path=${encodeURIComponent(path)}`;

// ── tool dispatch ─────────────────────────────────────────────────────────

/** Call a tool directly. `viaLLM` is false here by definition — that's the point. */
async function callTool(tool, args = {}, { log = true } = {}) {
  const response = await fetch("/api/tool", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool, args }),
  });
  const payload = await response.json();
  if (log) logAction(tool, payload, { noLLM: true });
  await refreshScript();
  refreshState();
  return payload;
}

// ── action log ────────────────────────────────────────────────────────────

function logAction(tool, payload, { noLLM = false } = {}) {
  const item = document.createElement("li");
  const ok = payload && payload.ok !== false;
  if (!ok) item.classList.add("failed");

  const marker = noLLM ? ' <span class="no-llm" title="No model call">·no-llm</span>' : "";
  const trace = payload && payload.tool_trace;
  const args = trace && trace.args && trace.args.length ? `(${trace.args.join(", ")})` : "()";
  item.innerHTML = `<span class="tool-name">${tool}</span>${args}${marker}`;

  // neuroglancer-chat's rule: a row that carries coordinates re-navigates on click,
  // with no new model call.
  const coords = payload && payload.coords;
  if (ok && Array.isArray(coords) && coords.length === 3) {
    item.classList.add("clickable");
    item.title = "Click to return to this location (no model call)";
    item.addEventListener("click", () =>
      callTool("navigate", { coords, space: payload.space })
    );
    item.innerHTML += ` → [${coords.map((c) => c.toFixed(1)).join(", ")}] ${payload.space}`;
  } else if (!ok) {
    item.title = payload.message || "failed";
    item.innerHTML += ` — ${payload.error || "failed"}`;
  }

  const log = $("action-log");
  log.appendChild(item);
  log.scrollTop = log.scrollHeight;
}

// ── script pane ───────────────────────────────────────────────────────────

async function refreshScript() {
  const text = await (await fetch("/api/script")).text();
  const pane = $("script-pane");
  pane.innerHTML = text
    .split("\n")
    .map((line) => {
      const escaped = line.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
      return line.trimStart().startsWith("#") ? `<span class="comment">${escaped}</span>` : escaped;
    })
    .join("\n");
  pane.scrollTop = pane.scrollHeight;
}

// ── session state → control panel ─────────────────────────────────────────

async function refreshState() {
  const data = await (await fetch("/api/state")).json();
  $("llm-counter").textContent = `LLM calls: ${data.llm_call_count}`;
  state.layers = data.viewer.layers;

  const list = $("layer-list");
  list.innerHTML = "";
  const volumes = data.volumes || {};
  for (const layer of state.layers) {
    const volume = volumes[layer.name] || {};
    const space = (volume.space && volume.space.space) || "?";
    const item = document.createElement("li");
    if (state.selected === layer.name) item.classList.add("selected");
    item.innerHTML =
      `<input type="checkbox" ${layer.visible ? "checked" : ""} title="Visible" />` +
      `<span class="layer-name">${layer.name}</span>` +
      `<span class="layer-space">${space}</span>`;
    item.querySelector("input").addEventListener("change", (event) => {
      event.stopPropagation();
      callTool("set_display", { volume: layer.name, visible: event.target.checked });
    });
    item.addEventListener("click", () => {
      state.selected = layer.name;
      refreshState();
    });
    list.appendChild(item);
  }
  if (!state.layers.length) list.innerHTML = '<p class="hint">No volume loaded.</p>';
  if (state.selected && !state.layers.some((l) => l.name === state.selected)) state.selected = null;

  renderSpacePrompts(volumes);
  renderDisplayControls(volumes);
  await syncViewer();

  if (data.atlas && data.atlas.atlas_id !== state.atlasId) {
    state.atlasId = data.atlas.atlas_id;
    state.atlasSpace = data.atlas.space;
    $("atlas-select").value = data.atlas.atlas_id;
    await refreshRegions();
  }
  if (!data.chat_available) disableChat();
}

/* A volume whose space we could not establish is not an error — it is a question we
 * cannot answer for the user. Ask it here, once, with the answer pre-filled from
 * geometry. Accepting reloads the volume with `space=` set, so the provenance still
 * records the space as the user's assertion and not as something we inferred. */
function renderSpacePrompts(volumes) {
  const host = $("space-prompts");
  host.innerHTML = "";
  for (const volume of Object.values(volumes)) {
    const space = volume.space || {};
    if (space.resolvable || !space.suggested_space) continue;

    const box = document.createElement("div");
    box.className = "space-prompt";
    box.innerHTML =
      `<strong>${volume.name}</strong> has no template space in its header, so region ` +
      `names are turned off for it.<br />` +
      `<span class="hint">${space.grid_hint ? "Geometry looks like " + space.grid_hint + "." : ""} ` +
      `Only you can confirm it was normalised.</span>`;

    const accept = document.createElement("button");
    accept.textContent = `Treat as ${space.suggested_space}`;
    accept.title = "Recorded as your assertion, not an inference";
    accept.addEventListener("click", async () => {
      accept.disabled = true;
      const result = await callTool("load_volume", {
        path: volume.path,
        name: volume.name,
        space: space.suggested_space,
      });
      if (!result.ok) addMessage("error", result.message);
    });
    box.appendChild(accept);
    host.appendChild(box);
  }
}

function renderDisplayControls(volumes) {
  const host = $("display-controls");
  const layer = state.layers.find((l) => l.name === state.selected);
  if (!layer) {
    host.innerHTML = '<p class="hint">Select a layer above.</p>';
    return;
  }
  const summary = (volumes[layer.name] || {}).values || {};
  const lo = summary.min ?? 0;
  const hi = summary.max ?? 1;
  const step = Math.max((hi - lo) / 200, 1e-6);
  const maps = ["gray", "hot", "cool", "viridis", "inferno", "magma", "plasma", "jet", "bone", "winter"];

  host.innerHTML = `
    <label>colormap
      <select id="ctl-cmap">${maps
        .map((m) => `<option ${m === layer.colormap ? "selected" : ""}>${m}</option>`)
        .join("")}</select>
    </label>
    <label>min
      <input id="ctl-min" type="range" min="${lo}" max="${hi}" step="${step}" value="${layer.cal_min ?? lo}" />
      <span class="value" id="ctl-min-v">${(layer.cal_min ?? lo).toFixed(2)}</span>
    </label>
    <label>max
      <input id="ctl-max" type="range" min="${lo}" max="${hi}" step="${step}" value="${layer.cal_max ?? hi}" />
      <span class="value" id="ctl-max-v">${(layer.cal_max ?? hi).toFixed(2)}</span>
    </label>
    <label>opacity
      <input id="ctl-op" type="range" min="0" max="1" step="0.05" value="${layer.opacity}" />
      <span class="value" id="ctl-op-v">${layer.opacity.toFixed(2)}</span>
    </label>`;

  const live = (id, valueId, format) => {
    const input = $(id);
    input.addEventListener("input", () => ($(valueId).textContent = format(input.value)));
    return input;
  };
  const fixed = (v) => Number(v).toFixed(2);
  const minInput = live("ctl-min", "ctl-min-v", fixed);
  const maxInput = live("ctl-max", "ctl-max-v", fixed);
  const opInput = live("ctl-op", "ctl-op-v", fixed);

  // Commit on release, not on every pixel of drag — one tool call, one script line.
  const commit = () =>
    callTool("set_display", {
      volume: layer.name,
      colormap: $("ctl-cmap").value,
      min: Number(minInput.value),
      max: Number(maxInput.value),
      opacity: Number(opInput.value),
    });
  [minInput, maxInput, opInput].forEach((el) => el.addEventListener("change", commit));
  $("ctl-cmap").addEventListener("change", commit);
}

// ── atlas panel ───────────────────────────────────────────────────────────

async function refreshRegions(query = "") {
  const url = `/api/regions${query ? `?query=${encodeURIComponent(query)}` : ""}`;
  const data = await (await fetch(url)).json();
  state.regions = data.regions || [];
  const list = $("region-list");
  list.innerHTML = "";
  for (const region of state.regions) {
    const item = document.createElement("li");
    item.innerHTML =
      `<span>${region.label}</span>` +
      `<span class="coords">${region.centroid.map((c) => c.toFixed(0)).join(" ")}</span>`;
    item.title = `${region.n_voxels} voxels · ${region.volume_mm3} mm³ · click to navigate (no model call)`;
    // R5: navigate with zero LLM calls. Also arms the cohort table with this region.
    item.addEventListener("click", () => {
      state.lastRegion = region.label;
      updateRegionTableButton();
      callTool("navigate", { region_label: region.label });
    });
    list.appendChild(item);
  }
  if (!state.regions.length) {
    list.innerHTML = `<li class="hint">${state.atlasId ? "No match." : "Load an atlas first."}</li>`;
  }
}

// ── library: many scans on disk, one at a time on screen ──────────────────

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    return { ok: false, message: detail.detail || response.statusText };
  }
  return response.json();
}

function renderLibrary() {
  const list = $("lib-list");
  list.innerHTML = "";
  for (const entry of state.library) {
    const item = document.createElement("li");
    if (entry.path === state.activeScan) item.classList.add("active");
    if (!entry.space_resolvable) item.classList.add("blocked");
    const size = `${entry.shape.join("×")} @ ${entry.voxel_size_mm[0]}mm`;
    item.innerHTML =
      `<span class="lib-name">${entry.name}</span>` +
      `<span class="lib-meta">${size} · ${entry.space}</span>`;
    item.title = `${entry.path}\nspace: ${entry.space} (${entry.space_source})` +
      (entry.space_resolvable ? "" : "\nRegion names are off until the space is asserted.");
    // Clicking a scan loads it and hides the others. No model call.
    item.addEventListener("click", () => selectScan(entry));
    list.appendChild(item);
  }
  $("lib-actions").hidden = state.library.length === 0;
  updateRegionTableButton();
}

async function selectScan(entry) {
  $("lib-status").textContent = `Loading ${entry.name}…`;
  const result = await post("/api/library/select", {
    path: entry.path,
    // If the header can't establish a space, carry the same assertion the volume
    // prompt would ask for — the user chose this scan knowing what the list said.
    space: entry.space_resolvable ? undefined : entry.suggested_space || undefined,
    solo: true,
  });
  if (!result.ok) {
    $("lib-status").textContent = result.message || "Could not load that scan.";
    return;
  }
  state.activeScan = entry.path;
  $("lib-status").textContent = `Showing ${entry.name}`;
  logAction("library.select", result, { noLLM: true });
  await refreshScript();
  await refreshState();
  renderLibrary();
}

function updateRegionTableButton() {
  const button = $("lib-table");
  const region = state.lastRegion;
  button.disabled = !region || state.library.length === 0;
  button.textContent = region
    ? `${region} across ${state.library.length} scans`
    : "Region across all scans";
  button.title = region
    ? "Measures this region in every scan in the library. No model call."
    : "Click a region in the atlas list first.";
}

async function runRegionTable() {
  const button = $("lib-table");
  button.disabled = true;
  button.textContent = "Measuring…";
  // If every unresolvable scan agrees on the same likely space, carry that assertion
  // through — it is the same one-click decision the volume prompt asks for, made once
  // for the cohort instead of once per scan.
  const blocked = state.library.filter((e) => !e.space_resolvable);
  const suggested = new Set(blocked.map((e) => e.suggested_space).filter(Boolean));
  const assume =
    blocked.length && suggested.size === 1 ? [...suggested][0] : undefined;

  const result = await post("/api/library/region_table", {
    region_label: state.lastRegion,
    assume_space: assume,
  });
  updateRegionTableButton();
  if (!result.ok) {
    switchBottomView("results");
    $("results-view").innerHTML = `<p class="hint">${result.message || "Failed."}</p>`;
    return;
  }
  logAction("library.region_table", result, { noLLM: true });
  renderResultTable(result);
  switchBottomView("results");
  await refreshScript();
}

function renderResultTable(result) {
  const host = $("results-view");
  const columns = result.columns;
  const head = columns.map((c) => `<th>${c}</th>`).join("");
  const body = result.rows
    .map(
      (row) =>
        `<tr data-path="${row.path}">` +
        columns.map((c) => `<td>${row[c] === null ? "—" : row[c]}</td>`).join("") +
        `</tr>`
    )
    .join("");

  host.innerHTML =
    `<p class="hint"><strong>${result.region}</strong> · ${result.atlas_id} · ` +
    `${result.n_measured} measured${result.n_skipped ? `, ${result.n_skipped} skipped` : ""}</p>` +
    `<table class="result-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>` +
    (result.skipped.length
      ? `<p class="hint skipped">Skipped: ` +
        result.skipped.map((s) => `${s.path.split("/").pop()} — ${s.reason}`).join("; ") +
        `</p>`
      : "") +
    result.notes.map((n) => `<p class="hint">${n}</p>`).join("");

  // Clicking a row jumps to that scan. Deterministic, no model call.
  host.querySelectorAll("tbody tr").forEach((row) => {
    row.title = "Click to view this scan (no model call)";
    row.addEventListener("click", () => {
      const entry = state.library.find((e) => e.path === row.dataset.path);
      if (entry) selectScan(entry);
    });
  });

  const csv = $("results-csv");
  csv.hidden = false;
  csv.onclick = () => window.open(`/api/library/table.csv?table_id=${result.table_id}`, "_blank");
}

function switchBottomView(view) {
  $("action-log").hidden = view !== "log";
  $("results-view").hidden = view !== "results";
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === view);
  });
}

// ── Niivue ────────────────────────────────────────────────────────────────

async function syncViewer() {
  const wanted = state.layers.filter((l) => l.visible);
  const signature = wanted.map((l) => l.name).join("|");
  if (signature !== state.signature) {
    state.signature = signature;
    if (!wanted.length) {
      nv.volumes = [];
      nv.updateGLVolume();
      return;
    }
    await nv.loadVolumes(
      wanted.map((layer) => ({
        url: fileUrl(layer.path),
        name: `${layer.name}.nii.gz`,
        colormap: layer.colormap,
        opacity: layer.opacity,
        cal_min: layer.cal_min ?? undefined,
        cal_max: layer.cal_max ?? undefined,
      }))
    );
  } else {
    wanted.forEach((layer, index) => {
      const volume = nv.volumes[index];
      if (!volume) return;
      if (volume.colormap !== layer.colormap) nv.setColormap(volume.id, layer.colormap);
      volume.opacity = layer.opacity;
      if (layer.cal_min !== null) volume.cal_min = layer.cal_min;
      if (layer.cal_max !== null) volume.cal_max = layer.cal_max;
    });
    nv.updateGLVolume();
  }
}

function moveCrosshair(mm, label, space) {
  if (nv.volumes.length) {
    nv.scene.crosshairPos = nv.mm2frac(mm);
    nv.drawScene();
  }
  const text = mm.map((c) => c.toFixed(1)).join(", ");
  $("crosshair-readout").textContent = `${label ? label + "  " : ""}[${text}] ${space}`;
  $("space-badge").textContent = `space: ${space}`;
}

nv.onLocationChange = (data) => {
  if (!data || !data.mm) return;
  const mm = Array.from(data.mm).slice(0, 3);
  const space = $("space-badge").textContent.replace("space: ", "");
  $("crosshair-readout").textContent = `[${mm.map((c) => c.toFixed(1)).join(", ")}] ${space}`;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "crosshair", coords: mm }));
  }
};

// ── WebSocket: viewer commands out, canvas snapshots back ─────────────────

let socket = null;

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws`);

  socket.onmessage = async (event) => {
    const command = JSON.parse(event.data);
    if (command.type === "navigate") {
      moveCrosshair(command.coords, command.label, command.space);
    } else if (command.type === "state") {
      // Sent on connect so a reloaded page picks up where the session already is.
      await refreshState();
      const view = command.state || {};
      if (view.crosshair_space && view.crosshair_space !== "unknown") {
        moveCrosshair(view.crosshair_mm, view.crosshair_label, view.crosshair_space);
      } else if (state.layers.length) {
        $("crosshair-readout").textContent = "click the canvas, or pick a region on the left";
      }
    } else if (command.type === "layers") {
      await refreshState();
    } else if (command.type === "snapshot_request") {
      // The backend cannot see WebGL output, so it asks the canvas for pixels.
      // Draw first: reading before a draw can return an empty buffer.
      nv.drawScene();
      requestAnimationFrame(() => {
        let dataUrl = "";
        try {
          dataUrl = nv.canvas.toDataURL("image/png");
        } catch (err) {
          dataUrl = "";
        }
        socket.send(
          JSON.stringify({ type: "snapshot", request_id: command.request_id, data_url: dataUrl })
        );
      });
    }
  };

  socket.onclose = () => setTimeout(connect, 1500);
}

// ── chat ──────────────────────────────────────────────────────────────────

function addMessage(kind, text) {
  const element = document.createElement("div");
  element.className = `msg ${kind}`;
  element.textContent = text;
  const log = $("chat-log");
  log.appendChild(element);
  log.scrollTop = log.scrollHeight;
  return element;
}

function disableChat() {
  $("chat-input").disabled = true;
  $("chat-send").disabled = true;
  $("chat-input").placeholder = "Chat needs ANTHROPIC_API_KEY. Every control still works without it.";
}

async function sendChat(text) {
  addMessage("user", text);
  let bubble = null;

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop();
    for (const chunk of chunks) {
      if (!chunk.startsWith("data: ")) continue;
      const event = JSON.parse(chunk.slice(6));

      if (event.type === "text") {
        if (!bubble) bubble = addMessage("assistant", "");
        bubble.textContent += event.text;
        $("chat-log").scrollTop = $("chat-log").scrollHeight;
      } else if (event.type === "tool_use") {
        bubble = null;
        addMessage("tool", `▸ ${event.tool}(${Object.keys(event.args).join(", ")})`);
      } else if (event.type === "tool_result") {
        logAction(event.tool, event.result, { noLLM: false });
        if (!event.ok) {
          const message = addMessage("tool failed", event.result.message || "failed");
          renderSuggestions(message, event.result.suggestions);
        }
      } else if (event.type === "script") {
        refreshScript();
      } else if (event.type === "refusal") {
        bubble = null;
        const message = addMessage(
          "refusal",
          `${event.message}\n\nUse instead:\n${(event.use_instead || []).map((u) => `  · ${u}`).join("\n")}`
        );
        message.title = "Recorded in the script as a comment; nothing was executed.";
      } else if (event.type === "error") {
        bubble = null;
        addMessage("error", event.message);
      } else if (event.type === "state") {
        refreshState();
      }
    }
  }
  refreshState();
}

/** Did-you-mean buttons. Re-running the tool with a real label costs no model call. */
function renderSuggestions(element, suggestions) {
  if (!suggestions || !suggestions.length) return;
  const box = document.createElement("div");
  box.className = "suggestions";
  box.textContent = "Did you mean:";
  for (const label of suggestions) {
    const button = document.createElement("button");
    button.textContent = label;
    button.title = "Navigate there now (no model call)";
    button.addEventListener("click", () => callTool("navigate", { region_label: label }));
    box.appendChild(button);
  }
  element.appendChild(box);
}

// ── wiring ────────────────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", async () => {
  nv.attachToCanvas($("nv-canvas"));

  const data = await (await fetch("/api/state")).json();
  const select = $("atlas-select");
  for (const [id, spec] of Object.entries(data.atlases || {})) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = `${id}${spec.bundled ? "" : " (downloads once)"}`;
    option.title = spec.description;
    select.appendChild(option);
  }

  select.addEventListener("change", async () => {
    if (!select.value) return;
    $("atlas-note").textContent = "Loading…";
    const result = await callTool("load_atlas", { atlas_name: select.value });
    $("atlas-note").textContent = result.ok
      ? `${result.n_regions} regions · ${result.space}`
      : result.message;
    if (result.ok) {
      state.atlasId = result.atlas_id;
      state.atlasSpace = result.space;
      await refreshRegions($("region-filter").value);
    }
  });

  let filterTimer = null;
  $("region-filter").addEventListener("input", (event) => {
    clearTimeout(filterTimer);
    filterTimer = setTimeout(() => refreshRegions(event.target.value), 150);
  });

  // -- library controls --
  const applyLibrary = (result) => {
    if (result.ok === false) {
      $("lib-status").textContent = result.message || "Scan failed.";
      return;
    }
    state.library = result.entries || [];
    state.activeScan = null;
    $("lib-dir").value = result.root || $("lib-dir").value;
    $("lib-status").textContent =
      `${result.n_found} scan(s). ` + (result.notes || []).join(" ");
    renderLibrary();
  };

  $("lib-scan").addEventListener("click", async () => {
    $("lib-status").textContent = "Scanning…";
    applyLibrary(await post("/api/library/scan", { directory: $("lib-dir").value.trim() }));
  });
  $("lib-dir").addEventListener("keydown", (event) => {
    if (event.key === "Enter") $("lib-scan").click();
  });

  $("lib-sample").addEventListener("click", async () => {
    $("lib-status").textContent = "Fetching OASIS MRI cohort (first run downloads, then cached)…";
    applyLibrary(await post("/api/library/sample_cohort", { n_subjects: 12 }));
  });

  $("lib-pet").addEventListener("click", async () => {
    $("lib-status").textContent = "Fetching PET cohort from OpenNeuro (~4MB per subject)…";
    applyLibrary(await post("/api/library/pet_cohort", { n_subjects: 8 }));
  });

  $("lib-table").addEventListener("click", runRegionTable);
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchBottomView(tab.dataset.view));
  });

  $("volume-load").addEventListener("click", async () => {
    const path = $("volume-path").value.trim();
    if (!path) return;
    const result = await callTool("load_volume", { path });
    if (!result.ok) addMessage("error", result.message);
    else $("volume-path").value = "";
  });

  $("export-btn").addEventListener("click", async () => {
    const result = await callTool("export_script", { path: "neurochat_session.py" });
    addMessage(result.ok ? "tool" : "error", result.ok ? `Wrote ${result.path}` : result.message);
  });

  $("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("chat-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    sendChat(text).catch((err) => addMessage("error", String(err)));
  });

  $("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("chat-form").requestSubmit();
    }
  });

  connect();
  await refreshState();
  await refreshScript();
});

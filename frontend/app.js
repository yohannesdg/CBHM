/* CBHM, distributed water balance for African basins: front end. Talks to the FastAPI backend under /api. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const FIELD_LABEL = {
  precip: "Rainfall", pet: "PET", et: "Actual ET", runoff: "Runoff", recharge: "Recharge",
  baseflow: "Baseflow", soil: "Soil moisture", gw: "Groundwater storage", discharge: "Discharge",
  "static:dem": "Elevation", "static:acc": "Upstream area", "static:zone": "Parameter zones", "static:lake": "Open water",
};

const S = {
  meta: null, res: null, params: null, zone: "global", layer: "runoff", mode: "annual",
  t: 0, month: 8, gauge: 0, range: "all", runId: 0, cell: null, playing: null, calTimer: null,
  grid: null, statics: {}, zones: null, history: [], pendingSnap: false, rtab: "hydro", opacity: 0.8,
  browse: false, level: 2, catLayer: null, catData: {}, pick: null, jobTimer: null, unit: "month",
  cmpSel: "", runs: [], calAt: null, sub: { flows: "hydro", balance: "partition" },
};
// Results are grouped; each group shows its views as sub-tabs
const RGROUPS = {
  flows: [["hydro", "Hydrograph"], ["season", "Seasonal cycle"]],
  balance: [["partition", "Partition"], ["subbasins", "Sub-basins"]],
  scenario: [["scenario", "Scenarios"]],
  cell: [["cell", "Cell"]],
};
const groupOf = (tab) => Object.keys(RGROUPS).find((g) => RGROUPS[g].some(([k]) => k === tab)) || "flows";
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Plain-language names and explanations for the model parameters
const PARAM_UI = {
  a: { name: "Runoff threshold", tip: "How full the soil must be before runoff starts. Close to 1: little runoff until the soil is nearly saturated. Lower: more runoff from moderate storms." },
  b: { name: "Soil storage capacity", unit: "mm", tip: "Water the soil can hold and give back as evaporation. Larger: more evaporation, less and slower runoff." },
  c: { name: "Groundwater share", tip: "Share of surplus water that recharges groundwater instead of running off straight away. Higher: more baseflow and a smoother hydrograph." },
  d: { name: "Groundwater drainage rate", unit: "/month", tip: "Share of groundwater released to rivers each month. Higher: groundwater empties faster, so dry-season flows fall faster." },
  umax: { name: "Surface storage", unit: "mm", tip: "Water held on leaves, in hollows and in the top of the soil. It evaporates first; rain beyond it becomes net rainfall for the root zone and runoff." },
  lmax: { name: "Root-zone storage", unit: "mm", tip: "Water the root zone can hold for plants. Larger: more evaporation and less runoff, and runoff starts later in the wet season." },
  cqof: { name: "Overland flow share", tip: "Share of net rainfall that runs off over the surface when the root zone is wet. Higher: flashier, larger wet-season peaks." },
  ckif: { name: "Interflow time", unit: "months", tip: "How slowly surface storage drains sideways as interflow. Shorter: more interflow each month." },
  tof: { name: "Overland flow threshold", tip: "How full the root zone (share of capacity) must be before overland flow starts." },
  tif: { name: "Interflow threshold", tip: "How full the root zone must be before interflow starts." },
  tg: { name: "Recharge threshold", tip: "How full the root zone must be before water recharges groundwater." },
  ckbf: { name: "Baseflow time", unit: "months", tip: "How slowly groundwater drains to rivers. Longer: smoother hydrograph and dry-season flow that lasts longer." },
  kc: { name: "PET multiplier", tip: "Scales the Modified Hargreaves potential evaporation. Above 1 raises evaporation demand everywhere in the zone." },
  kw: { name: "Lake evaporation factor", tip: "Evaporation from lakes and reservoirs, as a multiple of potential evaporation." },
  k_route: { name: "Channel delay", unit: "months", tip: "Holds water back in rivers and floodplains before it reaches the gauges. 0 means no delay." },
};
const FIT_COLORS = { good: "#3f8f4e", fair: "#c99a2e", poor: "#c0533b", none: "#9aa5a0" };
const fitClass = (k) => (k === null || k === undefined) ? "none" : k >= 0.7 ? "good" : k >= 0.5 ? "fair" : "poor";

// ------------------------------------------------------------------ utilities
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function toast(msg, isError = false, ms = 3200) {
  const t = $("#toast");
  t.textContent = msg; t.classList.toggle("error", isError); t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), ms);
}
async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? "–" :
  Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const debounce = (fn, ms) => { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; };
const hex = (h) => { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };

// ------------------------------------------------------------------ colour ramps
function ramp(stops) {
  const c = stops.map(hex);
  return (x) => {
    x = Math.min(1, Math.max(0, x)) * (c.length - 1);
    const i = Math.min(c.length - 2, Math.floor(x)), f = x - i;
    return [0, 1, 2].map((k) => Math.round(c[i][k] + (c[i + 1][k] - c[i][k]) * f));
  };
}
const RAMPS = {
  water: ramp(["#f3f7ef", "#c4e2da", "#79b9c9", "#2c7c99", "#103b4d"]),
  heat: ramp(["#fbf4de", "#f0d08a", "#d79c42", "#a2602a", "#5e3018"]),
  green: ramp(["#f4f6e6", "#cfe0a0", "#93b763", "#5e7d3a", "#2d4320"]),
  soil: ramp(["#f6efe0", "#dcc59b", "#9fae86", "#4a837a", "#1d4e52"]),
  terrain: ramp(["#e6eed6", "#c3d29b", "#d8bd84", "#ab7f52", "#6c4b35"]),
  div: ramp(["#8c4a1c", "#e3c08a", "#f2f2ec", "#8fc4d5", "#1f5f7a"]),
};
const LAYER_RAMP = { precip: "water", pet: "heat", et: "green", runoff: "water", recharge: "soil", baseflow: "water",
  soil: "soil", gw: "soil", discharge: "water", "static:dem": "terrain", "static:acc": "water" };
const ZONE_COLORS = ["#9aa5a0", "#3e7f9a", "#c98a2b", "#5e7d3a", "#8a5a8f", "#b4513a", "#3c9a8b", "#7a6a3a"];

// ------------------------------------------------------------------ map
let map, overlay, gaugeLayer, cellMarker;
// Background maps that need no API key. Esri first: it serves any referrer, including localhost.
const BASEMAPS = [
  { name: "Esri topographic", invert: true, url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri, HERE, Garmin, USGS, NGA, OpenStreetMap contributors" },
  { name: "Esri light grey", url: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri, HERE, Garmin, OpenStreetMap contributors", maxZoom: 16 },
  { name: "Esri dark grey", url: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri, HERE, Garmin, OpenStreetMap contributors", maxZoom: 16 },
  { name: "Esri satellite", url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri, Maxar, Earthstar Geographics" },
  { name: "OpenStreetMap", invert: true, url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap contributors" },
  { name: "OpenTopoMap", invert: true, url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", subdomains: "abc", maxZoom: 15,
    attribution: "&copy; OpenStreetMap contributors, SRTM | &copy; OpenTopoMap (CC-BY-SA)" },
];

function makeBase(b) {
  // light basemaps get a dark rendering in dark mode (CSS filter on .base-invert); satellite is left as it is
  const layer = L.tileLayer(b.url, { attribution: b.attribution, subdomains: b.subdomains || "abc", maxZoom: b.maxZoom || 17,
    className: b.invert ? "base-invert" : "" });
  // if a service refuses every tile, move on to the next one automatically
  let loaded = 0, failed = 0;
  layer.on("tileload", () => { loaded++; });
  layer.on("tileerror", () => {
    failed++;
    if (!loaded && failed === 6 && map.hasLayer(layer)) {
      autoSwitches++;
      if (autoSwitches >= BASEMAPS.length) {
        map.removeLayer(layer); baseLayers["No background"].addTo(map);
        toast("No background map could be loaded. Check the internet connection. The model layers still work.", true, 6000);
        return;
      }
      const next = BASEMAPS[(BASEMAPS.indexOf(b) + 1) % BASEMAPS.length];
      toast(`${b.name} map could not be loaded. Switching to ${next.name}.`);
      map.removeLayer(layer); baseLayers[next.name].addTo(map);
    }
  });
  return layer;
}
const baseLayers = {};
let autoSwitches = 0;

function initMap() {
  map = L.map("map", { zoomControl: true, attributionControl: true, preferCanvas: true, zoomSnap: 0.25 });
  BASEMAPS.forEach((b) => { baseLayers[b.name] = makeBase(b); });
  baseLayers["No background"] = L.layerGroup();
  let saved = null;
  try { saved = localStorage.getItem("volta-basemap"); } catch (_) { /* storage blocked */ }
  (baseLayers[saved] || baseLayers[BASEMAPS[0].name]).addTo(map);
  L.control.layers(baseLayers, null, { position: "topleft", collapsed: true }).addTo(map);
  map.on("baselayerchange", (e) => { try { localStorage.setItem("volta-basemap", e.name); } catch (_) { /* ignore */ } });
  map.createPane("outlines"); map.getPane("outlines").style.zIndex = 430; map.getPane("outlines").style.pointerEvents = "none";
  outlineLayer = L.layerGroup().addTo(map);
  highlightLayer = L.layerGroup().addTo(map);
  gaugeLayer = L.layerGroup().addTo(map);
  map.on("click", (e) => { if (!S.browse && S.meta) inspectCell(e.latlng); });
  map.on("mousemove", (e) => { if (S.browse) return; hoverLatLng = e.latlng; if (!hoverRaf) hoverRaf = requestAnimationFrame(showReadout); });
  map.on("mouseout", () => { $("#readout").hidden = true; });
}

// ---- cell lookup, hover readout, outlines -------------------------------------------
let outlineLayer, highlightLayer, hoverLatLng = null, hoverRaf = 0;
function cellAt(latlng) {
  const g = S.meta.grid;
  const row = Math.floor((g.north - latlng.lat) / g.res), col = Math.floor((latlng.lng - g.west) / g.res);
  if (row < 0 || col < 0 || row >= g.nrows || col >= g.ncols) return null;
  return { row, col, i: row * g.ncols + col };
}
function showReadout() {
  hoverRaf = 0;
  const el = $("#readout");
  if (!S.meta || !hoverLatLng || !S.statics.zone) { el.hidden = true; return; }
  const c = cellAt(hoverLatLng);
  const zid = c ? S.statics.zone.values[c.i] : null;
  if (zid === null || zid === undefined) { el.hidden = true; return; }
  const zone = S.meta.zones.find((z) => z.id === zid);
  const v = S.grid ? S.grid.values[c.i] : null;
  const lake = S.statics.lake && S.statics.lake.values[c.i] === 1;
  const name = FIELD_LABEL[S.layer] || S.layer;
  let val;
  if (S.layer === "static:zone") val = zone ? zone.name : zid;
  else if (S.layer === "static:lake") val = lake ? "Open water" : "Land";
  else val = `${v === null ? "–" : fmt(v, Math.abs(v) < 10 ? 1 : 0)} ${S.grid ? S.grid.units : ""}`;
  const dem = S.statics.dem ? S.statics.dem.values[c.i] : null;
  el.innerHTML = `<b>${name}: ${val}</b><br>${zone ? zone.name.replace(" (incremental)", "") : ""}${lake ? ", open water" : ""}` +
    `<br><span class="muted">${hoverLatLng.lat.toFixed(2)}° N, ${Math.abs(hoverLatLng.lng).toFixed(2)}° ${hoverLatLng.lng < 0 ? "W" : "E"}${dem !== null ? `, ${fmt(dem)} m` : ""}</span>`;
  el.hidden = false;
}

// Boundary segments between cells whose key differs (key = zone id; null outside the basin)
function boundarySegments(keyFn) {
  const g = S.meta.grid, segs = [];
  const lat = (r) => g.north - r * g.res, lon = (c) => g.west + c * g.res;
  for (let r = 0; r <= g.nrows; r++) {
    for (let c = 0; c <= g.ncols; c++) {
      const here = (r < g.nrows && c < g.ncols) ? keyFn(r * g.ncols + c) : null;
      if (c < g.ncols) {      // horizontal edge above (r, c)
        const up = r > 0 ? keyFn((r - 1) * g.ncols + c) : null;
        if (up !== here) segs.push({ a: up, b: here, p: [[lat(r), lon(c)], [lat(r), lon(c + 1)]] });
      }
      if (r < g.nrows) {      // vertical edge left of (r, c)
        const left = c > 0 ? keyFn(r * g.ncols + c - 1) : null;
        if (left !== here) segs.push({ a: left, b: here, p: [[lat(r), lon(c)], [lat(r + 1), lon(c)]] });
      }
    }
  }
  return segs;
}
function drawOutlines() {
  outlineLayer.clearLayers();
  if (!S.statics.zone) return;
  const z = S.statics.zone.values;
  const segs = boundarySegments((i) => (z[i] === null || z[i] === undefined) ? null : z[i]);
  const outer = segs.filter((s) => s.a === null || s.b === null).map((s) => s.p);
  const inner = segs.filter((s) => s.a !== null && s.b !== null).map((s) => s.p);
  const opts = { pane: "outlines", interactive: false };
  L.polyline(outer, { ...opts, color: "#ffffff", weight: 4, opacity: 0.8 }).addTo(outlineLayer);
  L.polyline(outer, { ...opts, color: css("--ink"), weight: 1.6, opacity: 0.9 }).addTo(outlineLayer);
  L.polyline(inner, { ...opts, color: css("--ink"), weight: 1.1, opacity: 0.65, dashArray: "4 4" }).addTo(outlineLayer);
}
function drawZoneHighlight() {
  highlightLayer.clearLayers();
  if (S.zone === "global" || !S.statics.zone) return;
  const zid = +S.zone, z = S.statics.zone.values;
  const segs = boundarySegments((i) => z[i] === zid ? 1 : null).map((s) => s.p);
  L.polyline(segs, { pane: "outlines", interactive: false, color: "#ffffff", weight: 6, opacity: 0.9 }).addTo(highlightLayer);
  L.polyline(segs, { pane: "outlines", interactive: false, color: css("--ochre"), weight: 3.2 }).addTo(highlightLayer);
}

function fitBasin() {
  const b = S.meta.basin_bounds || S.meta.bounds;
  map.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [16, 16] });
}

function drawGauges() {
  gaugeLayer.clearLayers();
  S.meta.gauges.forEach((g, i) => {
    const rep = S.res && S.res.gauges[i];
    const kge = rep ? rep.metrics.all.kge : null;
    const cls = fitClass(kge);
    const m = L.circleMarker([g.lat, g.lon], {
      radius: i === S.gauge ? 9 : 7, color: "#fff", weight: i === S.gauge ? 3 : 2, fillColor: FIT_COLORS[cls], fillOpacity: 1,
    }).bindTooltip(`${g.name.split(" (")[0]} <small>KGE ${kge === null ? "–" : fmt(kge, 2)}</small>`,
      { permanent: true, direction: crowdedEast(g) ? "left" : "right", offset: [crowdedEast(g) ? -9 : 9, 0], className: "gauge-label" });
    m.on("click", (e) => { L.DomEvent.stopPropagation(e); selectGauge(i, true); });
    m.addTo(gaugeLayer);
  });
}
// put a label on the left when another gauge sits just to the east (labels would overlap)
function crowdedEast(g) {
  return S.meta.gauges.some((h) => h !== g && h.lon > g.lon && h.lon - g.lon < 1.8 && Math.abs(h.lat - g.lat) < 0.6);
}
function selectGauge(i, showTab) {
  S.gauge = i; $("#gaugeSelect").value = i;
  drawHydrograph(); drawFitStrip(); drawGauges();
  if (showTab) setRtab("hydro");
}

const mapCache = new Map();
async function fetchLayer() {
  const isStatic = S.layer.startsWith("static:");
  let url;
  if (isStatic) url = `/api/static/${S.layer.slice(7)}`;
  else {
    const q = new URLSearchParams({ var: S.layer, mode: S.mode });
    if (S.mode === "month" || S.mode === "anomaly") q.set("t", S.t);
    if (S.mode === "clim") q.set("month", S.month);
    url = `/api/map?${q}`;
  }
  const key = `${url}|${isStatic ? 0 : S.runId}`;
  if (!mapCache.has(key)) {
    if (mapCache.size > 120) mapCache.delete(mapCache.keys().next().value);
    mapCache.set(key, api(url));
  }
  updateExportLink(isStatic);
  return mapCache.get(key);
}

function updateExportLink(isStatic) {
  const a = $("#exportMap");
  if (isStatic) { a.removeAttribute("href"); a.textContent = "Current map (pick a model layer)"; return; }
  const q = new URLSearchParams({ var: S.layer, mode: S.mode });
  if (S.mode === "month" || S.mode === "anomaly") q.set("t", S.t);
  if (S.mode === "clim") q.set("month", S.month);
  a.href = `/api/export/map.asc?${q}`; a.textContent = "Current map (ASCII grid)";
}

async function renderMap() {
  if (!S.meta) return;
  let grid;
  try { grid = await fetchLayer(); } catch (e) { toast(e.message, true); return; }
  const { nrows, ncols, values } = grid;
  const cvs = document.createElement("canvas");
  cvs.width = ncols; cvs.height = nrows;
  const ctx = cvs.getContext("2d");
  const img = ctx.createImageData(ncols, nrows);
  const layer = S.layer;
  const cat = layer === "static:zone" || layer === "static:lake";
  const logScale = layer === "discharge" || layer === "static:acc";
  const anomaly = S.mode === "anomaly" && !layer.startsWith("static:");
  let lo = grid.p2, hi = grid.p98;
  if (logScale) { lo = Math.log10(Math.max(grid.p98 / 3000, 1e-3)); hi = Math.log10(Math.max(grid.max, 1e-3)); }
  if (anomaly) { const m = Math.max(Math.abs(grid.p2), Math.abs(grid.p98)) || 1; lo = -m; hi = m; }
  if (hi <= lo) hi = lo + 1;
  const rf = anomaly ? RAMPS.div : RAMPS[LAYER_RAMP[layer] || "water"];
  S.grid = grid;
  // on land-process layers the lake would show as an extreme value; paint it as water instead
  const lakeVals = S.statics.lake ? S.statics.lake.values : null;
  const paintLake = lakeVals && ["runoff", "recharge", "baseflow", "soil", "gw"].includes(layer);
  const lakeRgb = hex("#5f93b3");
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null) continue;
    let rgb, a = 235;
    if (paintLake && lakeVals[i] === 1) { img.data.set([lakeRgb[0], lakeRgb[1], lakeRgb[2], 255], i * 4); continue; }
    if (cat) {
      if (layer === "static:lake") { if (!v) { rgb = hex("#d8ddd6"); a = 120; } else rgb = hex(css("--water")); }
      else rgb = hex(ZONE_COLORS[v % ZONE_COLORS.length]);
    } else if (logScale) {
      const x = (Math.log10(Math.max(v, 1e-6)) - lo) / (hi - lo);
      rgb = rf(0.15 + 0.85 * x); a = Math.round(255 * Math.min(1, Math.max(0.04, x * 1.3)));
    } else {
      rgb = rf((v - lo) / (hi - lo));
    }
    img.data.set([rgb[0], rgb[1], rgb[2], a], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  const b = S.meta.bounds;
  const bounds = [[b.south, b.west], [b.north, b.east]];
  const url = cvs.toDataURL();
  if (!overlay) overlay = L.imageOverlay(url, bounds, { opacity: S.browse ? 0 : S.opacity, className: "pixel", interactive: false }).addTo(map);
  else { overlay.setBounds(L.latLngBounds(bounds)); overlay.setUrl(url); }
  drawLegend(grid, { lo, hi, rf, logScale, cat, anomaly, paintLake });
  if (hoverLatLng && !$("#readout").hidden) showReadout();
}

function drawLegend(grid, o) {
  const el = $("#legend");
  const name = FIELD_LABEL[S.layer] || S.layer;
  let when = "";
  if (!S.layer.startsWith("static:")) {
    when = { annual: "mean annual", clim: `mean ${MONTHS[S.month - 1]}`, month: S.meta.dates[S.t],
             anomaly: `${S.meta.dates[S.t]} vs. its monthly mean` }[S.mode];
  }
  if (o.cat) {
    if (S.layer === "static:zone") {
      el.innerHTML = `<div class="ttl">Parameter zones</div><div class="cats">${S.meta.zones.map((z) =>
        `<div><span style="background:${ZONE_COLORS[z.id % ZONE_COLORS.length]}"></span>${z.name}</div>`).join("")}</div>`;
    } else {
      el.innerHTML = `<div class="ttl">Open water</div><div class="cats"><div><span style="background:${css("--water")}"></span>Lake cells (${S.meta.lake_cells})</div></div>`;
    }
    return;
  }
  const c = document.createElement("canvas"); c.width = 200; c.height = 1;
  const cx = c.getContext("2d");
  for (let x = 0; x < 200; x++) { const rgb = o.rf(o.logScale ? 0.15 + 0.85 * x / 199 : x / 199); cx.fillStyle = `rgb(${rgb})`; cx.fillRect(x, 0, 1, 1); }
  const tick = (v) => o.logScale ? fmt(10 ** v, v < 1 ? 1 : 0) : fmt(v, Math.abs(o.hi - o.lo) < 10 ? 1 : 0);
  el.innerHTML = `<div class="ttl">${name}${when ? `, ${when}` : ""}</div>`;
  el.appendChild(c);
  el.insertAdjacentHTML("beforeend", `<div class="ticks"><span>${tick(o.lo)}</span><span>${tick((o.lo + o.hi) / 2)}</span><span>${tick(o.hi)} ${grid.units}</span></div>`);
  if (o.paintLake && S.meta.lake_cells) el.insertAdjacentHTML("beforeend", `<div class="cats"><div><span style="background:#5f93b3"></span>Lakes (open water)</div></div>`);
}

async function inspectCell(latlng, switchTab) {
  const g = S.meta.grid;
  const cc = cellAt(latlng);
  if (!cc) return;
  const { row, col } = cc;
  try {
    S.cell = await api(`/api/cell?row=${row}&col=${col}`);
  } catch (e) { toast("That point is outside the basin.", true); return; }
  const c = S.cell;
  const r = g.res;
  const top = g.north - row * r, left = g.west + col * r;
  if (cellMarker) cellMarker.remove();
  cellMarker = L.rectangle([[top - r, left], [top, left + r]], { color: css("--ink"), weight: 2, fill: false }).addTo(map);
  if (switchTab !== false) setRtab("cell");
  drawCell();
}

// ------------------------------------------------------------------ ribbon (time scrubber)
function drawRibbon() {
  const cv = $("#ribbon");
  if (!S.res) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const p = S.res.basin.precip, T = p.length;
  const max = Math.max(...p) || 1;
  const bw = w / T;
  const barH = h - 14;
  for (let i = 0; i < T; i++) {
    const x = i * bw, v = p[i] / max;
    ctx.fillStyle = `rgb(${RAMPS.water(0.1 + 0.9 * v)})`;
    const bh = Math.max(2, v * barH);
    ctx.fillRect(x, barH - bh, Math.max(1, bw - (bw > 3 ? 0.5 : 0)), bh);
  }
  ctx.fillStyle = css("--ink-2"); ctx.font = `11px ${css("--font")}`;
  ctx.textBaseline = "bottom";
  for (let i = 0; i < T; i++) {
    const d = S.meta.dates[i];
    if (d.endsWith("-01") && (+d.slice(0, 4)) % 5 === 0) {
      ctx.fillRect(i * bw, barH, 1, 3);
      ctx.fillText(d.slice(0, 4), i * bw + 2, h);
    }
  }
  if (S.mode === "month" || S.mode === "anomaly") {
    ctx.fillStyle = css("--ochre");
    ctx.fillRect(S.t * bw - 1, 0, Math.max(2, bw + 2), barH + 3);
  }
  $("#ribbonDate").textContent = (S.mode === "month" || S.mode === "anomaly")
    ? `${MONTHS[+S.meta.dates[S.t].slice(5) - 1]} ${S.meta.dates[S.t].slice(0, 4)}: ${fmt(p[S.t])} mm basin rainfall`
    : `${S.meta.dates[0]} to ${S.meta.dates[T - 1]}`;
}
function ribbonPick(ev) {
  const cv = $("#ribbon");
  const rect = cv.getBoundingClientRect();
  const T = S.meta.dates.length;
  const t = Math.max(0, Math.min(T - 1, Math.floor((ev.clientX - rect.left) / rect.width * T)));
  if (S.mode !== "month" && S.mode !== "anomaly") setMode("month", false);
  if (t !== S.t) { S.t = t; drawRibbon(); renderMapSoon(); drawHydrograph(false); }
}
const renderMapSoon = debounce(renderMap, 60);

function setMode(mode, render = true) {
  S.mode = mode;
  $$("#modeSeg button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === mode));
  $("#climMonth").hidden = mode !== "clim";
  drawRibbon();
  if (render) renderMap();
}

// ------------------------------------------------------------------ charts
Chart.defaults.font.family = "Archivo, system-ui, sans-serif";
Chart.defaults.font.size = 11.5;
Chart.defaults.animation = false;
let hydroChart, seasonChart, cellChart, partChart, shareChart, scnSeasonChart, scnAnnualChart;

const periodShade = {
  id: "periodShade",
  beforeDatasetsDraw(chart) {
    if (!S.res || chart !== hydroChart) return;
    const { ctx, chartArea: a, scales: { x } } = chart;
    const dates = S.meta.dates;
    const band = (p, color) => {
      const i0 = dates.indexOf(p[0]), i1 = dates.indexOf(p[1]);
      if (i0 < 0 || i1 < 0) return;
      const x0 = Math.max(a.left, x.getPixelForValue(i0)), x1 = Math.min(a.right, x.getPixelForValue(i1));
      if (x1 <= x0) return;
      ctx.fillStyle = color; ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top);
    };
    ctx.save();
    band(S.res.periods.cal, "rgba(31,111,139,0.07)");
    if (S.mode === "month" || S.mode === "anomaly") {
      const px = x.getPixelForValue(S.t);
      if (px >= a.left && px <= a.right) { ctx.fillStyle = css("--ochre"); ctx.fillRect(px - 0.75, a.top, 1.5, a.bottom - a.top); }
    }
    ctx.restore();
  },
};
Chart.register(periodShade);

function axisColors() {
  return { grid: css("--rule"), ticks: css("--ink-2") };
}
function baseOpts(yTitle) {
  const c = axisColors();
  return {
    responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: { boxWidth: 12, boxHeight: 2, color: c.ticks } } },
    scales: {
      x: { grid: { display: false }, ticks: { color: c.ticks, maxRotation: 0, autoSkip: true, maxTicksLimit: 9,
        callback(v) { const l = this.getLabelForValue(v); return /^\d{4}-\d{2}$/.test(l) ? l.slice(0, 4) : l; } } },
      y: { grid: { color: c.grid }, ticks: { color: c.ticks }, title: { display: !!yTitle, text: yTitle, color: c.ticks } },
    },
  };
}

// Annual volumes in million m³ from monthly mean flows (m³/s). A year with any month missing is left out.
function annualMCM(series) {
  const by = new Map();
  S.meta.dates.forEach((d, i) => {
    const y = +d.slice(0, 4), m = +d.slice(5);
    const secs = new Date(Date.UTC(y, m, 0)).getUTCDate() * 86400;
    const e = by.get(y) || { v: 0, n: 0 };
    if (series[i] !== null && series[i] !== undefined) { e.v += series[i] * secs / 1e6; e.n++; }
    by.set(y, e);
  });
  const years = [...by.keys()].sort((p, q) => p - q);
  return { years: years.map(String), values: years.map((y) => by.get(y).n === 12 ? by.get(y).v : null) };
}

function drawHydrograph(full = true) {
  if (!S.res) return;
  const g = S.res.gauges[S.gauge];
  const annual = S.unit === "year";
  $("#gaugeField").hidden = !g; $("#outletTitle").hidden = !!g;
  const simM = g ? g.sim : S.res.basin.outlet_q;
  const obsM = g ? g.obs : null;
  const hasObs = !!obsM && obsM.some((v) => v !== null);
  let labels = S.meta.dates, sim = simM, obs = obsM;
  if (annual) {
    const a = annualMCM(simM); labels = a.years; sim = a.values;
    obs = hasObs ? annualMCM(obsM).values : null;
  }
  const cmpOn = !!(S.cmp && S.cmpSel);
  let cmpSeries = null;
  if (cmpOn) {
    const cg = g ? S.cmp.gauges.find((x) => x.id === g.id) : null;
    const raw = g ? (cg ? cg.sim : null) : S.cmp.outlet_q;
    cmpSeries = raw && (annual ? annualMCM(raw).values : raw);
  }
  const sig = `${annual}|${hasObs}|${!!g}|${!!cmpSeries}`;
  if (hydroChart && hydroChart.$sig !== sig) { hydroChart.destroy(); hydroChart = null; }
  const simLabel = `${g ? "Simulated" : "Simulated outflow"}, ${modelLabel()}`;
  if (!hydroChart) {
    const ds = [];
    if (hasObs) ds.push(annual
      ? { type: "bar", label: "Observed", data: obs, backgroundColor: css("--ochre"), borderWidth: 0, order: 2 }
      : { label: "Observed", data: obs, borderColor: css("--ochre"), backgroundColor: css("--ochre"), borderWidth: 1.6,
          pointRadius: obs.filter((v) => v !== null).length > 120 ? 0 : 2, pointHoverRadius: 4, spanGaps: false, order: 1 });
    ds.push(annual
      ? { type: "bar", label: simLabel, data: sim, backgroundColor: css("--water"), borderWidth: 0, order: 3 }
      : { label: simLabel, data: sim, borderColor: css("--water"), borderWidth: 1.4, pointRadius: 0, order: 2 });
    if (cmpSeries) ds.push(annual
      ? { type: "line", label: S.cmp.label, data: cmpSeries, borderColor: css("--ink-2"), borderDash: [5, 3], borderWidth: 1.5,
          pointRadius: 3, pointBackgroundColor: css("--surface"), pointBorderColor: css("--ink-2"), spanGaps: false, order: 0, $cmp: true }
      : { label: S.cmp.label, data: cmpSeries, borderColor: css("--ink-2"), borderDash: [5, 3], borderWidth: 1.4, pointRadius: 0, order: 3, $cmp: true });
    hydroChart = new Chart($("#hydroChart"), { type: annual ? "bar" : "line", data: { labels, datasets: ds },
      options: baseOpts(annual ? "million m³ per year" : "m³/s") });
    hydroChart.$sig = sig;
  } else if (full) {
    hydroChart.data.labels = labels;
    const dsets = hydroChart.data.datasets;
    if (hasObs) dsets[0].data = obs;
    dsets[hasObs ? 1 : 0].data = sim;
    dsets[hasObs ? 1 : 0].label = simLabel;
    if (cmpSeries) { dsets.at(-1).data = cmpSeries; dsets.at(-1).label = S.cmp.label; }
  }
  const p = S.res.periods;
  const rng = S.range === "cal" ? p.cal : S.range === "val" ? p.val : null;
  const idx = (d) => annual ? labels.indexOf(d.slice(0, 4)) : labels.indexOf(d);
  hydroChart.options.scales.x.min = rng ? idx(rng[0]) : undefined;
  hydroChart.options.scales.x.max = rng ? idx(rng[1]) : undefined;
  hydroChart.update("none");
  if (!full) return;
  drawCompareTable(g, sim, cmpSeries, annual);
  if (g && hasObs) { drawMetrics(g, annual ? { sim, obs, labels } : null); return; }
  const vals = sim.filter((v) => v !== null);
  const mean = vals.reduce((x, v) => x + v, 0) / (vals.length || 1);
  const u = annual ? "MCM/yr" : "m³/s";
  const what = annual ? "year" : "month";
  const note = g
    ? `No observed discharge for ${g.name} yet, so there is nothing to compare with. Add gauge records in the Data tab to see fit scores.`
    : "This basin has no gauges, so the chart shows simulated outflow at the basin outlet.";
  $("#metricsTable").innerHTML = `<tbody><tr><td>Mean ${annual ? "annual volume" : "flow"}</td><td>${fmt(mean)} ${u}</td></tr>
    <tr><td>Highest ${what}</td><td>${fmt(Math.max(...vals))} ${u}</td></tr><tr><td>Lowest ${what}</td><td>${fmt(Math.min(...vals), 1)} ${u}</td></tr>
    <tr><td colspan="2" class="muted">${note}</td></tr></tbody>`;
}

// ---- comparison: the other model (its own last parameters) or any saved run, recomputed on the current data
function buildCmpOptions() {
  if (!S.meta) return;
  const sel = $("#cmpSelect");
  const others = (S.meta.models || []).filter((m) => m.key !== S.meta.model);
  let html = `<option value="">Nothing</option>` +
    others.map((m) => `<option value="model:${m.key}">${esc(m.label)}, its last parameters</option>`).join("");
  if (S.runs.length) html += `<optgroup label="Saved runs">` +
    S.runs.map((r) => `<option value="run:${r.id}">${esc(r.label)}${r.climate_label ? ` (${esc(r.climate_label)})` : ""}</option>`).join("") + `</optgroup>`;
  sel.innerHTML = html;
  if (S.cmpSel && ![...sel.options].some((o) => o.value === S.cmpSel)) { S.cmpSel = ""; S.cmp = null; }
  sel.value = S.cmpSel;
}
async function loadCompare() {
  const v = S.cmpSel;
  if (!v) { S.cmp = null; drawHydrograph(); return; }
  try {
    const c = v.startsWith("run:") ? await api(`/api/runs/${encodeURIComponent(v.slice(4))}`)
      : await api(`/api/compare?model=${encodeURIComponent(v.slice(6))}`);
    if (S.cmpSel !== v) return;                     // the choice changed while this was running
    c.kind = v.startsWith("run:") ? "run" : "model";
    S.cmp = c;
  } catch (e) { toast(e.message, true); S.cmpSel = ""; S.cmp = null; $("#cmpSelect").value = ""; }
  drawHydrograph();
}
function cmpNote() {
  const c = S.cmp;
  if (c.kind === "run") {
    const r = S.runs.find((x) => x.id === c.run_id);
    return `Saved run${r ? ` from ${r.created}, ${esc(r.model_label)}` : ""}${c.climate ? ", with its climate change" : ""}, recomputed on the current data. Use it from the Model step to edit it.`;
  }
  return `${esc(c.label)} uses ${c.calibrated ? "its last calibrated" : "its last-used (or default)"} parameters. Switch model in the Model step to edit or calibrate it.`;
}
function drawCompareTable(g, sim, cmpSeries, annual) {
  const t = $("#cmpTable");
  if (!cmpSeries) { t.hidden = true; return; }
  t.hidden = false;
  const mean = (a) => { const v = a.filter((x) => x !== null); return v.reduce((s, x) => s + x, 0) / (v.length || 1); };
  const u = annual ? "MCM/yr" : "m³/s";
  const cg = g ? S.cmp.gauges.find((x) => x.id === g.id) : null;
  const rows = [
    { key: "", label: `Current, ${modelLabel()}`, m: g ? g.metrics.all : null, q: mean(sim) },
    { key: "dash", label: esc(S.cmp.label), m: cg ? cg.metrics.all : null, q: mean(cmpSeries) },
  ];
  const hasObs = g && g.metrics.all.n > 0;
  const best = (k, hi = true) => {
    const v = rows.map((r) => r.m ? r.m[k] : null);
    if (v.some((x) => x === null)) return -1;
    return k === "pbias" ? (Math.abs(v[0]) <= Math.abs(v[1]) ? 0 : 1) : ((hi ? v[0] >= v[1] : v[0] <= v[1]) ? 0 : 1);
  };
  const cell = (r, i, k, d, suf = "") => `<td class="${hasObs && best(k) === i ? "best" : ""}">${r.m && r.m[k] !== null ? fmt(r.m[k], d) + suf : "–"}</td>`;
  t.innerHTML = `<thead><tr><th>Comparison<br><span class="muted">${hasObs ? "scores on monthly flows, all years" : "simulated only"}</span></th>
      ${hasObs ? "<th>KGE</th><th>NSE</th><th>Bias</th>" : ""}<th>Mean ${annual ? "volume" : "flow"}</th></tr></thead><tbody>` +
    rows.map((r, i) => `<tr><td><span class="key ${r.key}" style="border-color:${i ? css("--ink-2") : css("--water")}"></span>${r.label}</td>
      ${hasObs ? cell(r, i, "kge", 2) + cell(r, i, "nse", 2) + cell(r, i, "pbias", 0, "%") : ""}<td>${fmt(r.q)} ${u}</td></tr>`).join("") +
    `<tr><td colspan="5" class="muted">${cmpNote()}</td></tr></tbody>`;
}

function drawMetrics(g, annual) {
  const rows = [["NSE", "nse", 2], ["KGE", "kge", 2], ["log-NSE", "log_nse", 2], ["Bias", "pbias", 1, "%"], ["R²", "r2", 2], ["Months with data", "n", 0]];
  const per = S.res.periods;
  const head = `<thead><tr><th>${g.name}, ${fmt(g.area_km2)} km²<br><span class="muted">monthly scores</span></th><th>Calibration<br><span class="muted">${per.cal[0].slice(0, 4)}–${per.cal[1].slice(0, 4)}</span></th><th>Validation<br><span class="muted">${per.val[0].slice(0, 4)}–${per.val[1].slice(0, 4)}</span></th><th>All</th></tr></thead>`;
  let body = rows.map(([lab, k, d, u = ""]) => `<tr><td>${lab}</td>${["cal", "val", "all"].map((s) => {
    const v = g.metrics[s][k];
    return `<td>${v === null ? "–" : fmt(v, d) + u}</td>`;
  }).join("")}</tr>`).join("");
  if (annual) {
    const pairs = annual.obs.map((o, i) => [o, annual.sim[i]]).filter(([o, s]) => o !== null && s !== null);
    const mo = pairs.reduce((x, [o]) => x + o, 0) / (pairs.length || 1), ms = pairs.reduce((x, [, s]) => x + s, 0) / (pairs.length || 1);
    body += `<tr><td>Mean annual volume</td><td colspan="3">observed ${fmt(mo)}, simulated ${fmt(ms)} MCM/yr (${pairs.length} complete years)</td></tr>`;
  }
  $("#metricsTable").innerHTML = head + `<tbody>${body}</tbody>`;
}

function monthlyMeans(series) {
  const sums = new Array(12).fill(0), n = new Array(12).fill(0);
  S.meta.dates.forEach((d, i) => { const m = +d.slice(5) - 1; if (series[i] !== null) { sums[m] += series[i]; n[m]++; } });
  return sums.map((s, i) => n[i] ? s / n[i] : null);
}

function drawSeason() {
  const b = S.res.basin;
  const ds = [
    { type: "bar", label: "Rainfall", data: monthlyMeans(b.precip), backgroundColor: "rgba(121,185,201,0.55)", borderWidth: 0, order: 3 },
    { label: "PET", data: monthlyMeans(b.pet), borderColor: css("--ochre"), borderDash: [5, 4], borderWidth: 1.6, pointRadius: 0, order: 1 },
    { label: "Actual ET", data: monthlyMeans(b.et), borderColor: css("--savanna"), borderWidth: 2, pointRadius: 0, order: 1 },
    { label: "Runoff", data: monthlyMeans(b.runoff), borderColor: css("--water-deep"), borderWidth: 2, pointRadius: 0, order: 1 },
    { label: "Recharge", data: monthlyMeans(b.recharge), borderColor: css("--ink-2"), borderDash: [2, 3], borderWidth: 1.4, pointRadius: 0, order: 1 },
  ];
  if (seasonChart) seasonChart.destroy();
  seasonChart = new Chart($("#seasonChart"), { type: "line", data: { labels: MONTHS, datasets: ds }, options: baseOpts("mm/month") });
}

function drawCell() {
  const c = S.cell;
  if (!c) return;
  const zone = S.meta.zones.find((z) => z.id === c.zone);
  $("#cellTitle").textContent = `Cell at ${c.lat.toFixed(3)}° N, ${Math.abs(c.lon).toFixed(3)}° ${c.lon < 0 ? "W" : "E"}`;
  $("#cellHint").hidden = true;
  const yrs = S.meta.dates.length / 12;
  const sum = (k) => c.series[k].reduce((a, v) => a + v, 0) / yrs;
  $("#cellMeta").innerHTML = `
    <dt>Zone</dt><dd>${zone ? zone.name : c.zone}${c.lake ? ", open water" : ""}</dd>
    <dt>Elevation</dt><dd>${fmt(c.dem)} m, ${fmt(c.upstream_km2)} km² drains through this cell</dd>
    <dt>Parameters</dt><dd>${modelDef(c.model || S.meta.model).label}: ${Object.entries(c.params).map(([k, v]) => `${k} ${fmt(v, decFor(k))}`).join(", ")}</dd>
    <dt>Mean annual</dt><dd>P ${fmt(sum("precip"))}, ET ${fmt(sum("et"))}, runoff ${fmt(sum("runoff"))}, recharge ${fmt(sum("recharge"))} mm</dd>`;
  const labels = S.meta.dates;
  const ds = [
    { type: "bar", label: "Rainfall", data: c.series.precip, backgroundColor: "rgba(121,185,201,0.6)", borderWidth: 0, order: 3 },
    { label: "Actual ET", data: c.series.et, borderColor: css("--savanna"), borderWidth: 1.3, pointRadius: 0, order: 1 },
    { label: "Runoff", data: c.series.runoff, borderColor: css("--water-deep"), borderWidth: 1.3, pointRadius: 0, order: 1 },
    { label: "Soil moisture", data: c.series.soil, borderColor: css("--ochre"), borderWidth: 1.2, pointRadius: 0, yAxisID: "y2", order: 1 },
  ];
  const o = baseOpts("mm/month");
  o.scales.y2 = { position: "right", grid: { display: false }, ticks: { color: css("--ink-2") }, title: { display: true, text: "soil, mm", color: css("--ink-2") } };
  if (cellChart) cellChart.destroy();
  cellChart = new Chart($("#cellChart"), { type: "line", data: { labels, datasets: ds }, options: o });
}

function drawBalance() {
  const b = S.res.balance, y = b.per_year;
  const err = Math.abs(b.residual) < 5e-4 ? 0 : b.residual;
  const ok = Math.abs(b.residual) < 0.01 * b.P;
  const tot = y.ET + y.Q || 1;
  const outM3s = y.Q * b.basin_area_km2 / 31557.6;
  const tile = (v, u, k, tip) => `<div class="kpi" title="${tip}"><div class="v">${v}<small>${u}</small></div><div class="k">${k}</div></div>`;
  $("#balance").innerHTML = `<div class="kpi-row">
      ${tile(fmt(y.P), "mm/yr", "Rainfall", "Mean rainfall over the basin")}
      ${tile(fmt(y.PET), "mm/yr", "PET", "Potential evaporation (Modified Hargreaves, times the PET multiplier)")}
      ${tile(fmt(y.ET), "mm/yr", "ET", `Actual evaporation, ${fmt(100 * y.ET / (y.PET || 1))} % of PET`)}
      ${tile(fmt(outM3s), "m³/s", "Outflow", `Mean outflow at the outlet: ${fmt(y.Q)} mm/yr, ${fmt(outM3s * 31.5576)} million m³/yr`)}
      ${tile(fmt(100 * b.runoff_ratio, 1), "%", "Runoff", "Runoff ratio: outflow as a share of rainfall")}
    </div>
    <div class="balance-foot">
      <div class="balance-bar" title="Where the rain goes: ${fmt(100 * y.ET / tot)} % evaporates, ${fmt(100 * y.Q / tot)} % leaves as river flow">
        <i style="width:${100 * y.ET / tot}%;background:${css("--savanna")}"></i><i style="width:${100 * y.Q / tot}%;background:${css("--water")}"></i></div>
      <span class="closure ${ok ? "ok" : "bad"}" title="Water-balance closure error over ${fmt(b.years)} years: ${fmt(err, 3)} mm (rainfall minus ET, outflow and storage change)">${ok ? "✓" : "!"} balance ${ok ? "closes" : `off ${fmt(err, 1)} mm`}</span>
    </div>`;
  $("#runInfo").textContent = `${modelLabel()}${S.res.seconds ? `, run ${S.res.seconds}s` : ""}`;
  drawFitStrip();
}

// One chip per gauge: name and KGE, coloured and worded; click to open that gauge's hydrograph
function drawFitStrip() {
  const el = $("#fitStrip");
  const gs = (S.res && S.res.gauges) || [];
  if (!gs.length) { el.hidden = true; return; }
  el.hidden = false;
  const word = { good: "good", fair: "fair", poor: "poor" };
  const scored = gs.map((g, i) => ({ g, i })).filter(({ g }) => g.metrics.all.n > 0);
  const missing = gs.length - scored.length;
  el.innerHTML = `<span class="fs-label">Fit (KGE)</span>` + scored.map(({ g, i }) => {
    const k = g.metrics.all.kge, cls = fitClass(k);
    return `<button type="button" class="chip ${cls}" data-g="${i}" aria-pressed="${i === S.gauge}"
      title="${esc(g.name)}: ${word[cls]} fit, KGE ${fmt(k, 2)}, NSE ${fmt(g.metrics.all.nse, 2)}, bias ${g.metrics.all.pbias === null ? "–" : fmt(g.metrics.all.pbias, 0) + " %"} (all years). Click to see its hydrograph.">
      <i style="background:${FIT_COLORS[cls]}"></i>${esc(g.name)} <b>${fmt(k, 2)}</b><span class="w">${word[cls]}</span></button>`;
  }).join("") + (missing ? `<button type="button" class="chip none" data-step="data" title="Gauges without observed discharge. Add records in step 2, Gauges.">+${missing} without records</button>` : "")
    + (!scored.length ? `<span class="fs-none">No gauge has observed discharge yet.</span>` : "");
  $$("button.chip[data-g]", el).forEach((b) => b.addEventListener("click", () => selectGauge(+b.dataset.g, true)));
  const m = $("button.chip[data-step]", el);
  if (m) m.addEventListener("click", () => openStep("data"));
  const cur = $('button.chip[aria-pressed="true"]', el);
  if (cur) cur.scrollIntoView({ block: "nearest", inline: "nearest" });
}

// ------------------------------------------------------------------ workflow steps (left tabs)
function openStep(name) { const b = $$(".tabs button").find((x) => x.dataset.tab === name); if (b) b.click(); }
function drawSteps() {
  const set = (k, text, done, tip) => {
    const st = $(`#st-${k}`); st.textContent = text || ""; st.parentElement.title = tip || text || "";
    st.parentElement.classList.toggle("done", !!done);
  };
  if (!S.meta || !S.res) { set("basins", "pick one", false); set("data", "", false); set("params", "", false); set("calib", "", false); return; }
  set("basins", S.meta.name, true, `${S.meta.name}: ${fmt(S.meta.basin_area_km2)} km²`);
  const gs = S.res.gauges, withObs = gs.filter((g) => g.metrics.all.n > 0).length;
  set("data", gs.length ? `${withObs}/${gs.length} with data` : "no gauges", withObs > 0,
    gs.length ? `${withObs} of ${gs.length} gauges have observed discharge` : "This basin has no gauges");
  const p = S.res.params, spec = S.meta.param_spec;
  const edited = Object.entries(spec).some(([k, v]) => Math.abs(p.global[k] - v.default) > 1e-9) || Object.keys(p.zones || {}).length > 0
    || Object.entries(S.meta.global_spec).some(([k, v]) => Math.abs(p[k] - v.default) > 1e-9);
  set("params", `${modelLabel()}, ${edited ? "edited" : "defaults"}`, edited);
  const running = !!S.calRunning;
  set("calib", running ? "running…" : S.calAt ? `calibrated ${S.calAt}` : withObs ? "not yet" : "needs records", !!S.calAt && !running);
}

// ------------------------------------------------------------------ saved runs
async function loadRuns() {
  try { S.runs = (await api("/api/runs")).runs; } catch (_) { S.runs = []; }
  drawRuns(); buildCmpOptions();
}
function drawRuns() {
  const ul = $("#runList");
  if (!S.runs.length) { ul.innerHTML = `<li class="muted">No saved runs yet.</li>`; return; }
  ul.innerHTML = S.runs.slice().reverse().map((r) => {
    const sm = r.summary || {};
    const facts = [r.model_label, r.climate_label, sm.runoff_ratio != null ? `runoff ${fmt(100 * sm.runoff_ratio, 1)} %` : "",
      sm.median_kge != null ? `median KGE ${fmt(sm.median_kge, 2)}` : ""].filter(Boolean).join(" · ");
    const on = S.cmpSel === `run:${r.id}`;
    return `<li data-id="${r.id}"><div class="rn"><span class="rl">${esc(r.label)}</span><span class="rd">${esc(r.created)}</span></div>
      <div class="rf">${esc(facts)}</div>
      <div class="ra"><button type="button" class="btn ghost small" data-act="cmp" aria-pressed="${on}">${on ? "Comparing" : "Compare"}</button>
        <button type="button" class="btn ghost small" data-act="use" title="Make these the current parameters${r.climate ? " (the climate change is not applied)" : ""}">Use</button>
        <button type="button" class="btn ghost small" data-act="ren">Rename</button>
        <button type="button" class="btn ghost small danger" data-act="del">Delete</button></div></li>`;
  }).join("");
  $$("button", ul).forEach((b) => b.addEventListener("click", () => runAction(b.closest("li").dataset.id, b.dataset.act, b)));
}
async function saveRun(label, climate = null) {
  try {
    const r = await api("/api/runs", { json: { label, climate } });
    S.runs.push(r); drawRuns(); buildCmpOptions();
    toast(`Saved "${r.label}". Compare it on the hydrograph.`);
    return r;
  } catch (e) { toast(e.message, true); return null; }
}
async function runAction(id, act, btn) {
  const r = S.runs.find((x) => x.id === id);
  if (!r) return;
  if (act === "cmp") {
    S.cmpSel = S.cmpSel === `run:${id}` ? "" : `run:${id}`;
    buildCmpOptions(); drawRuns();
    if (S.cmpSel) setRtab("hydro");
    loadCompare();
  } else if (act === "use") {
    await withBusy(`Using "${r.label}"…`, async () => {
      const out = await api(`/api/runs/${id}/apply`, { json: {} });
      S.pendingSnap = false; snapshot(); S.pendingSnap = false;
      afterModelChange(out);
    });
  } else if (act === "ren") {
    const li = btn.closest("li"), lab = $(".rl", li);
    const inp = document.createElement("input");
    inp.type = "text"; inp.value = r.label; inp.maxLength = 80; inp.className = "rl-edit"; inp.setAttribute("aria-label", "New name");
    lab.replaceWith(inp); inp.focus(); inp.select();
    let done = false;
    const finish = async (keep) => {
      if (done) return; done = true;
      const v = inp.value.trim();
      if (keep && v && v !== r.label) {
        try { Object.assign(r, await api(`/api/runs/${id}`, { method: "PATCH", json: { label: v } })); } catch (e) { toast(e.message, true); }
      }
      drawRuns(); buildCmpOptions();
      if (S.cmp && S.cmp.run_id === id) loadCompare();
    };
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); });
    inp.addEventListener("blur", () => finish(true));
  } else if (act === "del") {
    if (btn.dataset.confirm !== "1") {          // second click confirms; no browser dialog
      btn.dataset.confirm = "1"; btn.textContent = "Delete?";
      setTimeout(() => { if (btn.isConnected) { btn.dataset.confirm = ""; btn.textContent = "Delete"; } }, 3000);
      return;
    }
    try {
      await api(`/api/runs/${id}`, { method: "DELETE" });
      S.runs = S.runs.filter((x) => x.id !== id);
      if (S.cmpSel === `run:${id}`) { S.cmpSel = ""; S.cmp = null; drawHydrograph(); }
      drawRuns(); buildCmpOptions();
    } catch (e) { toast(e.message, true); }
  }
}

// ------------------------------------------------------------------ results tabs
function setRtab(name) {
  if (!document.getElementById(`rtab-${name}`)) name = "hydro";
  S.rtab = name;
  const grp = groupOf(name);
  if (S.sub[grp] !== undefined) S.sub[grp] = name;
  $$(".rtabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.group === grp));
  const views = RGROUPS[grp];
  const sub = $("#subtabs");
  sub.hidden = views.length < 2;
  sub.innerHTML = views.length < 2 ? "" : views.map(([k, lab]) => `<button type="button" role="tab" data-rtab="${k}" aria-pressed="${k === name}" aria-selected="${k === name}">${lab}</button>`).join("");
  $$("button", sub).forEach((b) => b.addEventListener("click", () => setRtab(b.dataset.rtab)));
  $$(".rpanel").forEach((p) => { p.hidden = p.id !== `rtab-${name}`; });
  const ch = { hydro: hydroChart, season: seasonChart, cell: cellChart }[name];
  if (ch) requestAnimationFrame(() => ch.resize());
  if (name === "subbasins") drawSubbasins();
  if (name === "partition") drawPartition();
  if (name === "scenario") [scnSeasonChart, scnAnnualChart].forEach((ch) => ch && requestAnimationFrame(() => ch.resize()));
}

// ------------------------------------------------------------------ water partition
const PART = [
  // key, label, colour token, ring: inner groups / outer details
  { k: "et", label: "Evaporation (ET)", c: "--p-et", ring: "in" },
  { k: "et_land", label: "from soil and plants", c: "--p-et", ring: "out", parent: "et" },
  { k: "et_lake", label: "from open water (lakes)", c: "--p-lake", ring: "out", parent: "et" },
  { k: "d_soil", label: "Change in soil storage", c: "--p-soil", ring: "both" },
  { k: "quickflow", label: "Quickflow (direct runoff)", c: "--p-quick", ring: "both" },
  { k: "recharge", label: "Groundwater recharge", c: "--p-base", ring: "in" },
  { k: "baseflow", label: "becomes baseflow to rivers", c: "--p-base", ring: "out", parent: "recharge" },
  { k: "d_gw", label: "stays in groundwater storage", c: "--p-gw", ring: "out", parent: "recharge" },
];
const zoneLabel = (z) => z.id === null ? "Whole basin" : z.name.replace(" (incremental)", "");
const zoneColor = (z) => z.id === null ? css("--ink-2") : ZONE_COLORS[z.id % ZONE_COLORS.length];

// percentage labels drawn on the larger segments
const sliceLabels = {
  id: "sliceLabels",
  afterDatasetsDraw(chart, _a, opts) {
    if (!opts || !opts.on) return;
    const { ctx } = chart;
    const total = opts.total;
    chart.data.datasets.forEach((ds, di) => {
      chart.getDatasetMeta(di).data.forEach((arc, i) => {
        const v = ds.data[i];
        if (!v || v / total < 0.06) return;
        const p = arc.tooltipPosition();
        ctx.save();
        ctx.font = "600 11px Archivo, system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillStyle = "#fff"; ctx.fillText(`${Math.round(100 * v / total)}%`, p.x, p.y);
        ctx.restore();
      });
    });
  },
};
Chart.register(sliceLabels);

function donutOpts(total, labelsOf, unit) {
  return {
    responsive: true, maintainAspectRatio: true, aspectRatio: 1, cutout: "44%",
    plugins: {
      legend: { display: false },
      sliceLabels: { on: true, total },
      tooltip: { callbacks: {
        title: () => "",
        label: (c) => `${labelsOf(c)}: ${fmt(c.raw)} ${unit} (${fmt(100 * c.raw / total, 1)}%)`,
      } },
    },
  };
}

function partZoneOptions() {
  if (!S.zones) return;
  const sel = $("#partZone");
  const keep = sel.value || "total";
  const multi = S.zones.zones.length > 1;
  sel.innerHTML = `<option value="total">Whole basin</option>` +
    (multi ? S.zones.zones.map((z) => `<option value="${z.id}">${zoneLabel(z)}</option>`).join("") : "");
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : "total";
}

function drawPartition() {
  const t = $("#partTable");
  if (!S.zones) { t.innerHTML = "<tbody><tr><td>Run the model to see the water partition.</td></tr></tbody>"; return; }
  partZoneOptions();
  const zid = $("#partZone").value;
  const z = zid === "total" ? S.zones.total : S.zones.zones.find((x) => String(x.id) === zid) || S.zones.total;
  const P = z.precip, part = { ...z.partition, et: z.partition.et_land + z.partition.et_lake };
  const toMCM = (mm) => mm * z.area_km2 / 1000;
  const pos = (v) => Math.max(0, v);
  const inner = PART.filter((p) => p.ring !== "out" && (p.k !== "d_soil" || Math.abs(part.d_soil) > 0.05));
  const outer = PART.filter((p) => p.ring !== "in" && (p.k !== "d_soil" || Math.abs(part.d_soil) > 0.05) && (p.k !== "et_lake" || part.et_lake > 0.05));
  const surface = css("--surface");
  const ring = (items) => ({
    data: items.map((p) => pos(part[p.k])), backgroundColor: items.map((p) => css(p.c)),
    borderColor: surface, borderWidth: 2, hoverOffset: 4, $labels: items.map((p) => p.parent
      ? `${PART.find((q) => q.k === p.parent).label.replace(/ \(.*\)/, "")} ${p.label}` : p.label),
  });
  if (partChart) partChart.destroy();
  partChart = new Chart($("#partChart"), {
    type: "doughnut",
    data: { labels: outer.map((p) => p.label), datasets: [ring(outer), { ...ring(inner), weight: 1.1 }] },
    options: donutOpts(P, (c) => c.dataset.$labels[c.dataIndex], "mm/yr"),
  });
  $("#partCenter").innerHTML = `<b>${fmt(P)}</b><span>mm/yr rainfall</span><span>${fmt(toMCM(P) / 1000, 1)} km³/yr</span>`;
  $("#partTitle").textContent = `Where the rainfall goes: ${zoneLabel(z)}`;
  const row = (p) => {
    const v = part[p.k];
    const neg = v < -0.05 ? ` <span class="muted">(released from storage)</span>` : "";
    return `<tr class="${p.parent ? "sub" : ""}"><td><span class="sw" style="background:${css(p.c)}"></span>${p.label}${neg}</td>
      <td>${fmt(v, Math.abs(v) < 10 ? 1 : 0)}</td><td>${fmt(toMCM(v))}</td><td>${fmt(100 * v / P, 1)}%</td></tr>`;
  };
  const order = ["et", "et_land", "et_lake", "d_soil", "quickflow", "recharge", "baseflow", "d_gw"]
    .map((k) => PART.find((p) => p.k === k)).filter((p) => p.k !== "et_lake" || part.et_lake > 0.05);
  const flow = part.quickflow + part.baseflow;
  t.innerHTML = `<thead><tr><th>Part of rainfall</th><th>mm/yr</th><th>MCM/yr</th><th>Share</th></tr></thead><tbody>
    ${order.map(row).join("")}
    <tr class="tot"><td>Streamflow generated (quickflow + baseflow)</td><td>${fmt(flow)}</td><td>${fmt(toMCM(flow))}</td><td>${fmt(100 * flow / P, 1)}%</td></tr>
    </tbody>`;
  $("#partHint").textContent = `Inner ring: the first split of rainfall. Outer ring: the same water in more detail. Long-term means over ${S.meta.dates[0].slice(0, 4)}–${S.meta.dates.at(-1).slice(0, 4)}, area ${fmt(z.area_km2)} km². MCM = million m³.`;
  drawShare();
}

function drawShare() {
  const zones = S.zones.zones;
  $("#shareBlock").hidden = zones.length < 2;
  if (zones.length < 2) { if (shareChart) { shareChart.destroy(); shareChart = null; } return; }
  const k = $("#shareVar").value;
  const vol = zones.map((z) => Math.max(0, z[k]) * z.area_km2 / 1000);
  const total = vol.reduce((a, v) => a + v, 0);
  const sel = $("#partZone").value;
  if (shareChart) shareChart.destroy();
  shareChart = new Chart($("#shareChart"), {
    type: "doughnut",
    data: { labels: zones.map(zoneLabel), datasets: [{ data: vol, backgroundColor: zones.map(zoneColor),
      borderColor: css("--surface"), borderWidth: 2, hoverOffset: 4,
      offset: zones.map((z) => String(z.id) === sel ? 8 : 0) }] },
    options: { ...donutOpts(total, (c) => c.label, "MCM/yr"), cutout: "50%",
      onClick: (_e, els) => { if (els.length) { $("#partZone").value = String(zones[els[0].index].id); drawPartition(); } } },
  });
  const what = $("#shareVar").selectedOptions[0].textContent.toLowerCase();
  $("#shareCenter").innerHTML = `<b>${fmt(total)}</b><span>MCM/yr</span>`;
  $("#shareTable").innerHTML = `<thead><tr><th>Sub-basin</th><th>Area km²</th><th>MCM/yr</th><th>Share of ${what}</th></tr></thead><tbody>` +
    zones.map((z, i) => `<tr data-zone="${z.id}" class="${String(z.id) === sel ? "sel" : ""}"><td><span class="sw" style="background:${zoneColor(z)}"></span>${zoneLabel(z)}</td>
      <td>${fmt(z.area_km2)}</td><td>${fmt(vol[i])}</td><td>${fmt(100 * vol[i] / total, 1)}%</td></tr>`).join("") + "</tbody>";
  $$("#shareTable tr[data-zone]").forEach((tr) => tr.addEventListener("click", () => { $("#partZone").value = tr.dataset.zone; drawPartition(); }));
}

// ------------------------------------------------------------------ sub-basin table
async function loadSubbasins() {
  try { S.zones = await api("/api/zones/summary"); } catch (e) { S.zones = null; }
  if (S.rtab === "subbasins") drawSubbasins();
  if (S.rtab === "partition") drawPartition();
}
function drawSubbasins() {
  const t = $("#subTable");
  if (!S.zones) { t.innerHTML = "<tbody><tr><td>Run the model to see sub-basin results.</td></tr></tbody>"; return; }
  const per = S.res.periods;
  const metric = (z, k, d, suf = "") => {
    const v = z.gauge ? z.gauge.metrics.val[k] ?? z.gauge.metrics.all[k] : null;
    return v === null || v === undefined ? "–" : fmt(v, d) + suf;
  };
  const row = (z, tag = "td") => `
    <td><span class="sw" style="background:${z.id === null ? "transparent" : ZONE_COLORS[z.id % ZONE_COLORS.length]}"></span>${z.name.replace(" (incremental)", "")}</td>
    <${tag}>${fmt(z.area_km2)}</${tag}><${tag}>${fmt(z.precip)}</${tag}><${tag} class="opt">${fmt(z.et)}</${tag}>
    <${tag}>${fmt(z.runoff)}</${tag}><${tag} class="opt">${fmt(z.recharge)}</${tag}><${tag}>${fmt(100 * z.runoff_coef, 1)}%</${tag}>`;
  const head = `<thead><tr><th>Sub-basin</th><th>Area<small>km²</small></th><th>Rainfall<small>mm/yr</small></th><th class="opt">ET<small>mm/yr</small></th>
    <th>Runoff<small>mm/yr</small></th><th class="opt">Recharge<small>mm/yr</small></th><th>Runoff<small>coefficient</small></th>
    <th class="opt">Flow<small>obs / sim m³/s</small></th><th>KGE<small>validation</small></th><th class="opt">NSE<small>validation</small></th><th>Bias<small>validation</small></th></tr></thead>`;
  const body = S.zones.zones.map((z) => {
    const cls = fitClass(z.gauge ? (z.gauge.metrics.val.kge ?? z.gauge.metrics.all.kge) : null);
    const sel = String(z.id) === String(S.zone) ? " sel" : "";
    return `<tr tabindex="0" data-zone="${z.id}" class="${sel.trim()}">${row(z)}
      <td class="opt">${z.gauge ? `${fmt(z.gauge.q_obs)} / ${fmt(z.gauge.q_sim)}` : "–"}</td>
      <td>${z.gauge ? `<span class="q" style="background:${FIT_COLORS[cls]}"></span>` : ""}${metric(z, "kge", 2)}</td>
      <td class="opt">${metric(z, "nse", 2)}</td><td>${metric(z, "pbias", 0, "%")}</td></tr>`;
  }).join("");
  const foot = `<tfoot><tr>${row(S.zones.total)}<td class="opt"></td><td></td><td class="opt"></td><td></td></tr></tfoot>`;
  t.innerHTML = head + `<tbody>${body}</tbody>` + foot;
  t.closest(".rpanel").querySelector(".hint").textContent =
    `Long-term means for each gauge's own sub-catchment (the area between it and the gauges upstream). Fit scores are for the validation period ${per.val[0].slice(0, 4)}–${per.val[1].slice(0, 4)}. Click a row to zoom to it and edit its parameters.`;
  $$("tbody tr", t).forEach((tr) => {
    const go = () => focusZone(tr.dataset.zone);
    tr.addEventListener("click", go);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });
}
function focusZone(id) {
  const z = S.zones && S.zones.zones.find((zz) => String(zz.id) === String(id));
  if (!z) return;
  const b = z.bounds;
  map.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [20, 20] });
  S.zone = String(z.id);
  $("#zoneSelect").value = S.zone;
  buildSliders(); drawZoneHighlight(); drawSubbasins();
  $$(".tabs button").find((x) => x.dataset.tab === "params").click();
  if (z.gauge) selectGauge(z.gauge.index, false);
}

// ------------------------------------------------------------------ African basins (HydroBASINS)
const AFRICA = [[-35.5, -18.5], [37.6, 52]];
const LEVEL_TEXT = { 2: "Largest units: whole river systems and coastal regions.", 3: "Major basins and large tributaries.",
  4: "Sub-basins of the major rivers.", 5: "Smallest units here: tributary catchments." };

async function enterBrowse() {
  S.browse = true;
  if (overlay) overlay.setOpacity(0);
  outlineLayer.remove(); highlightLayer.remove(); gaugeLayer.remove();
  if (cellMarker) cellMarker.remove();
  $("#legend").hidden = true;
  let st;
  try { st = await api("/api/catalog"); } catch (e) { toast(e.message, true); return; }
  $("#catalogMissing").hidden = st.ready;
  $("#catalogReady").hidden = !st.ready;
  if (!st.files.dem || !st.files.lakes) {
    $("#catalogMissing .hint").textContent = "Run download_data.bat first: the Africa DEM and lakes file are needed. Then download the basin outlines here (one time; it resumes if the connection drops).";
  }
  map.fitBounds(AFRICA);
  if (st.ready) await showLevel(S.level);
  refreshPrepared();
  if (st.job && st.job.running) pollJob();
}

function leaveBrowse() {
  if (!S.browse) return;
  S.browse = false;
  if (S.catLayer) { S.catLayer.remove(); S.catLayer = null; }
  if (overlay) overlay.setOpacity(S.opacity);
  outlineLayer.addTo(map); highlightLayer.addTo(map); gaugeLayer.addTo(map);
  if (cellMarker) cellMarker.addTo(map);
  $("#legend").hidden = false;
  if (S.meta) fitBasin();
}

async function showLevel(lv) {
  S.level = lv;
  $("#levelOut").textContent = lv;
  if (!S.catData[lv]) {
    $("#levelHint").textContent = "Loading outlines…";
    try { S.catData[lv] = await api(`/api/catalog/level/${lv}`); } catch (e) { toast(e.message, true); return; }
    const kids = {};
    S.catData[lv].features.forEach((f) => { (kids[f.properties.next_down] ||= []).push(f.properties.id); });
    S.catData[lv].kids = kids;
  }
  const gj = S.catData[lv];
  $("#levelHint").textContent = `${gj.features.length} basins at level ${lv}. ${LEVEL_TEXT[lv]}`;
  if (S.catLayer) S.catLayer.remove();
  S.catLayer = L.geoJSON(gj, {
    style: () => baseStyle(),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      layer.bindTooltip(`PFAF ${p.pfaf}<br>${fmt(p.sub_area)} km² unit, ${fmt(p.up_area)} km² with upstream`, { sticky: true, className: "basin-tip" });
      layer.on("mouseover", () => { if (!isPicked(p.id)) layer.setStyle({ fillOpacity: 0.3, weight: 2 }); });
      layer.on("mouseout", () => { if (!isPicked(p.id)) layer.setStyle(baseStyle()); });
      layer.on("click", (e) => { L.DomEvent.stopPropagation(e); pickBasin(p); });
    },
  }).addTo(map);
  if (S.pick && S.pick.level === lv) styleSelection(); else { S.pick = null; $("#basinPick").innerHTML = '<p class="hint">Click a basin on the map.</p>'; }
}
const baseStyle = () => ({ color: css("--basin-line"), weight: 1.2, fillColor: css("--basin-line"), fillOpacity: 0.07 });

// First view with no basin loaded: the basin browser over Africa at level 2, map focus
function startEmpty() {
  document.body.classList.add("no-data");
  placeGutters();
  S.meta = null; S.res = null;
  setLayout("map");
  if (S.level !== 2) { S.level = 2; }
  $("#levelSlider").value = S.level; $("#levelOut").textContent = S.level;
  drawSteps();
  openStep("basins");
}
function upstreamIds(lv, id) {
  const kids = S.catData[lv].kids, out = [], stack = [id];
  while (stack.length) { const h = stack.pop(); out.push(h); (kids[h] || []).forEach((k) => stack.push(k)); }
  return out;
}
function isPicked(id) { return S.pick && S.pick.set && S.pick.set.has(id); }
function styleSelection() {
  if (!S.catLayer) return;
  S.catLayer.eachLayer((l) => {
    const id = l.feature.properties.id;
    if (!isPicked(id)) l.setStyle(baseStyle());
    else l.setStyle({ color: css("--ochre"), weight: id === S.pick.id ? 2.6 : 1.2, fillColor: css("--ochre"), fillOpacity: id === S.pick.id ? 0.45 : 0.28 });
  });
}

async function pickBasin(p) {
  const up = $("#upstreamChk").checked;
  const ids = up ? upstreamIds(S.level, p.id) : [p.id];
  S.pick = { id: p.id, level: S.level, set: new Set(ids), props: p };
  styleSelection();
  const box = $("#basinPick");
  box.innerHTML = '<p class="hint">Working out the download…</p>';
  let pl;
  try { pl = await api(`/api/basins/plan?level=${S.level}&id=${p.id}&upstream=${up}`); } catch (e) { box.innerHTML = `<p class="hint">${e.message}</p>`; return; }
  S.pick.plan = pl;
  const b = pl.bbox;
  if (S.catLayer) map.fitBounds([[b[1], b[0]], [b[3], b[2]]], { padding: [30, 30], maxZoom: 7 });
  const opt = (o) => `<label class="resopt"><input type="radio" name="resopt" value="${o.res}" ${o.res === pl.res ? "checked" : ""}>
      <span><b>${o.res}°</b> ${fmt(o.cells)} cells</span><span>${o.download_mb ? `${fmt(o.download_mb, o.download_mb < 10 ? 1 : 0)} MB` : "downloaded"}${o.res === pl.auto_res ? " <em>suggested</em>" : ""}</span></label>`;
  box.innerHTML = `<h4>PFAF ${pl.pfaf}, level ${pl.level}</h4>
    <dl>
      <dt>Catchment</dt><dd>${fmt(pl.area_km2)} km²${pl.units > 1 ? ` in ${pl.units} units` : ""}</dd>
      ${pl.endorheic ? "<dt>Note</dt><dd>Inland-draining basin: the model routes it to its lowest edge point.</dd>" : ""}
    </dl>
    <fieldset class="resopts"><legend>Model grid and download (rainfall and temperature, 1981–2023)</legend>
      ${pl.options.filter((o) => o.cells >= 30).map(opt).join("")}</fieldset>
    ${pl.fallback_mb ? `<p class="hint">Rainfall comes from the IRI Data Library. If it doesn't answer, the app switches to the Climate Hazards Center tiles: about ${fmt(pl.fallback_mb)} MB at this grid, kept for neighbouring basins.</p>` : ""}
    <button class="btn primary" id="prepareBtn">${pl.prepared ? "Open this basin" : "Prepare and open"}</button>`;
  $$('input[name="resopt"]', box).forEach((r) => r.addEventListener("change", async () => {
    const res = r.value;
    try {
      const p2 = await api(`/api/basins/plan?level=${pl.level}&id=${pl.id}&upstream=${$("#upstreamChk").checked}&res=${res}`);
      S.pick.plan = p2;
      $("#prepareBtn").textContent = p2.prepared ? "Open this basin" : "Prepare and open";
    } catch (e) { toast(e.message, true); }
  }));
  $("#prepareBtn").addEventListener("click", () => { const p2 = S.pick.plan; p2.prepared ? openDataset(p2.dataset) : prepareBasin(p2); });
}

async function prepareBasin(pl) {
  try {
    await api("/api/basins/prepare", { json: { level: pl.level, id: pl.id, upstream: $("#upstreamChk").checked, res: String(pl.res) } });
  } catch (e) { toast(e.message, true); return; }
  $("#prepareBtn").disabled = true;
  pollJob();
}
async function buildCatalog() {
  try { await api("/api/catalog/build", { json: {} }); } catch (e) { toast(e.message, true); return; }
  $("#catalogBuildBtn").disabled = true;
  pollJob();
}
async function pollJob() {
  clearTimeout(S.jobTimer);
  let j;
  try { j = await api("/api/jobs/current"); } catch (e) { return; }
  const box = $("#jobBox");
  box.hidden = false;
  const d = j.download;
  const STEP = { starting: "Starting", rainfall: "Downloading rainfall (CHIRPS)", temperature: "Downloading temperature (TerraClimate)",
    building: "Building the model grid", done: "Done", "downloading HydroBASINS": "Downloading basin outlines (HydroBASINS)" };
  const part = /^rainfall (\d+)\/(\d+)(?: \(([^)]+)\))?$/.exec(j.step || "");
  let msg = part ? `Downloading rainfall (CHIRPS, IRI Data Library), part ${part[1]} of ${part[2]}${part[3] ? `, ${part[3]}` : ""}`
    : j.step === "rainfall from CHC" ? "Downloading rainfall (CHIRPS tiles from the Climate Hazards Center)" : `${STEP[j.step] || j.step}`;
  $("#jobCancelBtn").hidden = !j.running;
  $("#jobLog").textContent = (j.log || []).filter((l) => !l.startsWith("GET ")).join("\n");
  if (d && d.waiting) {
    msg += part
      ? `: waiting for the server (${fmt(d.for_s)} s). If it doesn't answer, the app switches to the Climate Hazards Center tiles.`
      : `: waiting for the server (${fmt(d.for_s)} s).`;
    $(".bar", box).style.setProperty("--p", part ? `${Math.round(100 * (part[1] - 1) / part[2])}%` : "4%");
  } else if (d) {
    const months = /months (\d+)\/(\d+)/.exec(d.file);
    msg += months ? `: ${months[1]} of ${months[2]} months, ${fmt(d.mb, 0)} MB at ${fmt(d.kBps)} kB/s`
      : `: ${d.file} ${fmt(d.mb, 1)}${d.total_mb ? ` of ${fmt(d.total_mb, 1)}` : ""} MB at ${fmt(d.kBps)} kB/s`;
    const frac = months ? months[1] / months[2] : d.total_mb ? d.mb / d.total_mb : 0;
    $(".bar", box).style.setProperty("--p", `${Math.round(100 * frac)}%`);
    if (d.total_mb && d.kBps > 0) msg += `, about ${fmt(Math.max(0, (d.total_mb - d.mb) * 1000 / d.kBps / 60), 0)} min left`;
  } else {
    $(".bar", box).style.setProperty("--p", j.running ? "8%" : "100%");
  }
  if (j.log && j.log.length && !d) msg += `. ${j.log.filter((l) => !l.startsWith("GET ")).at(-1) || ""}`;
  $(".jobmsg", box).textContent = j.running ? msg : (j.error === "cancelled" ? "Cancelled. Finished parts are kept, so preparing again continues from there." : j.error ? `Stopped: ${j.error}` : "Finished.");
  if (j.running) { S.jobTimer = setTimeout(pollJob, 1500); return; }
  $("#catalogBuildBtn").disabled = false;
  if (j.error) { if (j.error !== "cancelled") toast(j.error, true, 8000); if ($("#prepareBtn")) $("#prepareBtn").disabled = false; return; }
  if (j.kind === "catalog") { box.hidden = true; S.catData = {}; enterBrowse(); return; }
  if (j.kind === "basin" && j.result) { box.hidden = true; await afterOpen(); }
}
async function openDataset(name) {
  await withBusy("Opening basin…", async () => {
    await api("/api/datasets/load", { json: { name } });
    await afterOpen();
  });
}
async function afterOpen() {
  $$(".tabs button").find((x) => x.dataset.tab === "params").click();   // leaves browse mode
  await loadAll();
  setRtab("hydro");
}
async function refreshPrepared() {
  let st;
  try { st = await api("/api/status"); } catch (e) { return; }
  const list = st.datasets.filter((d) => !d.startsWith("synthetic"));
  const label = (d) => d === "volta.npz" ? "Volta (example)" : d.replace(/^basin_L(\d)_(\d+)(_unit)?(?:_([\d.]+))?\.npz$/, (m, lv, id, unit, res) => `Level ${lv} basin ${id}${unit ? " (unit only)" : ""}${res ? `, ${res}°` : ""}`);
  $("#preparedList").innerHTML = list.length
    ? list.map((d) => `<button type="button" data-ds="${d}" aria-current="${d === st.dataset}">${label(d)}</button>`).join("")
    : '<p class="hint">None yet.</p>';
  $$("#preparedList button").forEach((b) => b.addEventListener("click", () => openDataset(b.dataset.ds)));
}

// ------------------------------------------------------------------ model choice
function buildModelPicker() {
  const seg = $("#modelSeg");
  seg.innerHTML = (S.meta.models || []).map((m) =>
    `<button type="button" data-model="${m.key}" aria-pressed="${m.key === S.meta.model}" title="${m.description}">${m.label}</button>`).join("");
  $("#modelHint").textContent = modelDef(S.meta.model).description;
  $$("button", seg).forEach((b) => b.addEventListener("click", () => { if (b.dataset.model !== S.meta.model) switchModel(b.dataset.model); }));
  buildCmpOptions();
}
async function switchModel(key) {
  await withBusy(`Switching to ${modelDef(key).label}…`, async () => {
    afterModelChange(await api("/api/model", { json: { model: key } }));
  });
}
// After the server changed model or parameters ({model, param_spec, summary})
function afterModelChange(r) {
  const switched = r.model !== S.meta.model;
  S.meta.model = r.model; S.meta.param_spec = r.param_spec;
  S.res = r.summary; S.params = structuredClone(r.summary.params); S.runId++;
  if (switched) { S.history = []; S.pendingSnap = false; }
  if (S.cmpSel === `model:${r.model}`) S.cmpSel = "";       // can't compare a model with itself
  buildModelPicker(); buildSliders(); zoneOptions(); buildCalibrationForm();
  if (hydroChart) { hydroChart.destroy(); hydroChart = null; }
  S.cmp = null;
  refreshResults();
  if (S.cmpSel) loadCompare();
}

// ------------------------------------------------------------------ parameters
function zoneOptions() {
  const sel = $("#zoneSelect");
  const mark = (z) => (S.params && S.params.zones[String(z.id)] && Object.keys(S.params.zones[String(z.id)]).length) ? " ●" : "";
  sel.innerHTML = `<option value="global">Whole basin (default for every zone)</option>` +
    S.meta.zones.map((z) => `<option value="${z.id}">${z.name.replace(" (incremental)", " sub-basin")}${mark(z)}</option>`).join("");
  sel.value = S.zone;
}
function effective(k) {
  if (S.zone === "global") return S.params.global[k];
  const z = S.params.zones[S.zone] || {};
  return z[k] ?? S.params.global[k];
}
function stepFor(k) {
  return { a: 0.001, b: 10, c: 0.01, d: 0.005, kc: 0.01, kw: 0.01, k_route: 0.1,
    umax: 0.5, lmax: 5, cqof: 0.01, ckif: 0.1, tof: 0.01, tif: 0.01, tg: 0.01, ckbf: 0.5 }[k] || 0.01;
}
function decFor(k) {
  return { a: 3, b: 0, c: 2, d: 3, kc: 2, kw: 2, k_route: 1, umax: 1, lmax: 0, cqof: 2, ckif: 1, tof: 2, tif: 2, tg: 2, ckbf: 1 }[k] ?? 2;
}
const modelDef = (key) => (S.meta.models || []).find((m) => m.key === key) || { key, label: key, description: "" };
const modelLabel = () => modelDef(S.meta.model).label;

function isChanged(k, isGlobalParam) {
  if (isGlobalParam) return Math.abs(S.params[k] - S.meta.global_spec[k].default) > 1e-9;
  if (S.zone === "global") return Math.abs(S.params.global[k] - S.meta.param_spec[k].default) > 1e-9;
  return (S.params.zones[S.zone] || {})[k] !== undefined;
}
function buildSliders() {
  const box = $("#sliders");
  box.innerHTML = "";
  Object.entries(S.meta.param_spec).forEach(([k, spec]) => box.appendChild(slider(k, spec, effective(k), (v) => {
    if (S.zone === "global") S.params.global[k] = v;
    else S.params.zones[S.zone] = { ...(S.params.zones[S.zone] || {}), [k]: v };
  }, () => isChanged(k, false))));
  const gbox = $("#globalSliders");
  gbox.innerHTML = "";
  Object.entries(S.meta.global_spec).forEach(([k, spec]) => gbox.appendChild(slider(k, spec, S.params[k], (v) => { S.params[k] = v; }, () => isChanged(k, true))));
  const z = S.meta.zones.find((zz) => String(zz.id) === String(S.zone));
  $("#zoneHint").textContent = S.zone === "global"
    ? "These values apply to every sub-basin that has no values of its own."
    : `${fmt(z.area_km2)} km², outlined in orange on the map. A dot marks a value set for this sub-basin; the others follow the whole-basin values.`;
  $("#resetZoneBtn").textContent = S.zone === "global" ? "Reset basin values" : "Reset this sub-basin";
  $("#undoBtn").disabled = !S.history.length;
}
function slider(k, spec, value, onSet, isChangedFn) {
  const ui = PARAM_UI[k] || { name: spec.label, tip: "" };
  const wrap = document.createElement("div");
  wrap.className = "slider";
  const id = `sl-${k}`, tipId = `tip-${k}`;
  wrap.innerHTML = `<div class="top">
      <span class="name"><span class="dot" title="Changed from default"></span><label for="${id}">${ui.name}</label><span class="sym">${k}</span>
        <button type="button" class="info" aria-expanded="false" aria-controls="${tipId}" aria-label="What does ${ui.name} do?">?</button></span>
      <output for="${id}"></output></div>
    <p class="tip" id="${tipId}" hidden>${ui.tip}</p>
    <input type="range" id="${id}" min="${spec.min}" max="${spec.max}" step="${stepFor(k)}" value="${value}">`;
  const input = $("input", wrap), out = $("output", wrap), info = $(".info", wrap), tip = $(".tip", wrap);
  const show = () => {
    out.innerHTML = `${fmt(+input.value, decFor(k))}${ui.unit ? `<span class="unit">${ui.unit}</span>` : ""}`;
    wrap.classList.toggle("changed", isChangedFn());
  };
  show();
  info.addEventListener("click", () => {
    const open = info.getAttribute("aria-expanded") !== "true";
    info.setAttribute("aria-expanded", open); tip.hidden = !open;
  });
  input.addEventListener("input", () => { snapshot(); onSet(+input.value); show(); });
  input.addEventListener("change", () => { zoneOptions(); if ($("#autoRun").checked) runModel(); });
  return wrap;
}

// ---- undo history --------------------------------------------------------------------
function snapshot() {
  if (S.pendingSnap) return;
  S.history.push(structuredClone(S.params));
  if (S.history.length > 30) S.history.shift();
  S.pendingSnap = true;
  $("#undoBtn").disabled = false;
}
function undo() {
  if (!S.history.length) return;
  S.params = S.history.pop();
  S.pendingSnap = false;
  zoneOptions(); buildSliders(); runModel(true);
}
function resetZone() {
  snapshot();
  if (S.zone === "global") S.params.global = Object.fromEntries(Object.entries(S.meta.param_spec).map(([k, v]) => [k, v.default]));
  else delete S.params.zones[S.zone];
  zoneOptions(); buildSliders(); runModel();
}
let resetArmed = 0;
function resetAll() {
  const btn = $("#resetAllBtn");
  if (!resetArmed) {
    btn.classList.add("armed"); btn.textContent = "Click again to reset all";
    resetArmed = setTimeout(() => { resetArmed = 0; btn.classList.remove("armed"); btn.textContent = "Reset all"; }, 3500);
    return;
  }
  clearTimeout(resetArmed); resetArmed = 0; btn.classList.remove("armed"); btn.textContent = "Reset all";
  snapshot();
  S.params.global = Object.fromEntries(Object.entries(S.meta.param_spec).map(([k, v]) => [k, v.default]));
  S.params.zones = {};
  Object.entries(S.meta.global_spec).forEach(([k, v]) => { S.params[k] = v.default; });
  zoneOptions(); buildSliders(); runModel();
  toast("All parameters reset to their defaults. Undo brings them back.");
}

let runSeq = 0;
async function runModel(fromUndo) {
  if (!fromUndo) S.pendingSnap = false;
  const seq = ++runSeq;
  document.body.classList.add("busy");
  $("#runBtn").disabled = true;
  try {
    const res = await api("/api/run", { json: { params: S.params } });
    if (seq !== runSeq) return;           // a newer run (or an undo) has started since
    S.res = res;
    S.params = structuredClone(S.res.params);
    S.runId++;
    refreshResults();
    buildSliders(); zoneOptions();
  } catch (e) { if (seq === runSeq) toast(e.message, true); }
  finally { if (seq === runSeq) { document.body.classList.remove("busy"); $("#runBtn").disabled = false; } }
}

function refreshResults() {
  scenarioStale(); drawSteps();
  drawBalance(); drawRibbon(); drawHydrograph(); drawSeason(); drawGauges(); loadSubbasins();
  if (!S.layer.startsWith("static:")) renderMap();
  if (S.cell) inspectCell({ lat: S.cell.lat, lng: S.cell.lon }, false);
}

// ------------------------------------------------------------------ climate scenarios
function scnLabels() {
  const r = +$("#scnRain").value, t = +$("#scnTemp").value, p = +$("#scnPet").value;
  $("#scnRainOut").textContent = `${r > 0 ? "+" : ""}${r} %`;
  $("#scnTempOut").textContent = `+${t.toFixed(1)} °C`;
  $("#scnPetOut").textContent = `${p > 0 ? "+" : ""}${p} %`;
}
async function runScenario() {
  const req = { rain_pct: +$("#scnRain").value, temp_c: +$("#scnTemp").value, pet_pct: +$("#scnPet").value };
  $("#scnRun").disabled = true;
  $("#scnSummary").classList.remove("stale");
  $("#scnSummary").textContent = "Running the scenario…";
  try {
    S.scn = await api("/api/scenario", { json: req });
    S.scnRun = S.runId;
    drawScenario();
  } catch (e) { $("#scnSummary").textContent = `Scenario failed: ${e.message}`; }
  finally { $("#scnRun").disabled = false; }
}
function signed(v, d = 0, unit = " %") { return v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${fmt(v, d)}${unit}`; }
function drawScenario() {
  const r = S.scn;
  if (!r) return;
  const b = r.baseline, x = r.scenario, c = r.change_pct, q = r.request;
  const what = [q.rain_pct ? `${q.rain_pct > 0 ? q.rain_pct + " % more" : -q.rain_pct + " % less"} rain` : null,
    q.temp_c ? `${fmt(q.temp_c, 1)} °C warmer` : null, q.pet_pct ? `PET ${signed(q.pet_pct)}` : null].filter(Boolean).join(", ") || "no change";
  const dir = c.Q < 0 ? "falls" : "rises";
  $("#scnSummary").textContent = `With ${what}, river flow at the outlet ${dir} by ${fmt(Math.abs(c.Q), 0)} % ` +
    `(${fmt(b.outlet_mean_m3s)} → ${fmt(x.outlet_mean_m3s)} m³/s, ${fmt(b.outlet_mcm_per_year)} → ${fmt(x.outlet_mcm_per_year)} million m³ a year). ` +
    `Dry-season flow ${c.dry_season < 0 ? "falls" : "rises"} by ${fmt(Math.abs(c.dry_season), 0)} %.` +
    (r.elasticity !== null ? ` Each 1 % change in rainfall changes flow by about ${fmt(r.elasticity, 1)} %.` : "") +
    ` Model: ${modelDef(r.model).label}.`;
  const cls = () => "";        // changes are shown with their sign only: more evaporation is not "good" in green
  const row = (lab, bv, xv, cv, d = 0, u = "") => `<tr><td>${lab}</td><td>${fmt(bv, d)}${u}</td><td>${fmt(xv, d)}${u}</td><td class="${cls(cv)}">${signed(cv)}</td></tr>`;
  $("#scnTable").innerHTML = `<thead><tr><th>Basin, per year</th><th>Now</th><th>Scenario</th><th>Change</th></tr></thead><tbody>
    ${row("Rainfall, mm", b.per_year.P, x.per_year.P, c.P)}
    ${row("Potential evaporation, mm", b.per_year.PET, x.per_year.PET, c.PET)}
    ${row("Actual evaporation, mm", b.per_year.ET, x.per_year.ET, c.ET)}
    ${row("Runoff, mm", b.per_year.Q, x.per_year.Q, c.Q)}
    ${row("Outlet flow, m³/s", b.outlet_mean_m3s, x.outlet_mean_m3s, c.Q)}
    ${row("Outflow volume, MCM", b.outlet_mcm_per_year, x.outlet_mcm_per_year, c.Q)}
    ${row("Wettest month, m³/s", b.wettest_month_m3s, x.wettest_month_m3s, c.wettest_month)}
    ${row("Dry season (3 lowest months), m³/s", b.dry_season_m3s, x.dry_season_m3s, c.dry_season)}
    <tr><td>Runoff ratio</td><td>${fmt(100 * b.runoff_ratio, 1)} %</td><td>${fmt(100 * x.runoff_ratio, 1)} %</td><td></td></tr></tbody>`;
  $("#scnGauges").innerHTML = b.gauges.length ? `<thead><tr><th>Gauge</th><th>Now, m³/s</th><th>Scenario</th><th>Change</th></tr></thead><tbody>` +
    b.gauges.map((g, i) => { const v = x.gauges[i].mean_m3s, ch = 100 * (v / g.mean_m3s - 1);
      return `<tr><td>${g.name}</td><td>${fmt(g.mean_m3s)}</td><td>${fmt(v)}</td><td class="${cls(ch)}">${signed(ch)}</td></tr>`; }).join("") + "</tbody>"
    : "<tbody><tr><td class=\"muted\">This basin has no gauges.</td></tr></tbody>";
  const lines = (labels, a, z, ytitle) => ({
    type: "line",
    data: { labels, datasets: [
      { label: "Now", data: a, borderColor: css("--water"), borderWidth: 2, pointRadius: 0 },
      { label: "Scenario", data: z, borderColor: css("--p-base"), borderDash: [6, 3], borderWidth: 2, pointRadius: 0 }] },
    options: baseOpts(ytitle),
  });
  if (scnSeasonChart) scnSeasonChart.destroy();
  if (scnAnnualChart) scnAnnualChart.destroy();
  $("#scnOut").hidden = false; $("#scnExport").hidden = false; $("#scnSave").hidden = false;
  scnSeasonChart = new Chart($("#scnSeason"), lines(MONTHS, b.seasonal_m3s, x.seasonal_m3s, "m³/s"));
  scnAnnualChart = new Chart($("#scnAnnual"), lines(b.years, b.annual_mcm, x.annual_mcm, "million m³"));
}
function scenarioStale() {
  if (S.scn && S.scnRun !== S.runId) {
    $("#scnSummary").classList.add("stale");
    $("#scnSummary").textContent = "The model or parameters changed since this scenario ran. Run it again to compare with the current run.";
  }
}

// ------------------------------------------------------------------ open gauge records (ADHI)
let adhiLayer = null;
async function findAdhi() {
  const btn = $("#adhiBtn"), status = $("#adhiStatus");
  btn.disabled = true;
  try {
    let r = await fetch("/api/adhi/candidates");
    if (r.status === 409) {
      status.textContent = "Downloading the ADHI station list and monthly series (about 7 MB)…";
      await api("/api/adhi/fetch", { method: "POST" });
      for (;;) {
        await new Promise((ok) => setTimeout(ok, 1500));
        const j = await api("/api/jobs/current");
        const d = j.download;
        if (d && !d.waiting) status.textContent = `Downloading ${d.file}: ${fmt(d.mb, 1)}${d.total_mb ? ` of ${fmt(d.total_mb, 1)}` : ""} MB at ${fmt(d.kBps)} kB/s`;
        if (!j.running) { if (j.error) throw new Error(j.error); break; }
      }
      r = await fetch("/api/adhi/candidates");
    }
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    drawAdhi(await r.json());
  } catch (e) { status.textContent = `Could not get ADHI records: ${e.message}`; }
  finally { btn.disabled = false; }
}
function drawAdhi(res) {
  S.adhi = res;
  const list = $("#adhiList"), status = $("#adhiStatus");
  const c = res.candidates;
  if (adhiLayer) { adhiLayer.remove(); adhiLayer = null; }
  if (!c.length) { list.innerHTML = ""; status.textContent = "No ADHI station in this basin has records in the model period."; return; }
  const nOn = c.filter((x) => x.suggest !== "skip").length;
  status.textContent = `${c.length} station${c.length > 1 ? "s" : ""} in this basin with records between ${S.meta.dates[0].slice(0, 4)} and ${S.meta.dates.at(-1).slice(0, 4)}; ${nOn} selected by default (long records on larger rivers, no duplicates or reservoirs). Each new gauge becomes a calibration zone, so fewer, well-measured gauges calibrate better.`;
  const opt = (x) => {
    const o = [`<option value="skip">Skip</option>`, `<option value="new">Add as a new gauge</option>`];
    const ms = x.could_match || (x.match ? [x.match] : []);
    ms.forEach((m) => o.push(`<option value="${m.id}">Use as records for ${m.name}</option>`));
    return o.join("");
  };
  list.innerHTML = c.map((x, i) => `<div class="adhi-item" data-i="${i}">
      <div class="t"><b>${x.name}${x.already ? ` <span class="muted">(already added)</span>` : ""}</b><span class="muted">${x.river}</span></div>
      <div class="m">${fmt(x.area_model_km2)} km² in the model (ADHI ${fmt(x.area_km2)}), ${x.months} months ${x.first}–${x.last},
        mean ${fmt(x.mean_m3s)} m³/s${x.dams ? `, ${x.dams} dam${x.dams > 1 ? "s" : ""} upstream` : ""}${x.comment ? ` · ${x.comment}` : ""}</div>
      ${x.notes && x.notes.length ? `<div class="m ${x.reservoir || x.flag >= 2 ? "warn" : ""}">${x.suggest === "skip" ? "Skipped by default: " : ""}${x.notes.join("; ")}</div>` : ""}
      <select data-adhi="${x.adhi}" aria-label="What to do with ${x.name}">${opt(x)}</select></div>`).join("") +
    `<div class="adhi-actions"><button class="btn primary" id="adhiAddBtn" type="button">Add selected records</button>
      <span class="muted">Data: ADHI, CC BY 4.0</span></div>`;
  $$("select[data-adhi]", list).forEach((sel, i) => {
    sel.value = c[i].suggest;
    const mark = () => sel.closest(".adhi-item").classList.toggle("on", sel.value !== "skip");
    sel.addEventListener("change", mark); mark();
  });
  adhiLayer = L.layerGroup(c.map((x) => L.circleMarker([x.lat_model, x.lon_model], {
    radius: 6, color: css("--ochre"), weight: 2, fillColor: css("--surface"), fillOpacity: 1, pane: "markerPane",
  }).bindTooltip(`${x.name} (ADHI): ${x.months} months`, { direction: "top" }))).addTo(map);
  $("#adhiAddBtn").addEventListener("click", addAdhi);
}
async function addAdhi() {
  const choices = Object.fromEntries($$("select[data-adhi]").map((s) => [s.dataset.adhi, s.value]).filter(([, v]) => v !== "skip"));
  if (!Object.keys(choices).length) { toast("Choose at least one station to add.", true); return; }
  await withBusy("Adding gauge records and rerunning the model…", async () => {
    const r = await api("/api/adhi/add", { json: { choices } });
    if (adhiLayer) { adhiLayer.remove(); adhiLayer = null; }
    $("#adhiList").innerHTML = "";
    await loadAll();
    $("#adhiStatus").textContent = r.message;
    setRtab("hydro");
  });
}

// ------------------------------------------------------------------ calibration
function buildCalibrationForm() {
  const dflt = modelDef(S.meta.model).calibrate_default || Object.keys(S.meta.param_spec).filter((k) => k !== "kc");
  $("#calNames").innerHTML = Object.keys(S.meta.param_spec).map((k) =>
    `<label title="${(PARAM_UI[k] || {}).name || k}"><input type="checkbox" value="${k}" ${dflt.includes(k) ? "checked" : ""}> ${k}</label>`).join("");
  $("#calGauges").innerHTML = S.meta.gauges.map((g) =>
    `<label><input type="checkbox" value="${g.id}" checked> ${g.name}</label>`).join("");
  $("#calStart").value = S.meta.periods.cal[0]; $("#calEnd").value = S.meta.periods.cal[1];
  $("#calNoGauge").hidden = S.meta.has_gauges;
  $("#calStartBtn").disabled = !S.meta.has_gauges;
}
async function startCalibration() {
  const names = $$("#calNames input:checked").map((i) => i.value);
  const gauges = $$("#calGauges input:checked").map((i) => +i.value);
  if (!names.length || !gauges.length) { toast("Pick at least one parameter and one gauge.", true); return; }
  try {
    await api("/api/calibrate", { json: {
      names, gauges, objective: $("#calObjective").value, cal_start: $("#calStart").value, cal_end: $("#calEnd").value,
      maxiter: +$("#calIter").value, popsize: +$("#calPop").value } });
    $("#calStartBtn").disabled = true; $("#calCancelBtn").disabled = false; $("#calApplyBtn").hidden = true;
    $("#calProgress").hidden = false; $("#calLog").innerHTML = "";
    pollCalibration();
  } catch (e) { toast(e.message, true); }
}
async function pollCalibration() {
  clearTimeout(S.calTimer);
  let st;
  try { st = await api("/api/calibrate/status"); } catch (e) { toast(e.message, true); return; }
  if (!!st.running !== !!S.calRunning) { S.calRunning = !!st.running; drawSteps(); }
  const p = st.progress;
  if (p) {
    const frac = (p.zone_index + (p.evals && p.evals_est ? Math.min(1, p.evals / p.evals_est) : 0)) / p.zones_total;
    $("#calProgress .bar").style.setProperty("--p", `${Math.min(100, 100 * frac)}%`);
    $("#calProgress p").textContent = st.running
      ? `${p.zone}: ${p.evals ? `${p.evals} runs, best score ${fmt(p.best, 3)}` : "starting"} (${fmt(st.elapsed)} s)`
      : `Finished in ${fmt(st.elapsed)} s`;
  }
  if (st.log && st.log.length) {
    $("#calLog").innerHTML = `<table><thead><tr><th>Zone</th><th>Score</th>${Object.keys(st.log[0].params).map((k) => `<th>${k}</th>`).join("")}<th>s</th></tr></thead><tbody>${
      st.log.map((r) => `<tr><td>${r.gauge}</td><td>${fmt(r.score, 3)}</td>${Object.entries(r.params).map(([k, v]) => `<td>${fmt(v, decFor(k))}</td>`).join("")}<td>${fmt(r.seconds)}</td></tr>`).join("")}</tbody></table>`;
  }
  if (st.running) { S.calTimer = setTimeout(pollCalibration, 1500); return; }
  $("#calStartBtn").disabled = false; $("#calCancelBtn").disabled = true;
  if (st.error) { toast(st.error === "cancelled" ? "Calibration cancelled." : `Calibration failed: ${st.error}`, st.error !== "cancelled"); return; }
  if (st.result) { $("#calApplyBtn").hidden = false; toast("Calibration finished. Review the scores, then use the parameters."); }
}
async function applyCalibration() {
  S.pendingSnap = false; snapshot(); S.pendingSnap = false;
  try {
    S.res = await api("/api/calibrate/apply", { json: {} });
    S.params = structuredClone(S.res.params); S.runId++;
    S.calAt = new Date().toTimeString().slice(0, 5);
    zoneOptions(); buildSliders(); refreshResults();
    toast("Calibrated parameters applied.");
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ datasets
async function refreshDatasets() {
  const st = await api("/api/status");
  $("#datasetSelect").innerHTML = st.datasets.map((d) => `<option ${d === st.dataset ? "selected" : ""}>${d}</option>`).join("") || "<option disabled>No datasets yet</option>";
  return st;
}
async function withBusy(label, fn) {
  toast(label, false, 60000);
  document.body.classList.add("busy");
  try { await fn(); toast("Done."); } catch (e) { toast(e.message, true); }
  finally { document.body.classList.remove("busy"); }
}
async function loadAll() {
  const st = await refreshDatasets();
  if (!st.loaded) { startEmpty(); return; }
  if (document.body.classList.contains("no-data")) { document.body.classList.remove("no-data"); applySplit(); relayout(); }
  S.meta = await api("/api/meta");
  S.res = await api("/api/results");
  S.params = structuredClone(S.res.params);
  S.t = Math.min(S.t, S.meta.dates.length - 1);
  S.zone = "global"; S.cell = null; S.runId++; mapCache.clear();
  S.gauge = Math.max(0, S.res.gauges.findIndex((g) => g.metrics.all.n > 0));   // open on a gauge with records
  S.history = []; S.pendingSnap = false; S.zones = null;
  const [zone, lake, dem] = await Promise.all(["zone", "lake", "dem"].map((v) => api(`/api/static/${v}`)));
  S.statics = { zone, lake, dem };
  if (cellMarker) { cellMarker.remove(); cellMarker = null; }
  $("#cellHint").hidden = false; $("#cellMeta").innerHTML = ""; $("#cellTitle").textContent = "Cell";
  if (cellChart) { cellChart.destroy(); cellChart = null; }
  $("#synthBadge").hidden = !S.meta.synthetic;
  $("#subtitle").textContent = `${S.meta.name}: ${fmt(S.meta.basin_area_km2)} km², ${fmt(S.meta.n_cells)} cells at ${S.meta.grid.res}°, ${S.meta.dates[0]} to ${S.meta.dates.at(-1)}`;
  $("#gaugeSelect").innerHTML = S.meta.gauges.map((g, i) => `<option value="${i}">${g.name}${g.river ? ` (${g.river})` : ""}</option>`).join("");
  $("#gaugeSelect").value = String(S.gauge);
  $("#obsGauge").innerHTML = `<option value="">Match by column or station name</option>` +
    S.meta.gauges.map((g) => `<option value="${g.id}">${g.name}</option>`).join("");
  $("#obsInput").disabled = !S.meta.gauges.length;
  $("#obsStatus").textContent = S.meta.gauges.length ? "" : "This basin has no gauges defined, so observed discharge can't be attached yet.";
  S.cmp = null; S.adhi = null; S.scn = null;
  $("#scnOut").hidden = true; $("#scnExport").hidden = true; $("#scnSave").hidden = true; S.calAt = null; $("#scnSummary").textContent = "";
  if (adhiLayer) { adhiLayer.remove(); adhiLayer = null; }
  $("#adhiList").innerHTML = ""; $("#adhiStatus").textContent = "";
  $("#adhiBtn").disabled = !!S.meta.synthetic;
  if (S.meta.synthetic) $("#adhiStatus").textContent = "The demo basin uses synthetic gauge records. Open a real basin to add ADHI records.";
  buildModelPicker(); zoneOptions(); buildSliders(); buildCalibrationForm();
  const md = S.meta.meta;
  $("#metaList").innerHTML = Object.entries({ Name: S.meta.name, Rainfall: md.precip, Temperature: md.temperature,
    PET: md.pet, DEM: md.dem, Note: md.note, Created: md.created }).filter(([, v]) => v)
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  fitBasin(); drawOutlines(); drawZoneHighlight();
  if (hydroChart) { hydroChart.destroy(); hydroChart = null; }
  refreshResults();
  renderMap();
  await loadRuns();
  if (S.cmpSel) loadCompare();
}

// ------------------------------------------------------------------ wiring
function wire() {
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
    $$(".tabpanel").forEach((p) => { p.hidden = p.id !== `tab-${b.dataset.tab}`; });
    if (b.dataset.tab === "basins") enterBrowse(); else leaveBrowse();
  }));
  $("#levelSlider").addEventListener("input", (e) => { $("#levelOut").textContent = e.target.value; });
  $("#levelSlider").addEventListener("change", (e) => showLevel(+e.target.value));
  $("#upstreamChk").addEventListener("change", () => { if (S.pick && S.pick.props) pickBasin(S.pick.props); });
  $("#catalogBuildBtn").addEventListener("click", buildCatalog);
  $("#jobCancelBtn").addEventListener("click", () => api("/api/jobs/cancel", { json: {} }).then(() => toast("Stopping after the current step…")));
  $("#zoneSelect").addEventListener("change", (e) => {
    S.zone = e.target.value; buildSliders(); drawZoneHighlight();
    if (S.rtab === "subbasins") drawSubbasins();
    const g = S.meta.gauges.findIndex((gg) => String(gg.id) === S.zone);
    if (g >= 0) selectGauge(g, false);
  });
  $("#resetZoneBtn").addEventListener("click", resetZone);
  $("#resetAllBtn").addEventListener("click", resetAll);
  $("#undoBtn").addEventListener("click", undo);
  $("#runBtn").addEventListener("click", () => runModel());
  $$(".rtabs button").forEach((b) => b.addEventListener("click", () => {
    const g = b.dataset.group; setRtab(S.sub[g] || RGROUPS[g][0][0]);
  }));
  $("#runSaveForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (await saveRun($("#runName").value)) $("#runName").value = "";
  });
  $("#scnSave").addEventListener("click", async (e) => {
    if (!S.scn) return;
    const btn = e.currentTarget, n = S.runs.filter((r) => r.climate).length + 1;
    btn.disabled = true; btn.textContent = "Saving…";      // the run is recomputed under the changed climate
    try { await saveRun(`Climate scenario ${n}`, S.scn.request); } finally { btn.disabled = false; btn.textContent = "Save as run"; }
  });
  try { const o = localStorage.getItem("volta-opacity"); if (o) { S.opacity = +o; $("#opacity").value = Math.round(S.opacity * 100); } } catch (_) { /* ignore */ }
  $("#opacity").addEventListener("input", (e) => {
    S.opacity = +e.target.value / 100;
    if (overlay) overlay.setOpacity(S.opacity);
    try { localStorage.setItem("volta-opacity", S.opacity); } catch (_) { /* ignore */ }
  });
  $("#layerSelect").addEventListener("change", (e) => {
    S.layer = e.target.value;
    const st = S.layer.startsWith("static:");
    $$("#modeSeg button").forEach((b) => { b.disabled = st; });
    renderMap();
  });
  $$("#modeSeg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#climMonth").innerHTML = MONTHS.map((m, i) => `<option value="${i + 1}" ${i + 1 === S.month ? "selected" : ""}>${m}</option>`).join("");
  $("#climMonth").addEventListener("change", (e) => { S.month = +e.target.value; renderMap(); });
  $("#gaugeSelect").addEventListener("change", (e) => selectGauge(+e.target.value, false));
  $("#cmpSelect").addEventListener("change", (e) => { S.cmpSel = e.target.value; drawRuns(); loadCompare(); });
  $("#adhiBtn").addEventListener("click", findAdhi);
  ["scnRain", "scnTemp", "scnPet"].forEach((id) => $("#" + id).addEventListener("input", scnLabels));
  scnLabels();
  $("#scnRun").addEventListener("click", runScenario);
  $$("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    const [r, t] = b.dataset.preset.split(",").map(Number);
    $("#scnRain").value = r; $("#scnTemp").value = t; $("#scnPet").value = 0; scnLabels(); runScenario();
  }));
  $$("#unitSeg button").forEach((b) => b.addEventListener("click", () => {
    S.unit = b.dataset.unit;
    $$("#unitSeg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    drawHydrograph();
  }));
  $("#obsInput").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return;
    e.target.value = "";
    withBusy(`Reading ${f.name}…`, async () => {
      const q = new URLSearchParams({ name: f.name });
      if ($("#obsGauge").value) q.set("gauge", $("#obsGauge").value);
      const r = await api(`/api/observed?${q}`, { method: "POST", body: f });
      $("#obsStatus").textContent = r.message;
      S.res = r.summary; S.runId++;
      refreshResults(); setRtab("hydro");
    });
  });
  $$("#rangeSeg button").forEach((b) => b.addEventListener("click", () => {
    S.range = b.dataset.range;
    $$("#rangeSeg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    drawHydrograph(false);
  }));
  const rib = $("#ribbon");
  let dragging = false;
  rib.addEventListener("pointerdown", (e) => { dragging = true; rib.setPointerCapture(e.pointerId); ribbonPick(e); });
  rib.addEventListener("pointermove", (e) => { if (dragging) ribbonPick(e); });
  rib.addEventListener("pointerup", () => { dragging = false; });
  $("#playBtn").addEventListener("click", () => {
    if (S.playing) { clearInterval(S.playing); S.playing = null; $("#playBtn").textContent = "▶"; return; }
    if (S.mode !== "month" && S.mode !== "anomaly") setMode("month", false);
    $("#playBtn").textContent = "❚❚";
    S.playing = setInterval(() => {
      S.t = (S.t + 1) % S.meta.dates.length; drawRibbon(); renderMap(); drawHydrograph(false);
    }, 280);
  });
  window.addEventListener("resize", debounce(() => { drawRibbon(); map.invalidateSize(); }, 100));
  new ResizeObserver(debounce(() => map.invalidateSize(), 80)).observe($(".mapwrap"));
  $("#calStartBtn").addEventListener("click", startCalibration);
  $("#calCancelBtn").addEventListener("click", () => api("/api/calibrate/cancel", { json: {} }));
  $("#calApplyBtn").addEventListener("click", applyCalibration);
  $("#loadBtn").addEventListener("click", () => withBusy("Loading dataset…", async () => {
    await api("/api/datasets/load", { json: { name: $("#datasetSelect").value } }); await loadAll();
  }));
  $$("[data-synth]").forEach((b) => b.addEventListener("click", () => withBusy("Building the demo basin. This takes about 20 seconds…", async () => {
    await api("/api/datasets/synthetic", { json: { res: +b.dataset.synth } }); await loadAll();
  })));
  $("#uploadInput").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return;
    withBusy(`Uploading ${f.name}…`, async () => {
      await api(`/api/datasets/upload?name=${encodeURIComponent(f.name)}`, { method: "POST", body: f }); await loadAll();
    });
  });
}

// ------------------------------------------------------------------ view: layout and theme
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* storage blocked */ } },
};
function relayout() {
  requestAnimationFrame(() => {
    placeGutters();
    if (map) map.invalidateSize();
    [hydroChart, seasonChart, cellChart, partChart, shareChart].forEach((c) => c && c.resize());
  });
}
function setLayout(name) {
  if (!["data", "map", "charts"].includes(name)) name = "map";
  document.documentElement.dataset.layout = name;
  applySplit();
  $$("#layoutPick button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.layout === name));
  store.set("volta-layout", name);
  relayout();
}
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
const isDark = () => document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : darkQuery.matches;
function setTheme(choice, save = true) {
  document.documentElement.dataset.theme = choice;
  if (save) store.set("volta-theme", choice);
  themeChanged();
}
function themeButton() {
  const next = isDark() ? "light" : "dark";
  const label = next === "dark" ? "Dark" : "Light";
  $("#themeLabel").textContent = label;
  $("#themeToggle").title = $("#themeToggle").ariaLabel = `Switch to ${next} theme`;
}
function themeChanged() {
  themeButton();
  // grey canvas basemaps follow the theme; topographic and satellite are left alone
  const dark = isDark();
  const from = dark ? "Esri light grey" : "Esri dark grey", to = dark ? "Esri dark grey" : "Esri light grey";
  if (map && map.hasLayer(baseLayers[from])) { map.removeLayer(baseLayers[from]); baseLayers[to].addTo(map); }
  if (!S.res) return;
  // charts read their colours when created, so draw them again
  if (hydroChart) { hydroChart.destroy(); hydroChart = null; }
  drawHydrograph(); drawSeason(); drawCell();
  if (S.rtab === "partition") drawPartition();
  if (S.scn && S.rtab === "scenario") drawScenario();
  drawBalance(); drawRibbon(); drawGauges();
  if (S.layer && !S.layer.startsWith("static:")) renderMap();
}
// ---- draggable dividers between the three panels; widths are remembered per layout
const SPLIT_KEY = "cbhm-split";
const isNarrow = () => window.innerWidth < 980;
function splitGet() { try { return JSON.parse(localStorage.getItem(SPLIT_KEY)) || {}; } catch (_) { return {}; } }
function splitSet(v) { try { localStorage.setItem(SPLIT_KEY, JSON.stringify(v)); } catch (_) { /* storage blocked */ } }
function applySplit() {
  const lay = $(".layout"); if (!lay) return;
  const w = splitGet()[document.documentElement.dataset.layout];
  lay.style.gridTemplateColumns = (!isNarrow() && w) ? `${w.c}px minmax(0, 1fr) ${w.r}px` : "";
  placeGutters();
}
function placeGutters() {
  const lay = $(".layout"), gl = $("#gutL"), gr = $("#gutR");
  if (!lay || !gl) return;
  const L0 = lay.getBoundingClientRect().left;
  gl.hidden = isNarrow();
  gr.hidden = isNarrow() || document.body.classList.contains("no-data");
  if (!gl.hidden) gl.style.left = `${$(".controls").getBoundingClientRect().right - L0 - 3}px`;
  if (!gr.hidden) gr.style.left = `${$(".results").getBoundingClientRect().left - L0 - 3}px`;
}
function setSplit(c, r, save) {
  const lay = $(".layout"), W = lay.getBoundingClientRect().width, MAPMIN = 320;
  const noRes = document.body.classList.contains("no-data");
  c = Math.round(Math.max(200, Math.min(c, 640, W - (noRes ? 0 : r) - MAPMIN)));
  r = Math.round(Math.max(280, Math.min(r, 1100, W - c - MAPMIN)));
  lay.style.gridTemplateColumns = `${c}px minmax(0, 1fr) ${r}px`;
  placeGutters();
  if (map) map.invalidateSize({ pan: false });
  if (save) { const all = splitGet(); all[document.documentElement.dataset.layout] = { c, r }; splitSet(all); relayout(); }
}
function currentSplit() {
  const c = $(".controls").getBoundingClientRect().width;
  const cols = getComputedStyle($(".layout")).gridTemplateColumns.split(" ").map(parseFloat);
  return { c, r: cols.length === 3 ? cols[2] : 400 };
}
function initSplitters() {
  const lay = $(".layout");
  [["#gutL", "c"], ["#gutR", "r"]].forEach(([sel, side]) => {
    const g = $(sel);
    g.addEventListener("pointerdown", (e) => {
      if (isNarrow()) return;
      e.preventDefault(); g.setPointerCapture(e.pointerId);
      document.body.classList.add("resizing"); g.classList.add("active");
      const rect = lay.getBoundingClientRect(), start = currentSplit();
      let raf = 0, last = null;
      const move = (ev) => {
        last = side === "c" ? { c: ev.clientX - rect.left, r: start.r } : { c: start.c, r: rect.right - ev.clientX };
        if (!raf) raf = requestAnimationFrame(() => { raf = 0; setSplit(last.c, last.r, false); });
      };
      const up = () => {
        g.removeEventListener("pointermove", move); g.removeEventListener("pointerup", up); g.removeEventListener("pointercancel", up);
        document.body.classList.remove("resizing"); g.classList.remove("active");
        if (raf) { cancelAnimationFrame(raf); raf = 0; }
        const s = last || currentSplit(); setSplit(s.c, s.r, true);
      };
      g.addEventListener("pointermove", move); g.addEventListener("pointerup", up); g.addEventListener("pointercancel", up);
    });
    g.addEventListener("dblclick", () => {
      const all = splitGet(); delete all[document.documentElement.dataset.layout]; splitSet(all);
      applySplit(); relayout();
    });
    g.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const s = currentSplit(), d = (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 60 : 20);
      if (side === "c") s.c += d; else s.r -= d;
      setSplit(s.c, s.r, true);
    });
  });
  window.addEventListener("resize", () => { applySplit(); });
  if (window.ResizeObserver) new ResizeObserver(() => placeGutters()).observe($(".controls"));
}

function initView() {
  initSplitters();
  setLayout(store.get("volta-layout") || "map");
  $$("#layoutPick button").forEach((b) => b.addEventListener("click", () => setLayout(b.dataset.layout)));
  $("#themeToggle").addEventListener("click", () => setTheme(isDark() ? "light" : "dark"));
  // close an open menu when clicking anywhere else
  document.addEventListener("click", (e) => {
    $$("details.menu[open]").forEach((d) => { if (!d.contains(e.target)) d.open = false; });
  });
  darkQuery.addEventListener("change", () => { if (!document.documentElement.dataset.theme) themeChanged(); });
  $("#partZone").addEventListener("change", drawPartition);
  $("#shareVar").addEventListener("change", drawShare);
  themeChanged();
}

window.addEventListener("DOMContentLoaded", async () => {
  initMap(); wire(); initView(); setRtab(S.rtab); drawSteps();
  try {
    await loadAll();
    const st = await api("/api/calibrate/status");
    if (st.running) { $("#calProgress").hidden = false; $("#calStartBtn").disabled = true; $("#calCancelBtn").disabled = false; pollCalibration(); }
  } catch (e) { toast(`Could not reach the model server: ${e.message}`, true, 8000); }
});

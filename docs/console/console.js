/* ethikit console.
 *
 * Live mode (served by `python -m ethikit serve`): builds a run config, POSTs it to
 * /api/run and animates the flow graph from the NDJSON event stream.
 * Replay mode (static docs site, no server): loads replay.json -- a recording of a
 * real run -- and plays its events back through the exact same code path.
 */
(function () {
  "use strict";
  var $ = function (s, el) { return (el || document).querySelector(s); };
  var NS = "http://www.w3.org/2000/svg";
  var PRINCIPLE_ORDER = ["utility", "fairness", "inclusiveness", "explainability", "transparency",
                         "privacy", "safety"];
  var GOVERNANCE = ["accountability", "human_oversight", "wellbeing"];

  var S = {
    meta: null, live: false, replay: null,
    source: "synthetic", upload: null, synth: {},
    models: new Set(["logistic_regression", "decision_tree", "random_forest"]),
    modelParams: {}, estimators: [], modelFile: null,
    tools: null, toolParams: {}, weights: {},
    result: null, running: false,
    checklistId: "deon", answers: {},
  };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    for (var k in attrs || {}) {
      if (k === "class") e.className = attrs[k]; else e.setAttribute(k, attrs[k]);
    }
    if (html != null) e.innerHTML = html;
    return e;
  }
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmt(v) {
    if (v == null || v !== v) return "–";
    if (typeof v !== "number") return String(v);
    if (Number.isInteger(v)) return String(v);
    return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3);
  }
  function store(k, v) { try { localStorage.setItem("ethikit-" + k, JSON.stringify(v)); } catch (e) {} }
  function load(k) { try { return JSON.parse(localStorage.getItem("ethikit-" + k)); } catch (e) { return null; } }

  // ------------------------------------------------------------------ boot
  function boot() {
    fetch("api/meta").then(function (r) {
      if (!r.ok) throw new Error("no server");
      return r.json();
    }).then(function (meta) {
      S.meta = meta; S.live = true; init();
    }).catch(function () {
      fetch("replay.json").then(function (r) { return r.json(); }).then(function (rep) {
        S.replay = rep; S.meta = rep.meta; S.live = false;
        $("#replay-banner").hidden = false;
        init();
        applyConfig(rep.config);
      }).catch(function () {
        $("#log").textContent = "could not reach the ethikit server (python -m ethikit serve) " +
          "and no replay.json was found next to this page.";
      });
    });
  }

  function isRegression() { return S.source === "regression"; }
  function zoo() { return isRegression() ? S.meta.regressors : S.meta.classifiers; }
  function allZooModels() {
    var out = {}; var z = zoo();
    for (var f in z) for (var m in z[f]) out[m] = z[f][m];
    return out;
  }

  function init() {
    S.tools = new Set(S.meta.tools.filter(function (t) { return t.available; }).map(function (t) { return t.name; }));
    S.weights = Object.assign({}, S.meta.default_weights);
    S.synth = Object.assign({}, S.meta.synthetic_knobs);

    var src = $("#data-source");
    Object.keys(S.meta.datasets).forEach(function (k) {
      if (!S.live && k === "csv") return;
      var o = el("option", { value: k }, esc(k === "csv" ? "your own CSV (upload)" : k));
      src.appendChild(o);
    });
    src.value = S.source;
    src.addEventListener("change", function () { S.source = src.value; renderData(); renderModels(); update(); });

    $("#csv-file").addEventListener("change", uploadCsv);
    $("#model-file").addEventListener("change", uploadModel);
    ["#target", "#sensitive", "#exclude-sensitive", "#positive-label", "#sensitive-threshold",
     "#seed", "#test-size", "#no-scale"].forEach(function (s) {
      $(s).addEventListener("input", update);
      $(s).addEventListener("change", update);
    });
    $("#est-add").addEventListener("click", addEstimator);
    $("#copy").addEventListener("click", function () {
      navigator.clipboard && navigator.clipboard.writeText(commandLine());
      $("#copy").textContent = "copied"; setTimeout(function () { $("#copy").textContent = "copy"; }, 1200);
    });
    $("#run").addEventListener("click", run);
    if (!S.live) {
      $("#row-modelfile").hidden = true;
      $("#run").textContent = "▶ replay run";
    }
    document.querySelectorAll(".tabs button").forEach(function (b) {
      b.addEventListener("click", function () { showTab(b.dataset.tab); });
    });

    var saved = load("answers");
    if (saved && saved.checklist) { S.checklistId = saved.checklist; S.answers = saved.answers || {}; }
    renderData(); renderModels(); renderTools(); renderWeights(); renderGov(); update();
    drawGraphIdle();
  }

  function applyConfig(cfg) {
    S.source = cfg.data.source; $("#data-source").value = S.source;
    S.models = new Set(cfg.models.map(function (m) { return m.name; }));
    if (cfg.tools) S.tools = new Set(cfg.tools);
    S.weights = Object.assign({}, cfg.weights);
    renderData(); renderModels(); renderTools(); renderWeights(); update();
  }

  // ------------------------------------------------------------------ configure
  function renderData() {
    $("#data-desc").textContent = S.meta.datasets[S.source] || "";
    $("#row-upload").hidden = S.source !== "csv";
    $("#row-columns").hidden = !(S.source === "csv" && S.upload);
    $("#row-synth").hidden = S.source !== "synthetic_custom";
    var knobs = $("#synth-knobs"); knobs.innerHTML = "";
    Object.keys(S.meta.synthetic_knobs).forEach(function (k) {
      var w = el("span", { class: "kv" }, "<span>" + esc(k) + "=</span>");
      var inp = el("input", { type: "number", step: "any", value: S.synth[k], "aria-label": k });
      inp.addEventListener("input", function () { S.synth[k] = inp.value; update(); });
      w.appendChild(inp); knobs.appendChild(w);
    });
  }

  function uploadCsv(ev) {
    var f = ev.target.files[0]; if (!f) return;
    $("#csv-info").textContent = "uploading…";
    fetch("api/upload?name=" + encodeURIComponent(f.name), { method: "POST", body: f })
      .then(function (r) { return r.json(); }).then(function (info) {
        if (info.error) throw new Error(info.error);
        S.upload = info;
        $("#csv-info").textContent = info.n_rows + " rows × " + info.columns.length + " columns";
        ["#target", "#sensitive"].forEach(function (sel) {
          var s = $(sel); s.innerHTML = sel === "#sensitive" ? "<option value=''>(none)</option>" : "";
          info.columns.forEach(function (c) {
            s.appendChild(el("option", { value: c.name },
              esc(c.name) + " · " + esc(c.dtype) + " · " + c.nunique + " values"));
          });
        });
        $("#target").value = info.suggest.target;
        $("#sensitive").value = info.suggest.sensitive || "";
        renderData(); update();
      }).catch(function (e) { $("#csv-info").innerHTML = "<span class='err'>" + esc(e.message) + "</span>"; });
  }

  function uploadModel(ev) {
    var f = ev.target.files[0]; if (!f) return;
    $("#model-file-info").textContent = "loading…";
    fetch("api/upload-model", { method: "POST", body: f }).then(function (r) { return r.json(); })
      .then(function (info) {
        if (info.error) throw new Error(info.error);
        S.modelFile = { id: info.upload_model_id, name: f.name.replace(/\.[^.]+$/, "").replace(/\W+/g, "_"),
                        file: f.name, cls: info.class };
        $("#model-file-info").innerHTML = "<span class='ok'>✓</span> " + esc(info.class) + " loaded as <b>" +
          esc(S.modelFile.name) + "</b>" + (info.n_features_in ? " · expects " + info.n_features_in + " features" : "") +
          " — tip: use --no-scale if you trained it on raw values";
        update();
      }).catch(function (e) { $("#model-file-info").innerHTML = "<span class='err'>" + esc(e.message) + "</span>"; });
  }

  function chip(label, on, disabled, onToggle, tag) {
    var c = el("label", { class: "chip" + (on ? " on" : "") + (disabled ? " off" : "") });
    var i = el("input", { type: "checkbox" }); i.checked = on; i.disabled = !!disabled;
    i.addEventListener("change", function () { c.classList.toggle("on", i.checked); onToggle(i.checked); });
    c.appendChild(i); c.appendChild(document.createTextNode(label));
    if (tag) c.appendChild(el("span", { class: "tag" }, esc(tag)));
    return c;
  }

  function renderModels() {
    var box = $("#models"); box.innerHTML = "";
    var z = zoo(); var known = allZooModels();
    S.models.forEach(function (m) { if (!known[m]) S.models.delete(m); });
    if (!S.models.size) Object.keys(z).slice(0, 2).forEach(function (f) { Object.keys(z[f]).forEach(function (m) { S.models.add(m); }); });
    Object.keys(z).forEach(function (fam) {
      var row = el("div", { class: "fam" }, "<span class='famname'>" + esc(fam) + "</span>");
      Object.keys(z[fam]).forEach(function (m) {
        row.appendChild(chip(m, S.models.has(m), false, function (on) {
          on ? S.models.add(m) : S.models.delete(m); renderAdvanced(); update();
        }));
      });
      box.appendChild(row);
    });
    renderAdvanced();
  }

  function addEstimator() {
    var name = $("#est-name").value.trim().replace(/\W+/g, "_");
    var path = $("#est-path").value.trim();
    if (!name || !/^[\w.]+\.\w+$/.test(path)) { $("#est-path").focus(); return; }
    S.estimators.push({ name: name, path: path });
    $("#est-name").value = ""; $("#est-path").value = "";
    renderEstimators(); update();
  }
  function renderEstimators() {
    var box = $("#est-list"); box.innerHTML = "";
    S.estimators.forEach(function (e, i) {
      var c = el("span", { class: "pill blue" }, esc(e.name) + "=" + esc(e.path) + " ");
      var x = el("button", { class: "btn", type: "button", "aria-label": "remove " + e.name }, "×");
      x.style.padding = "0 5px";
      x.addEventListener("click", function () { S.estimators.splice(i, 1); renderEstimators(); update(); });
      c.appendChild(x); box.appendChild(c);
    });
  }

  function renderTools() {
    var box = $("#tools"); box.innerHTML = "";
    S.meta.tools.forEach(function (t) {
      var c = chip(t.name, S.tools.has(t.name), !t.available, function (on) {
        on ? S.tools.add(t.name) : S.tools.delete(t.name); renderAdvanced(); update();
      }, t.available ? t.principles.join("+") : "not installed");
      c.title = t.mechanism;
      box.appendChild(c);
    });
    renderAdvanced();
  }

  function renderWeights() {
    var box = $("#weights"); box.innerHTML = "";
    Object.keys(S.meta.default_weights).forEach(function (p) {
      var w = el("span", { class: "kv" }, "<span>" + esc(p) + "=</span>");
      var inp = el("input", { type: "number", step: "0.05", min: "0", max: "1", value: S.weights[p], "aria-label": p + " weight" });
      inp.addEventListener("input", function () {
        S.weights[p] = parseFloat(inp.value) || 0; update();
        if (S.result) renderResults();   // re-rank instantly, no re-run
      });
      w.appendChild(inp); box.appendChild(w);
    });
    box.appendChild(el("span", { class: "dim small" }, "re-orders instantly after a run — weights are your value judgement, not the tool's"));
  }

  function renderAdvanced() {
    if (!S.meta) return;
    var known = allZooModels();
    var mbox = $("#adv-models"); mbox.innerHTML = "";
    S.models.forEach(function (m) {
      var defaults = known[m]; if (!defaults) return;
      mbox.appendChild(paramCard(m + "  (model)", defaults, S.modelParams, m));
    });
    var tbox = $("#adv-tools"); tbox.innerHTML = "";
    S.meta.tools.forEach(function (t) {
      if (!S.tools.has(t.name) || !Object.keys(t.params).length) return;
      tbox.appendChild(paramCard(t.name + "  (tool)", t.params, S.toolParams, t.name));
    });
    countChanged();
  }
  function paramCard(title, defaults, store_, key) {
    var card = el("div", { class: "adv-card" }, "<h4>" + esc(title) + "</h4>");
    Object.keys(defaults).forEach(function (p) {
      var def = defaults[p];
      var cur = store_[key] && store_[key][p] != null ? store_[key][p] : pyRepr(def);
      var row = el("div", { class: "row" + (store_[key] && store_[key][p] != null ? " changed" : "") },
        "<span title='" + esc(p) + "'>" + esc(p) + "</span>");
      var inp = el("input", { type: "text", value: cur, "aria-label": key + " " + p });
      inp.addEventListener("input", function () {
        store_[key] = store_[key] || {};
        if (inp.value === pyRepr(def)) delete store_[key][p]; else store_[key][p] = inp.value;
        row.classList.toggle("changed", store_[key][p] != null);
        countChanged(); update();
      });
      row.appendChild(inp); card.appendChild(row);
    });
    return card;
  }
  function pyRepr(v) {
    if (v === null) return "None";
    if (v === true) return "True";
    if (v === false) return "False";
    return String(v);
  }
  function countChanged() {
    var n = 0;
    [S.modelParams, S.toolParams].forEach(function (s) { for (var k in s) n += Object.keys(s[k]).length; });
    $("#adv-count").textContent = n;
  }

  function config() {
    var data = { source: S.source };
    if (S.source === "csv") {
      data.source = "upload"; data.upload_id = S.upload && S.upload.upload_id;
      data.name = S.upload && S.upload.name;
      data.target = $("#target").value; data.sensitive = $("#sensitive").value || null;
      if ($("#exclude-sensitive").checked) data.exclude_sensitive = true;
      if ($("#positive-label").value) data.positive_label = $("#positive-label").value;
      if ($("#sensitive-threshold").value) data.sensitive_threshold = parseFloat($("#sensitive-threshold").value);
    }
    if (S.source === "synthetic_custom") data.synthetic = S.synth;
    var models = [];
    S.models.forEach(function (m) { models.push({ name: m, params: S.modelParams[m] || {} }); });
    S.estimators.forEach(function (e) { models.push({ name: e.name, estimator: e.path, params: {} }); });
    if (S.modelFile) models.push({ name: S.modelFile.name, upload_model_id: S.modelFile.id });
    var tp = {};
    for (var t in S.toolParams) if (S.tools.has(t) && Object.keys(S.toolParams[t]).length) tp[t] = S.toolParams[t];
    return {
      data: data,
      split: { test_size: parseFloat($("#test-size").value) || 0.3, seed: parseInt($("#seed").value, 10) || 0,
               scale: !$("#no-scale").checked },
      models: models, tools: Array.from(S.tools), tool_params: tp, weights: S.weights,
      checklist: Object.keys(S.answers).length ? { checklist: S.checklistId, answers: S.answers } : null,
    };
  }

  function commandLine() {
    var c = config(); var parts = ["ethikit run"];
    if (S.source === "csv") {
      parts.push("--data " + ((S.upload && S.upload.name) || "your_data") + ".csv");
      if (c.data.target) parts.push("--target " + c.data.target);
      if (c.data.sensitive) parts.push("--sensitive " + c.data.sensitive);
      if (c.data.exclude_sensitive) parts.push("--exclude-sensitive");
      if (c.data.positive_label) parts.push("--positive-label " + c.data.positive_label);
      if (c.data.sensitive_threshold != null) parts.push("--sensitive-threshold " + c.data.sensitive_threshold);
    } else if (S.source === "synthetic_custom") {
      var kv = []; var d = S.meta.synthetic_knobs;
      for (var k in S.synth) if (String(S.synth[k]) !== String(d[k])) kv.push(k + "=" + S.synth[k]);
      parts.push("--synthetic " + (kv.length ? kv.join(" ") : "n=" + S.synth.n));
    } else parts.push("--data " + S.source);
    var zooNames = Array.from(S.models);
    if (zooNames.length) parts.push("--models " + zooNames.join(","));
    zooNames.forEach(function (m) {
      for (var p in (S.modelParams[m] || {})) parts.push("--model-param " + m + "." + p + "=" + quote(S.modelParams[m][p]));
    });
    S.estimators.forEach(function (e) { parts.push("--estimator " + e.name + "=" + e.path); });
    if (S.modelFile) parts.push("--model-file " + S.modelFile.name + "=" + S.modelFile.file);
    var avail = S.meta.tools.filter(function (t) { return t.available; }).map(function (t) { return t.name; });
    if (S.tools.size !== avail.length) parts.push("--tools " + Array.from(S.tools).join(","));
    for (var t in c.tool_params) for (var q in c.tool_params[t]) parts.push("--tool-param " + t + "." + q + "=" + quote(c.tool_params[t][q]));
    var w = []; var dw = S.meta.default_weights;
    for (var p2 in S.weights) if (S.weights[p2] !== dw[p2]) w.push(p2 + "=" + S.weights[p2]);
    if (w.length) parts.push("--weights " + w.join(","));
    if (c.split.seed !== 0) parts.push("--seed " + c.split.seed);
    if (c.split.test_size !== 0.3) parts.push("--test-size " + c.split.test_size);
    if (!c.split.scale) parts.push("--no-scale");
    if (c.checklist) parts.push("--checklist " + S.checklistId + "_answers.json");
    return parts.join(" \\\n    ");
  }
  function quote(v) { return /[\s'"(),]/.test(v) ? '"' + String(v).replace(/"/g, '\\"') + '"' : v; }

  function update() {
    $("#cmd-preview").textContent = commandLine();
    var ok = S.models.size + S.estimators.length + (S.modelFile ? 1 : 0) > 0 && S.tools.size > 0 &&
             (S.source !== "csv" || (S.upload && $("#target").value));
    $("#run").disabled = S.running || !ok;
  }

  // ------------------------------------------------------------------ run + stream
  var LOG_T0 = 0;
  function log(kind, text) {
    var lg = $("#log");
    var t = ((performance.now() - LOG_T0) / 1000).toFixed(2).padStart(7, " ") + "s ";
    var line = el("span", null, "<span class='t'>" + t + "</span><span class='k-" + kind + "'>" +
      esc(text) + "</span>\n");
    lg.appendChild(line);
    lg.scrollTop = lg.scrollHeight;
  }

  function run() {
    if (S.running) return;
    S.running = true; S.result = null; update();
    $("#log").innerHTML = ""; LOG_T0 = performance.now();
    $("#run-label").textContent = "";
    if (!S.live) return replay();
    log("run", "» " + commandLine().replace(/\s*\\\n\s*/g, " "));
    fetch("api/run", { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(config()) })
      .then(function (resp) {
        var reader = resp.body.getReader(); var dec = new TextDecoder(); var buf = "";
        function pump() {
          return reader.read().then(function (r) {
            if (r.done) { if (buf.trim()) onEvent(JSON.parse(buf)); return finish(); }
            buf += dec.decode(r.value, { stream: true });
            var lines = buf.split("\n"); buf = lines.pop();
            lines.forEach(function (l) { if (l.trim()) onEvent(JSON.parse(l)); });
            return pump();
          });
        }
        return pump();
      }).catch(function (e) { log("err", "connection error: " + e.message); finish(); });
  }

  function replay() {
    var rep = S.replay; var evs = rep.events.slice();
    log("run", "» " + commandLine().replace(/\s*\\\n\s*/g, " ") + "   # replaying a recorded run");
    var total = evs.length ? evs[evs.length - 1].t : 1;
    var scale = Math.min(1, 14 / Math.max(total, 0.001));  // compress to ≤ ~14 s
    var i = 0; var start = performance.now();
    function tick() {
      var now = (performance.now() - start) / 1000;
      while (i < evs.length && evs[i].t * scale <= now) {
        var e = evs[i++];
        // make the replay readable even where the real run was instantaneous
        if (e.type === "tool_start" || e.type === "model_fit_start") { onEvent(e); break; }
        onEvent(e);
      }
      if (i < evs.length) setTimeout(tick, 70);
      else { onEvent(rep.result); finish(); }
    }
    tick();
  }

  function finish() { S.running = false; update(); }

  var G = null;   // graph state
  function onEvent(e) {
    switch (e.type) {
      case "prepare": log("run", "preparing data + models (run " + e.run_id + ")"); break;
      case "note": log("skip", "note: " + e.message); break;
      case "run_start":
        G = buildGraph(e);
        log("run", "run   dataset=" + e.dataset + " task=" + e.task + " train=" + e.n_train +
            " test=" + e.n_test + " features=" + e.n_features + " sensitive=" + e.sensitive);
        log("run", "      " + e.models.length + " models × " + e.tools.length + " tools");
        break;
      case "model_fit_start":
        log("fit", "fit   " + e.model + (e.prefit ? "  (pre-fitted, evaluated as-is)" : ""));
        G && G.fitStart(e.model); break;
      case "model_fit_done":
        log("ok", "      ✓ " + e.model + " ready in " + e.seconds.toFixed(2) + "s");
        G && G.fitDone(e.model); break;
      case "tool_start":
        log("run", "exec  " + e.tool + " → " + e.model);
        G && G.toolStart(e.model, e.tool); break;
      case "tool_done":
        log("ok", "  ✓   " + e.tool + "  " + e.results.map(function (r) { return r.metric + "=" + fmt(r.value); }).join("  ") +
            "  (" + e.seconds.toFixed(2) + "s)");
        G && G.toolDone(e.model, e.tool, e.results); break;
      case "tool_skip":
        log("skip", "skip  " + e.tool + " → " + e.model + ": " + e.reason);
        G && G.toolSkip(e.model, e.tool); break;
      case "tool_error":
        log("err", "  ✗   " + e.tool + " → " + e.model + ": " + e.error);
        G && G.toolError(e.model, e.tool); break;
      case "run_done": log("ok", "done  " + e.n_records + " measurements"); break;
      case "error": log("err", "error " + e.error); break;
      case "result":
        S.result = e;
        $("#run-label").textContent = e.dataset + " · " + Object.keys(e.objective).length + " models";
        G && G.final(e);
        renderResults(); renderInspect(); renderCardTab();
        log("ok", "pareto front: " + e.pareto.join(", "));
        break;
    }
  }

  // ------------------------------------------------------------------ flow graph
  function drawGraphIdle() {
    var g = $("#graph"); g.innerHTML = "";
    g.setAttribute("viewBox", "0 0 1000 220");
    var t = svg("text", { x: 500, y: 110, "text-anchor": "middle", class: "g-col" }, g);
    t.textContent = "the run's flow graph appears here: inputs → tools under each principle → results";
  }

  function buildGraph(ev) {
    var g = $("#graph"); g.innerHTML = "";
    var tools = ev.tools;
    var principles = {};
    tools.forEach(function (t) { t.principles.forEach(function (p) { (principles[p] = principles[p] || []).push(t.name); }); });
    var plist = Object.keys(principles).sort(function (a, b) {
      return (PRINCIPLE_ORDER.indexOf(a) + 99) % 99 - (PRINCIPLE_ORDER.indexOf(b) + 99) % 99;
    });
    var models = ev.models;
    var X_IN = 16, W_IN = 190, X_PR = 300, W_PR = 340, X_RES = 740, W_RES = 244;
    var CH = 24, CW = 160, PAD = 28;
    // principle boxes
    var y = 34, boxes = {};
    plist.forEach(function (p) {
      var n = principles[p].length; var rows = Math.ceil(n / 2);
      var h = PAD + rows * (CH + 6) + 4;
      boxes[p] = { y: y, h: h, tools: principles[p] };
      y += h + 12;
    });
    // governance box (not computed) for honesty
    var govY = y; y += 40;
    var resH = Math.max(36, 22 + models.length * 14);
    var H = Math.max(y, 34 + models.length * 50 + 70);
    // result nodes need room
    var resY = {}; var ry = 34;
    plist.forEach(function (p) { resY[p] = Math.max(ry, boxes[p].y + boxes[p].h / 2 - resH / 2); ry = resY[p] + resH + 8; });
    H = Math.max(H, ry + 10);
    g.setAttribute("viewBox", "0 0 1000 " + H);

    ["INPUTS", "TOOLS, GROUPED BY PRINCIPLE", "RESULTS PER PRINCIPLE"].forEach(function (lbl, i) {
      var t = svg("text", { x: [X_IN, X_PR, X_RES][i], y: 18, class: "g-col" }, g); t.textContent = lbl;
    });
    var edges = svg("g", null, g);
    var layer = svg("g", null, g);
    var parts = svg("g", null, g);

    // dataset node
    var dn = svg("g", { class: "g-node data" }, layer);
    svg("rect", { x: X_IN, y: 34, width: W_IN, height: 40, rx: 5 }, dn);
    var t1 = svg("text", { x: X_IN + 10, y: 51 }, dn); t1.textContent = trunc(ev.dataset, 24);
    var t2 = svg("text", { x: X_IN + 10, y: 66, class: "sub" }, dn);
    t2.textContent = ev.n_train + "+" + ev.n_test + " rows · A=" + (ev.sensitive || "–");

    var modelNodes = {};
    models.forEach(function (m, i) {
      var my = 92 + i * 44;
      var n = svg("g", { class: "g-node model" }, layer);
      svg("rect", { x: X_IN + 14, y: my, width: W_IN - 14, height: 34, rx: 5 }, n);
      var sw = svg("rect", { x: X_IN + 22, y: my + 12, width: 10, height: 10, rx: 2, class: "s" + (i % 10) }, n);
      sw.setAttribute("style", "fill:var(--s)");
      var t = svg("text", { x: X_IN + 40, y: my + 15 }, n); t.textContent = trunc(m, 20);
      var st = svg("text", { x: X_IN + 40, y: my + 28, class: "sub" }, n); st.textContent = "queued";
      var e = svg("path", { class: "g-edge", d: curve(X_IN + 30, 74, X_IN + 14, my + 17, true) }, edges);
      modelNodes[m] = { g: n, status: st, y: my + 17, edge: e, idx: i };
    });

    var toolNodes = {};   // tool -> [ {g, cnt} per principle ]
    var modelEdges = {};  // model|principle -> path
    var resNodes = {};
    plist.forEach(function (p) {
      var b = boxes[p];
      var bx = svg("g", { class: "g-box" }, layer);
      svg("rect", { x: X_PR, y: b.y, width: W_PR, height: b.h, rx: 6 }, bx);
      var tt = svg("text", { x: X_PR + 10, y: b.y + 16, class: "ph" }, bx); tt.textContent = p;
      b.tools.forEach(function (tn, j) {
        var cx = X_PR + 10 + (j % 2) * (CW + 10), cy = b.y + PAD - 4 + Math.floor(j / 2) * (CH + 6);
        var tg = svg("g", { class: "g-tool" }, bx);
        svg("rect", { x: cx, y: cy, width: CW, height: CH, rx: 4 }, tg);
        var lab = svg("text", { x: cx + 8, y: cy + 16 }, tg); lab.textContent = trunc(tn, 16);
        var cnt = svg("text", { x: cx + CW - 6, y: cy + 16, "text-anchor": "end", class: "cnt" }, tg);
        cnt.textContent = "0/" + models.length;
        (toolNodes[tn] = toolNodes[tn] || []).push({ g: tg, cnt: cnt, done: 0 });
      });
      models.forEach(function (m) {
        modelEdges[m + "|" + p] = svg("path", { class: "g-edge", d: curve(X_IN + W_IN, modelNodes[m].y, X_PR, b.y + b.h / 2) }, edges);
      });
      // result node
      var rn = svg("g", { class: "g-res" }, layer);
      var ry2 = resY[p];
      svg("rect", { x: X_RES, y: ry2, width: W_RES, height: resH, rx: 5 }, rn);
      var rt = svg("text", { x: X_RES + 8, y: ry2 + 14 }, rn); rt.textContent = p;
      var rs = svg("text", { x: X_RES + W_RES - 8, y: ry2 + 14, "text-anchor": "end", class: "sub" }, rn); rs.textContent = "waiting";
      var bars = svg("g", null, rn);
      var re = svg("path", { class: "g-edge res", d: curve(X_PR + W_PR, b.y + b.h / 2, X_RES, ry2 + resH / 2) }, edges);
      resNodes[p] = { g: rn, sub: rs, bars: bars, y: ry2, edge: re, latest: {} };
    });
    // governance: shown, not computed
    var gv = svg("g", { class: "g-box gov" }, layer);
    svg("rect", { x: X_PR, y: govY, width: W_PR, height: 30, rx: 6 }, gv);
    var gt = svg("text", { x: X_PR + 10, y: govY + 19, class: "ph" }, gv);
    gt.textContent = "governance · checklist, not a metric";

    // ---- particles
    var streams = {}; var dots = [];
    function stream(key, path, cls, on) {
      if (on) streams[key] = { path: path, cls: cls, last: 0 }; else delete streams[key];
    }
    function burst(path, cls, n) { for (var i = 0; i < n; i++) dots.push({ path: path, cls: cls, t: -i * 0.12, len: path.getTotalLength() }); }
    var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    function frame(ts) {
      if (!document.body.contains(g) || G !== api) return;
      for (var k in streams) {
        var s = streams[k];
        if (ts - s.last > 160) { s.last = ts; dots.push({ path: s.path, cls: s.cls, t: 0, len: s.path.getTotalLength() }); }
      }
      dots = dots.filter(function (d) {
        d.t += 0.018;
        if (d.t > 1) { d.c && d.c.remove(); return false; }
        if (d.t < 0) return true;
        if (!d.c) d.c = svg("circle", { r: 2.6, class: "pt " + d.cls }, parts);
        var p = d.path.getPointAtLength(d.t * d.len);
        d.c.setAttribute("cx", p.x); d.c.setAttribute("cy", p.y);
        return true;
      });
      requestAnimationFrame(frame);
    }
    if (!reduce) requestAnimationFrame(frame);

    var total = models.length * tools.length, done = 0;
    function progress() {
      $("#prog").style.width = (100 * done / Math.max(total, 1)) + "%";
      $("#prog-text").textContent = done + " / " + total + " tool runs";
    }
    progress();
    var toolPrinciples = {}; tools.forEach(function (t) { toolPrinciples[t.name] = t.principles; });

    var api = {
      fitStart: function (m) {
        var n = modelNodes[m]; if (!n) return;
        n.g.classList.add("active"); n.status.textContent = "fitting…";
        n.edge.classList.add("hot"); stream("fit", n.edge, "fit", true);
      },
      fitDone: function (m) {
        var n = modelNodes[m]; if (!n) return;
        n.status.textContent = "evaluating"; n.edge.classList.remove("hot"); stream("fit", null, "", false);
      },
      toolStart: function (m, t) {
        (toolNodes[t] || []).forEach(function (x) { x.g.classList.add("running"); x.g.classList.remove("done"); });
        (toolPrinciples[t] || []).forEach(function (p) {
          var e = modelEdges[m + "|" + p]; if (e) { e.classList.add("hot"); stream(m + "|" + p, e, "", true); }
        });
      },
      toolDone: function (m, t, results) {
        done++; progress();
        (toolNodes[t] || []).forEach(function (x) {
          x.done++; x.cnt.textContent = x.done + "/" + models.length;
          x.g.classList.remove("running"); x.g.classList.add("done");
        });
        (toolPrinciples[t] || []).forEach(function (p) {
          var e = modelEdges[m + "|" + p]; if (e) { e.classList.remove("hot"); stream(m + "|" + p, null, "", false); }
          var r = resNodes[p]; if (!r) return;
          r.edge.classList.add("hot"); burst(r.edge, "res", 4);
          setTimeout(function () { r.edge.classList.remove("hot"); }, 700);
        });
        results.forEach(function (res) {
          var r = resNodes[res.principle]; if (!r) return;
          r.latest[m] = r.latest[m] || {}; r.latest[m][res.metric] = res.value;
          drawLatest(r, m);
        });
      },
      toolSkip: function (m, t) { done++; progress(); },
      toolError: function (m, t) {
        done++; progress();
        (toolNodes[t] || []).forEach(function (x) { x.g.classList.remove("running"); x.g.classList.add("error"); });
      },
      final: function (res) {
        for (var m in modelNodes) { modelNodes[m].g.classList.remove("active"); modelNodes[m].g.classList.add("done"); modelNodes[m].status.textContent = res.pareto.indexOf(m) >= 0 ? "pareto-optimal" : "dominated"; }
        for (var p in resNodes) drawFinal(resNodes[p], p, res);
        streams = {};
        done = total; progress();
      },
    };
    function drawLatest(r, m) {
      r.bars.innerHTML = "";
      var i = 0, txt = [];
      for (var metric in r.latest[m]) txt.push(metric.replace(/_/g, " ") + " " + fmt(r.latest[m][metric]));
      txt.slice(0, Math.max(1, Math.floor((resH - 22) / 13))).forEach(function (s) {
        var t = svg("text", { x: X_RES + 8, y: r.y + 30 + (i++) * 13, class: "val" }, r.bars); t.textContent = trunc(s, 38);
      });
      r.sub.textContent = "raw · " + trunc(m, 16);
      r.g.querySelector("rect").setAttribute("class", "s" + modelNodes[m].idx);
      r.g.querySelector("rect").setAttribute("style", "stroke:var(--s)");
    }
    function drawFinal(r, p, res) {
      r.bars.innerHTML = "";
      r.sub.textContent = "0–1, 1 = best";
      r.g.querySelector("rect").removeAttribute("style");
      if (models.every(function (m) { return (res.objective[m] || {})[p] == null; })) {
        r.sub.textContent = "no metric";
        var nt = svg("text", { x: X_RES + 8, y: r.y + 30, class: "val" }, r.bars);
        nt.textContent = "listed by a tool but not scored —";
        var nt2 = svg("text", { x: X_RES + 8, y: r.y + 43, class: "val" }, r.bars);
        nt2.textContent = "evidenced by the model card instead";
        return;
      }
      models.forEach(function (m, i) {
        var v = (res.objective[m] || {})[p];
        var by = r.y + 22 + i * 14;
        svg("rect", { x: X_RES + 8, y: by, width: 120, height: 8, rx: 2, class: "bar-bg" }, r.bars);
        var bar = svg("rect", { x: X_RES + 8, y: by, width: Math.max(0, 120 * (v || 0)), height: 8, rx: 2, class: "s" + (i % 10) }, r.bars);
        bar.setAttribute("style", "fill:var(--s)");
        var t = svg("text", { x: X_RES + 134, y: by + 8, class: "val" }, r.bars);
        t.textContent = fmt(v) + " " + trunc(m, 12);
      });
    }
    return api;
  }
  function curve(x1, y1, x2, y2, vertical) {
    if (vertical) return "M" + x1 + "," + y1 + " C" + x1 + "," + (y1 + y2) / 2 + " " + x1 + "," + y2 + " " + x2 + "," + y2;
    var mx = (x1 + x2) / 2;
    return "M" + x1 + "," + y1 + " C" + mx + "," + y1 + " " + mx + "," + y2 + " " + x2 + "," + y2;
  }
  function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  // ------------------------------------------------------------------ results
  function scored() {
    var r = S.result; var models = Object.keys(r.objective);
    var cols = [];
    PRINCIPLE_ORDER.forEach(function (p) { if (models.some(function (m) { return r.objective[m][p] != null; })) cols.push(p); });
    for (var m0 in r.objective) for (var c in r.objective[m0]) if (cols.indexOf(c) < 0) cols.push(c);
    var w = {}, wsum = 0;
    for (var p in S.weights) if (cols.indexOf(p) >= 0 && S.weights[p] > 0) { w[p] = S.weights[p]; wsum += S.weights[p]; }
    var score = {};
    models.forEach(function (m) {
      var s = 0; for (var p in w) s += (r.objective[m][p] || 0) * w[p] / (wsum || 1); score[m] = s;
    });
    models.sort(function (a, b) { return score[b] - score[a]; });
    return { models: models, cols: cols, score: score, wsum: wsum };
  }
  function colourIndex(m) { return Object.keys(S.result.objective).indexOf(m) % 10; }

  function renderResults() {
    var r = S.result; var sc = scored();
    var box = $("#tab-results"); box.innerHTML = "";
    var head = "<tr><th>model</th><th>family</th>" + sc.cols.map(function (c) { return "<th>" + esc(c) + "</th>"; }).join("") +
      "<th>weighted</th><th>pareto</th></tr>";
    var rows = sc.models.map(function (m) {
      return "<tr><td><span class='sw s" + colourIndex(m) + "'></span> " + esc(m) + "</td><td class='l dim'>" + esc(r.families[m] || "custom") + "</td>" +
        sc.cols.map(function (c) { var v = r.objective[m][c]; return "<td class='heat' style='--h:" + (v || 0) + "'>" + fmt(v) + "</td>"; }).join("") +
        "<td><b>" + fmt(sc.score[m]) + "</b></td><td>" + (r.pareto.indexOf(m) >= 0 ? "<span class='pill'>front</span>" : "<span class='dim'>–</span>") + "</td></tr>";
    }).join("");
    var wtxt = Object.keys(S.weights).filter(function (p) { return S.weights[p] > 0; }).map(function (p) { return p + "=" + S.weights[p]; }).join(", ");
    box.innerHTML =
      "<h3>comparison</h3><p class='prose'>Each column is min-max normalised within this run, so 1 = best <i>among these candidates</i>, not an absolute grade. " +
      "The weighted column only exists because you supplied weights (" + esc(wtxt) + "); change them in <code>--weights</code> above and the order updates without re-running. This is a selection aid, not a verdict: open <b>inspect</b> before trusting any number.</p>" +
      "<div class='tscroll'><table class='t'>" + head + rows + "</table></div>" +
      "<div class='res-grid'><div><h3>per-principle radar</h3><div id='radar'></div><div class='legend' id='legend'></div></div>" +
      "<div><h3>pareto front</h3><p class='prose'>Models no other candidate beats on <b>every</b> column at once. Choosing among them is a value judgement — that's what the weights express.</p>" +
      "<div>" + r.pareto.map(function (m) { return "<span class='pill'>" + esc(m) + "</span>"; }).join("") + "</div>" +
      "<div class='note'>Scores are relative to this dataset and this split. The same model can score differently on other data — try <code>--synthetic minority_share=0.15</code> or your own CSV, or vary <code>--seed</code>.</div>" +
      (r.notes || []).map(function (n) { return "<div class='note'>" + esc(n) + "</div>"; }).join("") +
      (S.live ? "<p style='margin-top:12px'><a class='btn' href='api/report/" + esc(r.run_id) + "' target='_blank' rel='noopener'>open aggregate report ↗</a></p>" : "") +
      "</div></div>";
    radar($("#radar"), sc);
  }

  function radar(box, sc) {
    var r = S.result;
    var axes = sc.cols.filter(function (c) { return c !== "utility"; });
    if (axes.length < 3) axes = sc.cols.slice();
    var W = 520, H = 380, cx = W / 2, cy = H / 2 + 4, R = 135;
    var g = svg("svg", { viewBox: "0 0 " + W + " " + H, class: "radar", width: "100%", role: "img", "aria-label": "Radar chart of per-principle scores" });
    [0.25, 0.5, 0.75, 1].forEach(function (f) {
      svg("polygon", { class: "grid", points: axes.map(function (_, i) { var a = ang(i); return (cx + Math.cos(a) * R * f) + "," + (cy + Math.sin(a) * R * f); }).join(" ") }, g);
    });
    axes.forEach(function (ax, i) {
      var a = ang(i);
      svg("line", { class: "grid", x1: cx, y1: cy, x2: cx + Math.cos(a) * R, y2: cy + Math.sin(a) * R }, g);
      var t = svg("text", { x: cx + Math.cos(a) * (R + 16), y: cy + Math.sin(a) * (R + 16) + 3,
        "text-anchor": Math.abs(Math.cos(a)) < 0.2 ? "middle" : (Math.cos(a) > 0 ? "start" : "end") }, g);
      t.textContent = ax;
    });
    var polys = {};
    sc.models.forEach(function (m) {
      var p = svg("polygon", { class: "poly s" + colourIndex(m), points: axes.map(function (ax, i) {
        var v = r.objective[m][ax] || 0, a = ang(i); return (cx + Math.cos(a) * R * v) + "," + (cy + Math.sin(a) * R * v);
      }).join(" ") }, g);
      polys[m] = p;
    });
    function ang(i) { return -Math.PI / 2 + i * 2 * Math.PI / axes.length; }
    box.appendChild(g);
    var lg = $("#legend");
    sc.models.forEach(function (m) {
      var it = el("span", { tabindex: "0", style: "cursor:pointer" }, "<span class='sw s" + colourIndex(m) + "'></span> " + esc(m));
      function hi(on) { for (var k in polys) polys[k].classList.toggle("dim", on && k !== m); }
      it.addEventListener("mouseenter", function () { hi(true); });
      it.addEventListener("mouseleave", function () { hi(false); });
      it.addEventListener("focus", function () { hi(true); });
      it.addEventListener("blur", function () { hi(false); });
      lg.appendChild(it);
    });
  }

  // ------------------------------------------------------------------ inspect
  function renderInspect() {
    var r = S.result; var box = $("#tab-inspect"); box.innerHTML = "";
    var models = Object.keys(r.traces);
    var bar = el("div", { class: "ins-bar" });
    var ms = el("select", { "aria-label": "model" }); models.forEach(function (m) { ms.appendChild(el("option", { value: m }, esc(m))); });
    var ts = el("select", { "aria-label": "tool" });
    bar.appendChild(el("span", { class: "dim" }, "» ethikit run … --trace  ·  model"));
    bar.appendChild(ms); bar.appendChild(el("span", { class: "dim" }, "tool")); bar.appendChild(ts);
    box.appendChild(bar);
    var body = el("div"); box.appendChild(body);
    function fillTools() {
      ts.innerHTML = ""; Object.keys(r.traces[ms.value] || {}).forEach(function (t) { ts.appendChild(el("option", { value: t }, esc(t))); });
      show();
    }
    function show() { body.innerHTML = ""; body.appendChild(inspectView(ms.value, ts.value)); }
    ms.addEventListener("change", fillTools); ts.addEventListener("change", show);
    if (models.length) fillTools(); else body.innerHTML = "<div class='empty'>no traces</div>";
  }

  function inspectView(model, tool) {
    var r = S.result; var tr = (r.traces[model] || {})[tool] || {}; var t = tr.trace || {};
    var meta = S.meta.tools.filter(function (x) { return x.name === tool; })[0] || {};
    var recs = r.records.filter(function (x) { return x.model === model && x.evaluator === tool; });
    var wrap = el("div", { class: "ins-grid" });
    var left = el("div"), right = el("div");
    left.innerHTML =
      "<div class='block'><h3>what it does</h3><p class='prose'>" + esc(meta.mechanism || "") + "</p></div>" +
      "<div class='block'><h3>the maths</h3><pre class='code'>" + esc(tr.formula || meta.formula || "") + "</pre></div>" +
      (t.steps && t.steps.length ? "<div class='block'><h3>worked through, this run</h3><ol class='steps'>" +
        t.steps.map(function (s) { return "<li>" + esc(s) + "</li>"; }).join("") + "</ol></div>" : "") +
      "<div class='block'><h3>measurements</h3><div class='tscroll'><table class='t'><tr><th>metric</th><th>principle</th><th>inspects</th><th>value</th><th>better</th></tr>" +
      recs.map(function (x) { return "<tr><td>" + esc(x.metric) + "</td><td class='l'>" + esc(x.principle) + "</td><td class='l'>" + esc(x.target) + "</td><td>" + fmt(x.value) + "</td><td class='l'>" + (x.higher_is_better ? "higher" : "lower") + "</td></tr>"; }).join("") +
      "</table></div></div>" +
      "<div class='block'><h3>run-specific mechanism</h3>" + (tr.mechanisms || []).map(function (m) { return "<p class='prose'>" + esc(m) + "</p>"; }).join("") + "</div>" +
      (tr.params && Object.keys(tr.params).length ? "<div class='block'><h3>tool parameters used</h3><pre class='code'>" +
        Object.keys(tr.params).map(function (k) { return "--tool-param " + tool + "." + k + "=" + tr.params[k]; }).join("\n") + "</pre></div>" : "");
    (t.charts || []).forEach(function (c) { right.appendChild(chart(c)); });
    (t.tables || []).forEach(function (tb) {
      var d = el("div", { class: "block" }, "<h3>" + esc(tb.title) + "</h3>");
      d.innerHTML += "<div class='tscroll'><table class='t'><tr>" + tb.columns.map(function (c) { return "<th>" + esc(c) + "</th>"; }).join("") + "</tr>" +
        tb.rows.map(function (row) { return "<tr>" + row.map(function (v) { return "<td>" + esc(typeof v === "number" ? fmt(v) : v) + "</td>"; }).join("") + "</tr>"; }).join("") + "</table></div>";
      right.appendChild(d);
    });
    (t.text || []).forEach(function (x) {
      right.appendChild(el("div", { class: "block" }, "<h3>" + esc(x.title) + "</h3><pre class='code'>" + esc(x.body) + "</pre>"));
    });
    if (!right.childNodes.length) right.innerHTML = "<div class='empty'>this tool records no intermediate artefacts</div>";
    wrap.appendChild(left); wrap.appendChild(right);
    return wrap;
  }

  function chart(c) {
    var d = el("div", { class: "block" }, "<h3>" + esc(c.title) + "</h3>");
    var labels = c.labels || []; var series = c.series || [];
    var W = 520, H = 200, L = 40, B = 34, T = 8;
    var max = 0; series.forEach(function (s) { s.values.forEach(function (v) { if (v > max) max = v; }); });
    max = max || 1;
    var g = svg("svg", { viewBox: "0 0 " + W + " " + H, class: "chart", width: "100%", role: "img", "aria-label": c.title });
    svg("line", { class: "axis", x1: L, y1: H - B, x2: W - 4, y2: H - B }, g);
    svg("line", { class: "axis", x1: L, y1: T, x2: L, y2: H - B }, g);
    [0, 0.5, 1].forEach(function (f) {
      var t = svg("text", { x: L - 4, y: H - B - f * (H - B - T) + 3, "text-anchor": "end" }, g); t.textContent = fmt(max * f);
    });
    var n = labels.length || 1; var slot = (W - L - 8) / n;
    var overlay = c.type === "hist";
    var bw = overlay ? slot * 0.9 : slot * 0.8 / Math.max(series.length, 1);
    series.forEach(function (s, si) {
      s.values.forEach(function (v, i) {
        var h = (H - B - T) * (v / max);
        var x = L + 4 + i * slot + (overlay ? 0 : si * bw);
        var b = svg("rect", { x: x, y: H - B - h, width: Math.max(1, bw - 1), height: h, class: "bar s" + ([0, 1, 2, 3][si] || si) }, g);
        if (overlay) b.setAttribute("fill-opacity", "0.55");
        var tt = svg("title", null, b); tt.textContent = s.name + " · " + labels[i] + ": " + fmt(v);
      });
    });
    var step = Math.ceil(n / 10);
    labels.forEach(function (l, i) {
      if (i % step) return;
      var t = svg("text", { x: L + 4 + i * slot + slot / 2, y: H - B + 14, "text-anchor": "middle" }, g); t.textContent = trunc(l, 10);
    });
    if (c.x_label) { var xl = svg("text", { x: W / 2, y: H - 4, "text-anchor": "middle" }, g); xl.textContent = c.x_label; }
    d.appendChild(g);
    if (series.length > 1) {
      d.appendChild(el("div", { class: "legend" }, series.map(function (s, si) {
        return "<span><span class='sw s" + si + "'></span> " + esc(s.name) + "</span>";
      }).join("")));
    }
    return d;
  }

  // ------------------------------------------------------------------ model card
  function renderCardTab() {
    var r = S.result; var box = $("#tab-card"); box.innerHTML = "";
    var bar = el("div", { class: "cardbar" });
    var ms = el("select", { "aria-label": "model" });
    scored().models.forEach(function (m) { ms.appendChild(el("option", { value: m }, esc(m))); });
    var open = el("a", { class: "btn", target: "_blank", rel: "noopener" }, "open ↗");
    var dl = el("a", { class: "btn" }, "download .html");
    bar.appendChild(el("span", { class: "dim" }, "model card for")); bar.appendChild(ms);
    bar.appendChild(open); bar.appendChild(dl);
    bar.appendChild(el("span", { class: "dim small" }, "includes every formula + intermediate table, and your governance checklist once attached"));
    var frame = el("iframe", { class: "card", title: "Model card" });
    box.appendChild(bar); box.appendChild(frame);
    function show() {
      var m = ms.value;
      if (S.live) {
        var u = "api/card/" + encodeURIComponent(r.run_id) + "/" + encodeURIComponent(m) + "?v=" + Date.now();
        frame.removeAttribute("srcdoc"); frame.src = u; open.href = u; dl.href = u;
      } else {
        var html = (S.replay.cards || {})[m] || "";
        frame.srcdoc = html;
        var blob = URL.createObjectURL(new Blob([html], { type: "text/html" }));
        open.href = blob; dl.href = blob;
      }
      dl.setAttribute("download", m + "_model_card.html");
    }
    ms.addEventListener("change", show); show();
    S.refreshCard = show;
  }

  // ------------------------------------------------------------------ governance
  function checklist() { return S.meta.checklists.filter(function (c) { return c.id === S.checklistId; })[0]; }
  function renderGov() {
    var box = $("#tab-gov"); box.innerHTML = "";
    var c = checklist();
    var bar = el("div", { class: "ins-bar" });
    var sel = el("select", { "aria-label": "checklist" });
    S.meta.checklists.forEach(function (x) { sel.appendChild(el("option", { value: x.id }, esc(x.name))); });
    sel.value = S.checklistId;
    sel.addEventListener("change", function () { S.checklistId = sel.value; S.answers = {}; save(); renderGov(); update(); });
    bar.appendChild(el("span", { class: "dim" }, "» ethikit checklist")); bar.appendChild(sel);
    var attach = el("button", { class: "btn primary", type: "button" }, "attach to model cards");
    var md = el("button", { class: "btn", type: "button" }, "export .md");
    var js = el("button", { class: "btn", type: "button" }, "export answers .json");
    var clr = el("button", { class: "btn", type: "button" }, "clear");
    bar.appendChild(attach); bar.appendChild(md); bar.appendChild(js); bar.appendChild(clr);
    var status = el("span", { class: "dim small" }); bar.appendChild(status);
    box.appendChild(bar);
    box.appendChild(el("p", { class: "prose" }, "Accountability, human oversight and well-being can't be measured from a model and a dataset — no tool can. " +
      "What a team <i>can</i> do is answer a recognised checklist and keep the answers next to the metrics. Source: <a href='" + esc(c.url) + "' target='_blank' rel='noopener'>" +
      esc(c.source) + "</a>. " + esc(c.scope) + " Items are ethikit's condensed paraphrase — use the original for a formal audit."));
    var grid = el("div", { class: "gov-grid" });
    var left = el("div"), right = el("div", { class: "sticky" });
    var sec = null;
    c.items.forEach(function (it) {
      if (it.section !== sec) { sec = it.section; left.appendChild(el("div", { class: "gsec" }, esc(sec))); }
      var a = S.answers[it.id] || {};
      var row = el("div", { class: "gitem" }, "<span class='id'>" + esc(it.id) + "</span>");
      var body = el("div", null, "<div class='q'>" + esc(it.text) + "</div><div class='tags'>→ " + it.principles.map(esc).join(", ") + "</div>");
      var ctl = el("div", { class: "ctl" });
      ["yes", "partial", "no", "n/a"].forEach(function (v) {
        var lab = el("label", { class: v.replace("/", "") }, "");
        var r = el("input", { type: "radio", name: "g-" + it.id, value: v }); r.checked = a.answer === v;
        r.addEventListener("change", function () { set(it.id, { answer: v }); });
        lab.appendChild(r); lab.appendChild(document.createTextNode(v)); ctl.appendChild(lab);
      });
      var note = el("input", { type: "text", placeholder: "evidence / note (who, where, link)", value: a.note || "" });
      note.addEventListener("input", function () { set(it.id, { note: note.value }, true); });
      ctl.appendChild(note); body.appendChild(ctl); row.appendChild(body); left.appendChild(row);
    });
    grid.appendChild(left); grid.appendChild(right); box.appendChild(grid);

    function set(id, patch, quiet) {
      S.answers[id] = Object.assign({}, S.answers[id] || {}, patch);
      save(); summary(); if (!quiet) update();
    }
    function save() { store("answers", { checklist: S.checklistId, answers: S.answers }); }
    function summary() {
      var sum = {};
      c.items.forEach(function (it) {
        var ans = (S.answers[it.id] || {}).answer;
        it.principles.forEach(function (p) {
          var s = sum[p] = sum[p] || { pts: 0, app: 0, ans: 0 };
          if (ans === "n/a") return;
          s.app++; if (ans) s.ans++;
          s.pts += ans === "yes" ? 1 : ans === "partial" ? 0.5 : 0;
        });
      });
      var keys = Object.keys(sum).sort(function (a, b) { return (GOVERNANCE.indexOf(b) >= 0) - (GOVERNANCE.indexOf(a) >= 0); });
      right.innerHTML = "<h3>evidence by principle</h3><p class='dim small'>yes = 1, partial = 0.5, over applicable items. Governance principles first.</p>" +
        keys.map(function (p) {
          var s = sum[p]; var v = s.app ? s.pts / s.app : 0;
          return "<div class='sumrow'><span" + (GOVERNANCE.indexOf(p) >= 0 ? " style='color:var(--gold)'" : "") + ">" + esc(p) + "</span><span class='b'><i style='width:" + (100 * v) + "%'></i></span><span>" + s.ans + "/" + s.app + "</span></div>";
        }).join("") +
        "<div class='note'>Process evidence about the team and the system — shown on the model card next to the metrics, never mixed into the model scores.</div>";
    }
    summary();

    attach.addEventListener("click", function () {
      if (!S.result) { status.textContent = "run an evaluation first"; return; }
      if (!S.live) { status.textContent = "attaching needs the local console (python -m ethikit serve); export .json and pass --checklist instead"; return; }
      fetch("api/checklist/" + encodeURIComponent(S.result.run_id), { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checklist: S.checklistId, answers: clean() }) })
        .then(function (r) { return r.json(); }).then(function (res) {
          if (res.error) throw new Error(res.error);
          status.innerHTML = "<span class='ok'>✓ attached</span> — see the model-card tab";
          S.refreshCard && S.refreshCard();
        }).catch(function (e) { status.innerHTML = "<span class='err'>" + esc(e.message) + "</span>"; });
    });
    md.addEventListener("click", function () { download(S.checklistId + ".md", toMarkdown(c), "text/markdown"); });
    js.addEventListener("click", function () {
      download(S.checklistId + "_answers.json", JSON.stringify({ checklist: S.checklistId, answers: clean() }, null, 2), "application/json");
    });
    clr.addEventListener("click", function () { S.answers = {}; save(); renderGov(); update(); });
  }
  function clean() {
    var out = {};
    for (var k in S.answers) if (S.answers[k].answer) out[k] = { answer: S.answers[k].answer, note: S.answers[k].note || "" };
    return out;
  }
  function toMarkdown(c) {
    var lines = ["# " + c.name, "", "Source: " + c.source + " -- " + c.url, "", "_Condensed, paraphrased items via ethikit; use the original for a formal audit._"];
    var sec = null;
    c.items.forEach(function (it) {
      if (it.section !== sec) { sec = it.section; lines.push("", "## " + sec, ""); }
      var a = S.answers[it.id] || {};
      var mark = { yes: "[x]", partial: "[~]", no: "[ ]", "n/a": "[-]" }[a.answer] || "[ ]";
      lines.push("- " + mark + " **" + it.id + "** " + it.text + " _(" + it.principles.join(", ") + ")_" + (a.note ? " -- " + a.note : ""));
    });
    return lines.join("\n") + "\n";
  }
  function download(name, text, type) {
    var a = el("a", { href: URL.createObjectURL(new Blob([text], { type: type })), download: name });
    document.body.appendChild(a); a.click(); a.remove();
  }

  // ------------------------------------------------------------------ tabs
  function showTab(name) {
    document.querySelectorAll(".tabs button").forEach(function (b) { b.setAttribute("aria-selected", String(b.dataset.tab === name)); });
    ["results", "inspect", "card", "gov"].forEach(function (t) { $("#tab-" + t).hidden = t !== name; });
  }

  boot();
})();

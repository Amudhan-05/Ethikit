/* ethikit theme switcher + terminal title bar.
 *
 * Load in <head> (not deferred) so a remembered choice applies before first paint:
 *   <script src="theme.js" data-path="docs/proposal" data-nav="docs"></script>
 * - No choice stored -> follows the OS light/dark setting.
 * - ☀ / ☾ buttons pin light / dark; pressing the pinned one again returns to "auto".
 * - The choice is shared by every ethikit page on the same origin (localStorage) and
 *   syncs live across open tabs and into embedded model cards.
 * data-nav="docs" adds the docs links; data-nav="none" shows only the toggle.
 *
 * Canonical copy: ethikit/web/theme.js (docs/theme.js is copied by export-site).
 */
(function () {
  var KEY = "ethikit-theme";
  var root = document.documentElement;
  var script = document.currentScript;

  function stored() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function save(v) { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch (e) {} }
  function apply(v) {
    if (v === "light" || v === "dark") root.setAttribute("data-theme", v);
    else root.removeAttribute("data-theme");
    var btns = document.querySelectorAll(".ek-theme button");
    for (var i = 0; i < btns.length; i++) {
      btns[i].setAttribute("aria-pressed", String(btns[i].dataset.mode === v));
    }
    root.dispatchEvent(new CustomEvent("ethikit-theme", { detail: v || "auto" }));
  }
  apply(stored());

  window.addEventListener("storage", function (e) { if (e.key === KEY) apply(e.newValue); });

  function toggle() {
    var el = document.createElement("span");
    el.className = "ek-theme";
    el.setAttribute("role", "group");
    el.setAttribute("aria-label", "Colour theme");
    [["light", "☀", "Light theme"], ["dark", "☾", "Dark theme"]].forEach(function (m) {
      var b = document.createElement("button");
      b.type = "button"; b.dataset.mode = m[0]; b.textContent = m[1];
      b.title = m[2] + " (press again to follow your system setting)";
      b.setAttribute("aria-label", m[2]);
      b.setAttribute("aria-pressed", String(stored() === m[0]));
      b.addEventListener("click", function () {
        var next = stored() === m[0] ? null : m[0];
        save(next); apply(next);
      });
      el.appendChild(b);
    });
    return el;
  }
  window.ethikitThemeToggle = toggle;

  var DOCS = [
    ["index.html", "home"], ["proposal.html", "proposal"], ["understanding.html", "internals"],
    ["field-guide.html", "field-guide"], ["report.html", "sample-report"], ["console/", "console"],
  ];

  function bar() {
    if (!script || script.dataset.bar === "none" || document.querySelector(".ek-bar")) return;
    if (window.self !== window.top) return;  // embedded (e.g. a model card in the console): host has the bar
    var path = script.dataset.path || location.pathname.replace(/^\//, "");
    var nav = script.dataset.nav || "docs";
    var base = script.dataset.base || "";
    if (nav === "console") {
      // the console is served both from the docs site (docs/console/) and locally by
      // `ethikit serve`, where the docs aren't -- link to the published docs then.
      nav = "docs";
      var onSite = /github\.io$/.test(location.hostname) || location.protocol === "file:";
      base = onSite ? "../" : "https://amudhan-05.github.io/Ethikit/";
    }
    var b = document.createElement("header");
    b.className = "ek-bar";
    var dots = '<span class="ek-dots" aria-hidden="true"><i></i><i></i><i></i></span>';
    b.innerHTML = dots + '<span class="ek-path">~/<b>ethikit</b>/' + path + " $</span>";
    var n = document.createElement("nav");
    n.className = "ek-nav";
    n.setAttribute("aria-label", "ethikit pages");
    if (nav === "docs") {
      var here = location.pathname.split("/").pop() || "index.html";
      DOCS.forEach(function (d) {
        var a = document.createElement("a");
        a.href = base + d[0]; a.textContent = d[1];
        if (d[0] === here || (d[0] === "console/" && /\/console\/?/.test(location.pathname))) {
          a.setAttribute("aria-current", "page");
        }
        n.appendChild(a);
      });
    }
    b.appendChild(n);
    b.appendChild(toggle());
    document.body.insertBefore(b, document.body.firstChild);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bar);
  else bar();
})();

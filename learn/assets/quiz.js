/* Retrieval-practice quiz. Usage: <div class="quiz" data-id="L1"></div> + window.QUIZ["L1"] = [...]
 * Each item: {q, options:[...], answer:index, why}. Keep options the same length (no formatting clues).
 * Instant feedback; the explanation appears only after answering. Best score kept locally (optional). */
(function () {
  document.querySelectorAll(".quiz").forEach(function (box) {
    var id = box.dataset.id, items = (window.QUIZ || {})[id] || [], got = 0, answered = 0;
    var score = document.createElement("p"); score.className = "score";
    items.forEach(function (it, qi) {
      var q = document.createElement("div"); q.className = "q";
      var p = document.createElement("p"); p.textContent = (qi + 1) + ". " + it.q; q.appendChild(p);
      var why = document.createElement("div"); why.className = "why"; why.textContent = it.why;
      it.options.forEach(function (o, oi) {
        var b = document.createElement("button"); b.type = "button"; b.className = "opt"; b.textContent = o;
        b.addEventListener("click", function () {
          if (q.classList.contains("done")) return;
          q.classList.add("done"); answered++;
          q.querySelectorAll("button").forEach(function (x, xi) {
            x.disabled = true; if (xi === it.answer) x.classList.add("right");
          });
          if (oi === it.answer) got++; else b.classList.add("wrong");
          score.textContent = got + " / " + answered + " correct" + (answered === items.length ? " — now say each answer out loud without looking." : "");
          if (answered === items.length) { try { localStorage.setItem("learn-" + id, got + "/" + items.length); } catch (e) {} }
        });
        q.appendChild(b);
      });
      q.appendChild(why); box.appendChild(q);
    });
    box.appendChild(score);
  });
})();

"""Rehearse the field guide's demo script: run every step that has a terminal equivalent and
time it against its budget, so nothing surprises you in the room.

    python -m examples.rehearse_demo            # all steps
    python -m examples.rehearse_demo --quick    # skip the slow optional ones

A step's budget is the machine time it may take *while you are talking over it*: the spoken
cue is longer (see docs/field-guide.html). Everything runs in a temporary folder with its own
run history, so nothing in your working directory changes. Exit code 1 if any step fails or
runs over budget.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

# (cue, what, budget seconds, ethikit arguments)
STEPS = [
    ("02", "data profile", 20, ["profile"]),
    ("03", "run 4 models with a requirement", 120,
     ["run", "--models", "logistic_regression,decision_tree,random_forest,naive_bayes",
      "--require", "demographic_parity_diff<=0.15", "--no-save", "--out", "out_run"]),
    ("05", "audit bundle", 120,
     ["run", "--models", "logistic_regression,random_forest", "--no-save", "--out", "out_bundle",
      "--bundle", "audit.zip"]),
    ("05", "verify the bundle", 120, ["bundle", "verify", "audit.zip"]),
    ("06", "deep models + DP (optional, needs torch)", 90,
     ["run", "--family", "deep", "--tools", "performance,membership_inference,captum_ig",
      "--no-save", "--out", "out_deep"]),
]
SPOKEN = [("01", "framing", 45), ("02", "data profile", 60), ("03", "run, require, stability", 150),
          ("04", "inspect + model card", 120), ("05", "audit bundle", 60), ("06", "privacy flex (optional)", 90),
          ("07", "architecture", 60), ("08", "close", 30)]


def _console_boots(budget: float = 10.0):
    from http.server import ThreadingHTTPServer
    from ethikit.server import Handler
    t0 = time.perf_counter()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/api/meta", timeout=budget) as r:
            ok = r.status == 200 and "tools" in json.loads(r.read())
    finally:
        httpd.shutdown(); httpd.server_close()
    return ok, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the optional steps")
    a = ap.parse_args(argv)
    from ethikit.deep import has_torch

    rows = []
    ok, secs = _console_boots()
    rows.append(("00", "console starts and answers", 10, secs, "ok" if ok else "FAILED"))
    with tempfile.TemporaryDirectory(prefix="ethikit-rehearsal-") as tmp:
        import ethikit
        src = os.path.dirname(os.path.dirname(os.path.abspath(ethikit.__file__)))   # works installed or not
        env = {**os.environ, "ETHIKIT_RUNS": os.path.join(tmp, "runs"), "NO_COLOR": "1",
               "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
        for cue, what, budget, args in STEPS:
            if "optional" in what and (a.quick or not has_torch()):
                rows.append((cue, what, budget, None, "skipped" + ("" if a.quick else " (no torch)")))
                continue
            print(f"» ethikit {' '.join(args)}", flush=True)
            t0 = time.perf_counter()
            r = subprocess.run([sys.executable, "-m", "ethikit", *args], cwd=tmp, env=env,
                               capture_output=True, text=True)
            secs = time.perf_counter() - t0
            status = "ok" if r.returncode == 0 else "FAILED"
            if r.returncode != 0:
                print((r.stdout + r.stderr)[-1500:])
            rows.append((cue, what, budget, secs, status))

    print("\ncue  step                                       budget    took   result")
    bad = 0
    for cue, what, budget, secs, status in rows:
        over = secs is not None and secs > budget
        if status == "FAILED" or over:
            bad += 1
        took = "-" if secs is None else f"{secs:6.1f}s"
        print(f"{cue:<4} {what:<42} {budget:>5}s  {took:>7}   {status}{'  OVER BUDGET' if over else ''}")
    total = sum(s for _, _, s in SPOKEN)
    print(f"\nspoken script: {total / 60:.1f} min ({(total - 90) / 60:.1f} without the optional cue); "
          f"cues: " + ", ".join(f"{c} {w} {s}s" for c, w, s in SPOKEN))
    print("ready." if not bad else f"{bad} step(s) failed or ran over budget -- fix before the demo.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""M7.3: the built wheel is complete on its own -- the console's web files and the Model Card
schema are inside it, the `ethikit` entry point works, and the console serves from the installed
package, not from this source tree."""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.slow


def _build_wheel(out: Path) -> Path:
    attempts = []
    if shutil.which("uv"):
        attempts.append(["uv", "build", "--wheel", "-o", str(out), str(ROOT)])
    attempts.append([sys.executable, "-m", "build", "--wheel", "-o", str(out), str(ROOT)])
    attempts.append([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out), str(ROOT)])
    errors = []
    for cmd in attempts:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=out)
        wheels = list(out.glob("ethikit-*.whl"))
        if r.returncode == 0 and wheels:
            return wheels[0]
        errors.append(f"{' '.join(cmd[:3])}: {r.stderr.strip().splitlines()[-1:] if r.stderr else r.returncode}")
    pytest.skip("no way to build a wheel here: " + "; ".join(errors))


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pkg")
    wheel = _build_wheel(tmp)
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    site = tmp / "site"
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        z.extractall(site)                 # a pure-Python wheel installs by unpacking it
    return wheel, names, site, tmp


def _run(site: Path, cwd: Path, code: str) -> subprocess.CompletedProcess:
    # -I: isolated mode, so neither this checkout nor PYTHONPATH is importable -- only the wheel
    boot = f"import sys; sys.path.insert(0, {str(site)!r}); " + code
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-I", "-c", boot], capture_output=True, text=True, cwd=cwd,
                          env=env, timeout=300)


def test_the_wheel_contains_the_console_and_the_schema(installed):
    _, names, _, _ = installed
    for f in ["ethikit/web/index.html", "ethikit/web/console.js", "ethikit/web/console.css",
              "ethikit/web/theme.css", "ethikit/web/theme.js",
              "ethikit/report/schemas/model_card.schema.v0.0.2.json", "ethikit/cli.py"]:
        assert f in names, f
    eps = next(n for n in names if n.endswith("entry_points.txt"))
    assert not any(n.startswith(("tests/", "learn/", "docs/")) for n in names)
    with zipfile.ZipFile(installed[0]) as z:
        assert "ethikit = ethikit.cli:main" in z.read(eps).decode()


def test_the_installed_package_runs_without_the_source_tree(installed):
    _, _, site, tmp = installed
    r = _run(site, tmp, "import ethikit, ethikit.server as s; print(ethikit.__file__); print(s.WEB)")
    assert r.returncode == 0, r.stderr
    assert str(site) in r.stdout and str(ROOT) not in r.stdout
    r = _run(site, tmp, "from ethikit.cli import main; sys.argv=['ethikit','--help']; main()")
    assert "ethikit" in r.stdout and "bundle" in r.stdout, r.stdout + r.stderr


def test_the_installed_console_serves_its_static_files(installed):
    _, _, site, tmp = installed
    code = """
import threading, urllib.request, json
from http.server import ThreadingHTTPServer
from ethikit.server import Handler
httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{httpd.server_address[1]}/'
out = {}
for path in ['', 'console.js', 'theme.css', 'api/meta']:
    with urllib.request.urlopen(base + path) as r:
        out[path] = [r.status, len(r.read())]
print(json.dumps(out))
httpd.shutdown()
"""
    r = _run(site, tmp, code)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert all(v[0] == 200 and v[1] > 100 for v in got.values()), got

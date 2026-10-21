"""Theme shared by every generated HTML artefact (aggregate report, model cards).

Inlines the canonical ethikit/web/theme.css + theme.js -- the same palette,
light/dark auto-switching and ☀/☾ toggle as the docs site and the console --
so a report is still one self-contained file (it is also embedded in the
console and in Streamlit, and opened straight from disk).
"""
from pathlib import Path

_WEB = Path(__file__).resolve().parent.parent / "web"
THEME_CSS = (_WEB / "theme.css").read_text(encoding="utf-8")
THEME_JS = (_WEB / "theme.js").read_text(encoding="utf-8")


def head_assets(path: str) -> str:
    """<style> + <script> for a report <head>; `path` is shown in the title bar."""
    js = THEME_JS.replace("</script", "<\\/script")
    return (f"<style>{THEME_CSS}</style>\n"
            f'<script data-path="{path}" data-nav="none">{js}</script>')


# Element-level styling common to the aggregate report and the model cards.
BASE = """
*{box-sizing:border-box}
body{font-family:var(--font-body);margin:0;color:var(--ink);background:var(--bg);line-height:1.55}
.page{padding:28px 40px 48px}
h1{font-family:var(--font-mono);font-weight:600;letter-spacing:-.01em}
h1::before{content:"$ ";color:var(--accent)}
b,strong{color:var(--ink)}
.sub{color:var(--ink-dim);margin-bottom:28px;font-family:var(--font-mono);font-size:13px}
h2{text-transform:uppercase;letter-spacing:.08em;color:var(--accent);
  font-family:var(--font-mono);font-weight:600}
h2::before{content:"## ";color:var(--ink-faint)}
th{color:var(--ink-dim);font-family:var(--font-mono);font-size:.92em}
tr:hover td,tr:hover th{background:var(--bg-raised)}
.muted{font-size:13px;color:var(--ink-dim)}
.pill{display:inline-block;background:var(--accent-soft);color:var(--accent);
  border:1px solid var(--accent-rule);border-radius:20px;font-family:var(--font-mono)}
.pill.gov{background:var(--gold-soft);color:var(--gold);border-color:var(--gold-rule)}
.pill.other{background:var(--blue-soft);color:var(--blue);border-color:var(--blue-rule)}
.pill.bad{background:var(--red-soft);color:var(--red);border-color:var(--red-rule)}
.note{background:var(--gold-soft);border-left:3px solid var(--gold);color:var(--ink-dim)}
code,pre{background:var(--mono-surface);color:var(--mono-ink);border-radius:4px;
  font-family:var(--font-mono)}
pre{padding:10px 12px;overflow-x:auto;font-size:12px;line-height:1.5;white-space:pre}
"""

"""Dark theme shared by every generated HTML artefact (report, model cards).

Same palette as docs/theme.css, which in turn comes from the project proposal
(ink #17211d, green #2f6f57, clay #c1663b, amber #e3a54a, blue #3d6a8f, purple #6b4d86).
Keep the two in sync if you change a colour.

The reports are single self-contained files (they are also embedded in the Streamlit
demo), so the tokens are inlined rather than linked.
"""

TOKENS = """
:root{
  color-scheme:dark;
  --bg:#0e1511; --bg-raised:#17211d; --bg-sunken:#0a100d;
  --ink:#e6eee9; --ink-dim:#a2b3aa; --ink-faint:#72857b;
  --rule:#25352d; --rule-strong:#35493f;
  --accent:#5fb893; --accent-soft:#17291f; --accent-rule:#2f6f57;
  --gold:#e3a54a; --gold-soft:#2a2113; --gold-rule:#6e5224;
  --red:#e08a62; --red-soft:#2d1b14; --red-rule:#7a4128;
  --blue:#82b4da; --blue-soft:#14222e; --blue-rule:#3d6a8f;
  --purple:#b8a0d6; --purple-soft:#221a2d; --purple-rule:#6b4d86;
  --mono-surface:#121b17; --mono-ink:#c3d4ca;
}
"""

# Element-level styling common to the aggregate report and the model cards.
BASE = TOKENS + """
*{box-sizing:border-box}
body{font-family:"Public Sans",-apple-system,"Segoe UI",Roboto,sans-serif;margin:40px;
  color:var(--ink);background:var(--bg);line-height:1.55}
b,strong{color:var(--ink)}
.sub{color:var(--ink-dim);margin-bottom:28px}
h2{text-transform:uppercase;letter-spacing:.08em;color:var(--accent);
  font-family:"IBM Plex Mono",Consolas,monospace;font-weight:600}
th{color:var(--ink-dim)}
tr:hover td,tr:hover th{background:var(--bg-raised)}
.muted{font-size:13px;color:var(--ink-dim)}
.pill{display:inline-block;background:var(--accent-soft);color:var(--accent);
  border:1px solid var(--accent-rule);border-radius:20px}
.pill.gov{background:var(--gold-soft);color:var(--gold);border-color:var(--gold-rule)}
.pill.other{background:var(--blue-soft);color:var(--blue);border-color:var(--blue-rule)}
.note{background:var(--gold-soft);border-left:3px solid var(--gold);color:var(--ink-dim)}
code{background:var(--mono-surface);color:var(--mono-ink);border-radius:4px;
  font-family:"IBM Plex Mono",Consolas,monospace}
"""

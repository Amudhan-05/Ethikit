"""M7.2: no broken internal links across the README, the project docs and the docs site."""
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

ROOT = Path(__file__).resolve().parent.parent
MARKDOWN = ["README.md", "ROADMAP.md", "TESTING.md", "THIRD_PARTY_NOTICES.md"]
HTML = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.html")) + ["docs/console/index.html"]

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HREF = re.compile(r"""(?:href|src)=["']([^"']+)["']""")
ID = re.compile(r"""\bid=["']([^"']+)["']""")


def _local_targets(path: str, text: str, pattern):
    for raw in pattern.findall(text):
        u = urlparse(raw)
        if u.scheme or raw.startswith(("//", "mailto:", "data:", "javascript:")) or "${" in raw or "' +" in raw:
            continue
        yield raw, unquote(u.path), u.fragment


def _github_anchor(heading: str) -> str:
    h = heading.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


@pytest.mark.parametrize("doc", MARKDOWN + HTML)
def test_every_internal_link_resolves(doc):
    src = ROOT / doc
    text = src.read_text(encoding="utf-8")
    pattern = MD_LINK if doc.endswith(".md") else HREF
    broken = []
    for raw, path, frag in _local_targets(doc, text, pattern):
        target = (src.parent / path).resolve() if path else src
        if path.endswith("/"):
            target = target / "index.html"
        if not target.exists():
            broken.append(f"{raw} (no file {target.relative_to(ROOT) if ROOT in target.parents else target})")
            continue
        if frag and target.suffix == ".html":
            if frag not in ID.findall(target.read_text(encoding="utf-8")):
                broken.append(f"{raw} (no id '{frag}')")
        if frag and target.suffix == ".md":
            heads = [_github_anchor(h) for h in re.findall(r"^#+\s+(.*)$", target.read_text(encoding="utf-8"), re.M)]
            if frag not in heads:
                broken.append(f"{raw} (no heading '{frag}')")
    assert broken == []


def test_the_docs_site_links_to_the_readme_section_that_exists():
    idx = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for frag in re.findall(r"github\.com/[^\"#]+#([\w-]+)", idx):
        heads = [_github_anchor(h) for h in re.findall(r"^#+\s+(.*)$", readme, re.M)]
        assert frag in heads, frag

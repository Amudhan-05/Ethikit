"""ethikit is a model-selection and audit tool, not a ranker. This keeps ranking language from
creeping back: any line that says "best model" / "most ethical" / "winner" must also carry a
qualifier that makes clear it is negated, scoped to a suite/run, or quoting the old claim."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ([ROOT / "README.md"] + list((ROOT / "docs").glob("*.html"))
         + [p for p in (ROOT / "ethikit").rglob("*") if p.suffix in {".py", ".js", ".html"}])
BANNED = re.compile(r"best model|most ethical|\bwinner\b|\bcrown", re.I)
OK_CONTEXT = re.compile(r"\bnot\b|\bno\b|\bnever\b|n't|\bnor\b|instead of|rather than|without|"
                        r"pretend|\bsuite\b|this run|tendenc|\bnone\b|single|\bany\b|\bone\b|"
                        r"old claim|used to", re.I)


def test_no_unqualified_ranking_language():
    offenders = []
    for path in FILES:
        if "console/replay" in str(path) or path.name == "replay.json":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if BANNED.search(line) and not OK_CONTEXT.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:120]}")
    assert not offenders, "unqualified ranking language:\n" + "\n".join(offenders)


def test_domain_result_columns_never_name_a_winner():
    from ethikit.domains.runner import DomainResult
    names = " ".join(f.name for f in DomainResult.__dataclass_fields__.values())
    assert not re.search(r"best|winner|rank", names)

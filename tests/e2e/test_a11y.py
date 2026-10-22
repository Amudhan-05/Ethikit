"""M7.1 accessibility: axe-core finds no serious/critical violations in either theme, the tabs
work from the keyboard with a visible focus ring, and reduced motion stops all animation while
the run still completes. Needs `pip install axe-playwright-python` (bundles axe-core)."""
import pytest

from tests.e2e.test_console import _fast_run, browser  # noqa: F401  (shared fixture)

pytestmark = pytest.mark.e2e
axe_mod = pytest.importorskip("axe_playwright_python.sync_playwright")

TABS = ["results", "inspect", "card", "data", "suite", "history", "gov"]
DOCS = ["docs/", "docs/proposal.html", "docs/understanding.html", "docs/field-guide.html", "docs/report.html"]


def _serious(pg):
    res = axe_mod.Axe().run(pg).response
    return [f"{v['id']} ({v['impact']}): {v['help']} -> {[n['target'] for n in v['nodes'][:3]]}"
            for v in res["violations"] if v["impact"] in ("serious", "critical")]


@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_axe_finds_no_serious_violations(browser, server, scheme):  # noqa: F811
    ctx = browser.new_context(color_scheme=scheme, viewport={"width": 1400, "height": 1000})
    pg = ctx.new_page()
    pg.goto(server + "/")
    pg.wait_for_selector("#models .chip")
    problems = [f"start: {p}" for p in _serious(pg)]
    _fast_run(pg)
    for tab in TABS:
        pg.click(f"[data-tab={tab}]")
        pg.wait_for_timeout(400)
        problems += [f"tab {tab}: {p}" for p in _serious(pg)]
    for doc in DOCS:
        pg.goto(server + "/" + doc)
        problems += [f"{doc}: {p}" for p in _serious(pg)]
    ctx.close()
    assert problems == []


def test_tabs_work_from_the_keyboard_with_a_visible_focus_ring(browser, server):  # noqa: F811
    pg = browser.new_page()
    pg.goto(server + "/")
    pg.wait_for_selector("#models .chip")
    pg.focus("#tabbtn-results")
    pg.keyboard.press("ArrowRight")
    assert pg.evaluate("document.activeElement.id") == "tabbtn-inspect"
    assert pg.get_attribute("#tabbtn-inspect", "aria-selected") == "true"
    assert pg.is_visible("#tab-inspect") and not pg.is_visible("#tab-results")
    assert pg.get_attribute("#tabbtn-inspect", "tabindex") == "0"
    assert pg.get_attribute("#tabbtn-results", "tabindex") == "-1"
    pg.keyboard.press("End")
    assert pg.evaluate("document.activeElement.id") == "tabbtn-gov"
    pg.keyboard.press("ArrowRight")                                   # wraps around
    assert pg.evaluate("document.activeElement.id") == "tabbtn-results"
    assert pg.evaluate("getComputedStyle(document.activeElement).outlineStyle") != "none"
    assert pg.get_attribute("#tab-gov", "aria-labelledby") == "tabbtn-gov"
    # chips are real checkboxes: focusable, and Space toggles them
    chip = pg.locator("#models .chip").first
    before = "on" in (chip.get_attribute("class") or "")
    chip.locator("input").focus()
    pg.keyboard.press("Space")
    assert ("on" in (chip.get_attribute("class") or "")) != before
    # every control in the configure area is reachable with Tab (none has tabindex=-1)
    assert pg.evaluate("document.querySelectorAll('main [tabindex=\"-1\"]:not([role=tab])').length") == 0
    pg.close()


def test_reduced_motion_stops_animation_but_the_run_still_updates(browser, server):  # noqa: F811
    ctx = browser.new_context(reduced_motion="reduce")
    pg = ctx.new_page()
    pg.goto(server + "/")
    pg.wait_for_selector("#models .chip")
    _fast_run(pg)
    assert pg.evaluate("document.getAnimations().filter(a => a.playState === 'running').length") == 0
    assert pg.locator("#tab-results table").count() >= 1           # the result still arrived
    ctx.close()

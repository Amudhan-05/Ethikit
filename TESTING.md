# Testing ethikit

Every milestone in [ROADMAP.md](ROADMAP.md) ships with two kinds of check, and is only done
when **both** pass:

- **Automated** -- commands you run; they print pass/fail. (`pytest` suites arrive in M1; until
  then the "Today" section below is the baseline you can run right now.)
- **Manual / visual** -- a numbered checklist where you look at the console, a report or a model
  card and compare it with what you should see. Each step says *what to do* and *what you must see*.

All checks run locally. Use Python 3.12 (the standard for this project). Start the console in one terminal with
`python -m ethikit serve` and keep it running; open http://127.0.0.1:8765/.

---

## Today (baseline for what is already built)

**Automated (run now)**

| # | Command | Must happen |
|---|---|---|
| T1 | `python -m ethikit list tools` | 9 tools listed; each says available or "not installed" (giskard and drift_detection may be missing on your machine) |
| T2 | `python -m ethikit run --models logistic_regression --tools performance,fairness --out /tmp/ek` | log shows `✓ performance` and `✓ fairness`, then a comparison table and `pareto front:` |
| T3 | `python -m ethikit run --synthetic minority_share=0.15 --models decision_tree --tools aif360 --out /tmp/ek` | `disparate_impact_gap` is printed; no traceback |
| T4 | `python -m ethikit checklist deon --markdown` | a markdown checklist with `[ ]` boxes in 5 sections |
| T5 | `python -m examples.run_phase3_demo` | privacy section ranks `random_forest`/`knn` as leakier than `logistic_regression` / `naive_bayes` |
| T6 | `python -m ethikit run --data nope.csv --target y` | a one-line `error:`, not a stack trace |

**Manual / visual (console)**

| # | Do | You must see |
|---|---|---|
| V1 | Open `/`; click the ☾ then the ☀ button; reload | page switches theme each click and **keeps** your choice after reload; clicking the active button again returns to following your system |
| V2 | In *configure*, tick/untick models | the `»` command line at the bottom rewrites itself immediately |
| V3 | Press **run** | in *execution graph*: model boxes pulse "fitting…" then "evaluating"; tools under each principle go blue then green with a count (`2/3`); dots travel along the edges; the log streams the same events |
| V4 | When finished, look at *results* | a comparison table with heat-shaded cells, a radar chart, a Pareto list; hover a legend name -> that polygon highlights |
| V5 | Change a `--weights` number | the table order and the "weighted" column change **without** re-running |
| V6 | *inspect* tab -> `random_forest` -> `membership_inference` | a formula block, "worked through, this run" steps with real numbers, and two overlapping histograms (members vs non-members) |
| V7 | *inspect* -> `aif360` | "fails/passes the 80% rule" appears in the worked steps |
| V8 | *governance* tab: answer 3 items, click **attach to model cards**, open *model-card* | a "Governance checklist" section on the card with your three answers |
| V9 | Open `/docs/` and each page | same colours and `»` title bar; nav links work offline; dark and light both readable |
| V10 | Narrow the browser to phone width | no sideways scrolling; the configure panel stacks |

---

## M1 -- Trust the foundation

**Automated**
- `pytest tests/test_scoring.py` -- `normalize` puts every metric in [0,1] with 1 = best (incl. lower-is-better metrics); `pareto_front` returns no dominated model; `weighted_score` ignores unknown keys and sums weights to 1.
- `pytest tests/test_evaluators.py` -- for **every registered evaluator** on a tiny synthetic set: values are finite and in a sane range, `mechanism` and `formula` are non-empty, a `trace` is present.
- `pytest tests/test_fallbacks.py` -- numpy fallback and fairlearn agree to 1e-9 on the same predictions.
- `pytest tests/test_loaders.py` -- `prepare_frame`: NaNs imputed, many-level categoricals collapse to "other", numeric sensitive column splits at the median, single-class target raises a clear error, Adult now includes categorical columns.
- `pytest tests/test_checklists.py` -- unanswered items count as "not evidenced", `n/a` excluded, scores in [0,1].
- `pytest tests/e2e/test_console.py` -- Playwright: a full run finishes, weights re-order without a new run, inspect shows a formula, a checklist attaches to the card, no console errors, in both themes.
- CI (GitHub Actions) runs all of the above on Python 3.12 -- green badge in the README.

**Manual / visual**
1. Run `pytest -q`; **see** a summary line with 0 failures.
2. Start the console with a tool deliberately uninstalled (`pip uninstall -y aif360`); **see** that chip greyed with "not installed" and a hint on how to install it; the run still completes.
3. Run on Adult; **see** categorical columns (e.g. `workclass=Private`) among the features in the model card.

## M2 -- Reference suites per domain

**Automated**
- `pytest tests/test_domains.py` -- every suite declares target + sensitive attribute for every dataset; loaders return the declared columns; a dataset that needs a manual download raises a clear message naming the folder.
- `pytest tests/test_domain_runner.py` (small cached fixtures) -- results carry `domain` and `dataset`; aggregation is **within** a domain; fairness metrics are present (the old suite had none); Friedman is skipped, with a reason, when there are too few datasets.
- `pytest tests/test_cv.py` -- `--cv 5x3` yields 15 scores per metric; the spread (std) is reported; seeds reproduce exactly.
- `python -m ethikit domains run census --max-datasets 3` -- prints a family x principle table **with a fairness column** and a spread, and ends with the suite's caveats.
- Language-audit test: a script greps docs and UI strings for "best model" / "most ethical" and fails if one appears without the words "suite" or "this run".

**Manual / visual**
1. Console -> **domain** selector (new): choose *census*; **see** the datasets in the suite, the sensitive attribute for each, and a "caveats" box.
2. Run it; **see** the graph run the sweep dataset by dataset, then a family table with +/- spreads.
3. Compare two domains; **see** an explicit "family ordering changes / does not change" table, not a single winner.
4. Click **use this shortlist on my data**; **see** the configure panel pre-filled with those families and nothing run yet.
5. Open the field guide's rigor table; **see** numbers that match `results.json` from the same run.

## M3 -- Decision support

**Automated**
- `pytest tests/test_constraints.py` -- `--require demographic_parity_diff<=0.1` keeps only passing models and records, for each failure, which constraint failed and by how much.
- `pytest tests/test_sensitivity.py` -- when one model dominates on every principle the "weights needed to change the pick" is infinite/none; for a two-model trade-off it returns the correct crossover weight.
- `pytest tests/test_runs.py` -- save -> list -> load round-trips a run; `diff` lists exactly the changed metrics.
- `pytest tests/test_config.py` -- `ethikit run --config run.json` reproduces the same numbers as the original run (same seed).
- `pytest tests/test_profile.py` -- group sizes and base rates per group match a hand-computed pandas groupby; a feature that predicts the sensitive attribute is flagged as a proxy.

**Manual / visual**
1. Set a constraint in the console; **see** models that fail it in a separate "excluded, and why" list, not silently missing.
2. Open **choice stability**; **see** a plain sentence such as "the pick changes if fairness weight drops below 0.2".
3. Run twice with different seeds, then **compare runs**; **see** a side-by-side diff with the changes highlighted.
4. **Export config**, restart the server, **import config**, run; **see** identical numbers.
5. The new **data profile** step appears before any model runs; **see** group sizes, base rates and a proxy-feature table.

## M4 -- Tool depth

**Automated**
- `pytest tests/test_evaluator_contract.py` -- a single parametrised test over every registered evaluator: has `formula`, `mechanism`, `principle_ids`, a trace with at least one step, values in range, deterministic for a fixed seed. A new tool that breaks the contract fails here.
- `pytest tests/test_mitigation.py` -- fairlearn mitigation improves the target gap on the biased synthetic set and **never** writes into the audit scores (it is returned separately).
- `pytest tests/test_calibration.py` -- a perfectly calibrated toy predictor scores ECE ~ 0; an overconfident one scores higher.
- `python -m ethikit coverage` -- every tool-addressable principle lists **two or more** tools.

**Manual / visual**
1. Flow graph: each tool-addressable principle box shows at least two tool chips.
2. Open a mitigation what-if; **see** it labelled "counterfactual -- not part of the audit" and a before/after fairness vs utility chart.
3. Inspect the calibration tool; **see** a reliability diagram close to the diagonal for a well-calibrated model.
4. Inspect SHAP and the second explainer on the same model; **see** the top features roughly agree, or an honest note where they do not.

## M5 -- Deep learning

**Automated**
- `pytest tests/test_adapters.py` -- sklearn, PyTorch and Keras-style wrappers all expose `fit/predict/predict_proba` and pass the evaluator contract test.
- `pytest tests/test_captum.py` -- integrated-gradients attributions sum (within tolerance) to prediction minus baseline (the completeness property).
- `pytest tests/test_dp.py` -- the DP-trained MLP reports its epsilon; its membership-inference advantage is <= the non-private one on the same data.
- `python -m ethikit run --models mlp,mlp_dp --tools membership_inference,performance` -- runs end to end.

**Manual / visual**
1. Console model list: a **deep** family appears; its advanced panel shows layers / learning rate / epochs.
2. Run `mlp` vs `mlp_dp`; **see** a privacy bar difference on the radar and a utility drop, plus the epsilon in the model card.
3. Inspect Captum; **see** a per-feature bar chart and a "completeness check" line showing the sum matches.

## M6 -- Reporting and governance

**Automated**
- `pytest tests/test_bundle.py` -- the audit bundle contains config, results, report, cards, traces, checklist answers and a README; re-running from its `config.json` reproduces `results.json` exactly.
- `pytest tests/test_modelcard_json.py` -- the exported card validates against the Model Card JSON schema.
- `pytest tests/test_checklists_source.py` -- every checklist records source version and date; item counts match the recorded originals.
- `pytest tests/test_datasheet.py` -- pre-filled sections match the dataset profile numbers.

**Manual / visual**
1. Click **export audit bundle**; open the folder; **see** every file listed in its README.
2. Print-preview a model card (Ctrl/Cmd+P); **see** a clean light layout with no title bar or buttons.
3. Open the datasheet; **see** composition numbers already filled in, motivation/collection sections waiting for you.
4. Compare one checklist item against the original source; **see** the recorded version/date and a note where wording was paraphrased.

## M7 -- Finish and polish

**Automated**
- `pytest tests/e2e/test_a11y.py` -- axe-core finds no serious violations in either theme; every control is reachable by keyboard.
- `pytest tests/test_packaging.py` -- build a wheel, install into a clean venv, `ethikit --help` works and the console serves its static files.
- `pytest tests/test_docs_links.py` -- no broken internal links across README, docs and ROADMAP.

**Manual / visual**
1. Clean-machine rehearsal: clone, follow only the README, reach a working demo with **no edits**; time it.
2. Keyboard only: tab through configure -> run -> tabs; **see** a visible focus ring every time; with "reduce motion" on, the flow graph stops animating but still updates.
3. Run the field guide's demo script start to finish; **see** each cue match the screen and fit the time budget.

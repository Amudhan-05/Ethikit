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

## M1 -- Trust the foundation  *(built; these checks pass today)*

**Automated**

```bash
pip install -r requirements-dev.txt          # once; plus `playwright install chromium` for e2e
python -m pytest -m "not e2e"                # 73 tests, ~7 s: scoring, every tool, fallbacks, loaders, checklists, runner, CLI, server API
python -m pytest -m e2e                      # 11 browser tests, ~30 s: the real console in headless Chromium, dark and light
```

What each file guarantees:
- `tests/test_scoring.py` -- every metric is oriented so 1 = best (lower-is-better flipped); a constant metric is 1.0 not NaN; error rows dropped; the Pareto front never contains a dominated model and keeps ties; weights are normalised and unknown keys ignored.
- `tests/test_evaluators.py` -- **the contract, for every registered tool**: finite values, known principle, a formula, a run-specific mechanism, a trace with worked steps, and **identical results for a fixed seed**. Also: unavailable tools must give an install hint; more noise never makes the model look more stable; regression data runs.
- `tests/test_fallbacks.py` -- the numpy fairness paths agree with fairlearn to 1e-9.
- `tests/test_loaders.py` -- `prepare_frame`: NaNs, positive-label rules, many-level categoricals -> "other", numeric sensitive columns split at the median or a threshold, `--exclude-sensitive`, regression inferred, single-class / unknown-column / too-few-rows errors; Adult now has categorical columns; scaling uses train statistics only; synthetic knobs actually change the data.
- `tests/test_checklists.py` -- every item maps to a real principle; scoring of yes/partial/no/n-a/unanswered; answers-file validation; markdown export.
- `tests/test_runner_session.py` -- event order, per-tool traces, a crashing tool doesn't break the sweep, pre-fitted models are not refit, bad tool/param names give helpful errors, weights drive the order, regression and custom-synthetic runs.
- `tests/test_cli.py` -- flags become the same config the console builds; one-line errors, never a traceback; a full run writes report, cards and results; a checklist reaches the card.
- `tests/test_server.py` -- the real server: static types, docs served locally, **path traversal blocked**, NDJSON run stream, bad config -> error event (server survives), CSV upload then run, card/report endpoints, checklist attach.
- `tests/e2e/test_console.py` -- theme toggle persists and returns to auto; the `ethikit run` line follows clicks; a run lights up the graph; **changing a weight re-orders the table without re-running** and the top row is really the best on that principle; inspect shows a formula and steps; a checklist attaches to the model card; no sideways scroll on a phone; no console errors in either theme.

**These tests were checked to be able to fail**: deliberately breaking normalisation, deleting a tool's formula, and disconnecting the weight re-render each made the matching test fail.

**Bugs the new tests found and that are now fixed**
- The HopSkipJump attack gave different numbers on identical runs (ART builds an unseeded random generator internally), so `--seed` didn't make audits reproducible. It is now seeded for the duration of the attack.
- Adult previously used numeric columns only; it now includes the categorical ones (workclass, education, race, sex, ...).

**Manual / visual**
1. Run `python -m pytest`; **see** `... passed` and no `failed`.
2. Open the console. Hover a greyed tool chip (e.g. *giskard* if not installed); **see** "Not installed. To enable: pip install giskard". Press run anyway; **see** the run complete without it.
3. `python -m ethikit list tools`; **see** the same install hints in the terminal.
4. Run any model, open its model card and scroll to **Tools that did not run**; **see** each missing tool with its install command.
5. With network: `python -m ethikit run --data adult`; **see** `workclass=...`-style features in the model card's Quantitative Analysis, and a fairness gap on `sex`.
6. Run the same command twice with `--seed 5`; **see** byte-identical numbers in `results.json` (compare with `diff`), including `adversarial_perturbation_size`.

## M2 -- Reference suites per domain  *(built; live downloads need YOUR check, see below)*

**Automated** (all offline; none touches your real dataset cache)

```bash
python -m pytest -m "not e2e"     # now ~130 tests
python -m pytest -m e2e           # 17 browser tests
```

- `tests/test_domains.py` -- every domain has caveats and an honest size label (`gap`/`single`/`small`/`suite`); thin suites say so; dataset ids are unique; the census suite is tasks x states + Adult; **hiring is an explicit gap, not a synthetic stand-in**; the ACS adapter builds a dataset from PUMS-shaped data for **all five folktables tasks**; the COMPAS adapter applies ProPublica's cleaning and **drops the leaky COMPAS score and race from the features**; fairlearn adapters name the real columns when a schema differs and report network failure clearly; row caps are deterministic and keep the sensitive attribute aligned; the cache round-trips; regression datasets are refused.
- `tests/test_domain_runner.py` -- profiles are 0-1 for every dataset; **families are averaged inside a family before the suite mean** (a family with more members doesn't count more); leading family + agreement are consistent; real-unit table present; Friedman runs with enough data and is **skipped with a stated reason** when not; **a failing dataset is skipped and reported, not fatal**; if nothing loads, the error lists every reason; thin suites carry an "indicative" note; `--cv` is reproducible; **the shortlist always weighs utility and shows its weights**; `compare_domains` reports agreement and never a winner.
- `tests/test_cv_and_domain_interfaces.py` -- repeated CV in ordinary runs gives mean +/- spread and is reproducible; the graph animates the first fold only; invalid CV / pre-trained-model-with-CV are rejected; the `ethikit domains` CLI; the streaming `/api/domain-run` endpoint (events, skip, result, errors-as-events, server survives).
- `tests/test_language_audit.py` -- fails if "best model" / "most ethical" / "winner" appears anywhere in README, docs, UI or code without a negating or scoping qualifier; result classes can't grow a "winner" field.
- `tests/e2e/test_console.py` -- suite mode hides the my-data rows and rewrites the command; a gap domain disables run and tells you to bring your own data; a suite run shows family evidence, real units, caveats, a skipped dataset greyed out, and the shortlist hands back to my-data mode; `--cv` shows up in the command and the table shows a spread.

**Verified against real data in the build sandbox**: COMPAS (downloaded from GitHub; cleaned to 6,172 rows, the count in ProPublica's published analysis; accuracy about 0.65-0.69, in line with the literature).
**Not verifiable there (no access to census.gov, OpenML, UCI)**: ACS download, Adult, German Credit, credit-card default, bank marketing, diabetes hospital. They are tested offline on shaped fixtures; **please run the live checks below once on your machine**.

**Live checks -- please run these (need internet once; data is cached afterwards)**

| # | Command | Must happen |
|---|---|---|
| L1 | `pip install folktables && python -m ethikit domains run census --max-datasets 1` | downloads one state's ACS file, then a family table; if it fails, the error names the dataset and the fix |
| L2 | `python -m ethikit domains run census --max-datasets 5` | five tasks for the first state all load (`✓ N rows`); positive rates look plausible (income ~ 0.3-0.5, employment ~ 0.4-0.6) |
| L3 | `python -m ethikit domains run lending` | three datasets; **any loader whose columns differ from what we assumed fails with the real column list** -- paste me that message and I fix the one line |
| L4 | `python -m ethikit domains run health` | the diabetes dataset loads; family table with a fairness column |
| L5 | `python -m ethikit domains run justice` | already verified here: `✓ 5000 rows`, accuracy ~0.65-0.69 |
| L6 | `python -m ethikit domains run census --cv 3x2 --max-datasets 3` | slower; every number carries a +/- |
| L7 | run L1 again | instant (served from `~/.cache/ethikit`) |
| L8 | `python -m ethikit domains run hiring` | exits with a one-line message telling you to bring your own data |

**Manual / visual (console)**
1. Open the console, choose `--mode reference suite`; **see** the `--data`, `--estimator`, `--model-file` rows disappear and a **domain** row, a **datasets** list (each marked *verified* or *first run on your machine*) and a yellow **caveats** box appear. The command line at the bottom now starts `ethikit domains run census`.
2. Pick **hiring**; **see** a red `gap` badge, an explanation to bring your own data, and the run button disabled.
3. Pick **justice**, press run; **see** the dataset row go *running…* then *done · 5000 rows*, then the **reference-suite** tab open with: a relative table (0-1), a "real units" table, a "does the ordering hold up?" table, and caveats saying "one dataset: not cross-dataset evidence". Standard deviations and agreement percentages show `–`, not a fake `0.00` / `100%`.
4. Pick **census** with `--max-datasets 3`; **see** three rows light up in turn; the tables now show `±` spreads.
5. Click **use this shortlist on my data**; **see** the page return to *my data & models* with the shortlisted models ticked and nothing run yet.
6. In *my data* mode type `3x2` in `--cv` and run; **see** `±` on every cell of the comparison table and "cross-validation fold i / n" counting up while the flow graph animates once.
7. Search the page text for "best" / "most ethical" / "winner"; **see** none presented as a claim.

## M3 -- Decision support  *(built)*

**Automated**

```bash
python -m pytest tests/test_decision.py tests/test_profile_runs.py      # ~40 tests
python -m pytest -m e2e -k "requirement or profile or history or export"
```

- `tests/test_decision.py` -- requirement parsing (and a helpful error with an example on a typo); failures report **which** requirement and **by how much** (`margin`, "above"/"below"); an unknown metric lists the available ones; a metric reported by two tools must be prefixed (`aif360.theil_index<0.2`); **an unmeasured metric counts as a failure, never a pass**; stability finds the switch point and who takes over, a dominant model is "stable for any weight", single-candidate / no-weight cases explain themselves; reproducible.
- `tests/test_profile_runs.py` -- group sizes and base rates match a hand computation; under-represented groups are flagged; **the proxy screen finds the injected proxy (`proxy_income`) and never counts the sensitive column itself**; missingness comes from the original file; the profile is emitted before any model runs; requirements + stability appear in every summary; a requirement typo fails before training; "nothing feasible" is explained; every run is saved, listed, **reloaded and reproduced identically**; diffs name config and score changes; console uploads are made portable in saved configs; CLI `--require`, `--config`, `--export-config`, `runs diff`, `profile`.
- `tests/test_server.py` -- the run result keeps its own run id (a saved-run id must not overwrite it), carries requirements/stability/profile; the model card shows them; history endpoints list, load and diff runs.
- `tests/e2e/test_console.py` -- requirement chips (and their error message), feasible models listed first and failing ones struck through, data-profile tab, history compare + load-config, export/import config round-trip.

**Bugs found and fixed while building M3**
- A run's weighted order used to put models that *fail* your requirements first; feasible ones now come first and failing rows are struck through.
- The stability check took ~57 s per run: pandas deep-copied the run's traces on every one of thousands of small operations. It now runs on plain arrays (whole test suite 101 s -> 26 s).
- The console logged a 404 for `/favicon.ico` on every load; it now has an icon.

**Manual / visual**

| # | Do | You must see |
|---|---|---|
| M3.1 | `python -m ethikit run --models logistic_regression,decision_tree,naive_bayes --require "demographic_parity_diff<=0.45"` | a **data profile** block first (groups, base rates, proxy AUC ~0.71 and two `!` warnings), then the comparison, then **requirements** with `naive_bayes fails ... 0.61 is 0.16 above the limit`, then **how stable is the order?** with one plain sentence per weighted principle, then `saved as run <id>` |
| M3.2 | `python -m ethikit run --config <that id>` then `python -m ethikit runs diff <id1> <id2>` | `identical: same config, same scores, same metrics` |
| M3.3 | repeat M3.1 with `--seed 7`, then `runs diff` against the first | `split.seed 0 -> 7` and a list of score changes |
| M3.4 | `python -m ethikit run --require "parity<0.1"` | a one-line error naming the metrics that do exist |
| M3.5 | `python -m ethikit profile --synthetic proxy_strength=2.5` | proxy AUC well above 0.8 and `proxy_income` at the top of the mutual-information list |
| M3.6 | `python -m ethikit profile --data your.csv --target ... --sensitive ...` | a "missing values before imputation" section if your CSV has blanks |
| M3.7 | Console: type `demographic_parity_diff<=0.44` in `--require`, press Enter, run | a gold chip, the flag in the command line, a **requirements** column (meets / fails), failing models **struck through and listed last**, an "excluded, and why" box with margins, and the stability sentences |
| M3.8 | Console: open **data-profile** right after pressing run | the profile is there before the models finish |
| M3.9 | Console: run twice with different seeds, open **history**, tick both, **compare selected** | config / scores / raw-metrics tables with green and red deltas |
| M3.10 | **history** -> **load config** on an older run | every control (models, params, tools, weights, seed, cv, requirements) returns to that run's values |
| M3.11 | **export config**, reload the page, **import config** with that file | the command line comes back identical |
| M3.12 | Open a model card of a failing model | "Requirements you set" with *does not meet* and the reason; "What the data says (before any model)" |

## M4 -- Tool depth  *(built)*

**Automated**

```bash
python -m pytest tests/test_m4_tools.py tests/test_evaluators.py
python -m pytest -m e2e -k "whatif or explains or negative"
```

- `tests/test_m4_tools.py` -- every new tool checked against a case whose answer is known:
  calibration ECE < 0.02 for a calibrated model and > 0.2 for an overconfident one, and the
  per-group gap catches a group whose scores are 0.3 too high; worst-group accuracy equals a
  hand computation; permutation importance sees **direct** use of the sensitive attribute and
  reports exactly 0 when it is not an input; attribute inference is small for an honest model and
  large when the output encodes the group; **every tool-addressable principle has >= 2 tools and
  every tool states what it does not measure**; the ART trace states its attack budget; the
  mitigation what-if halves the targeted gap, is reproducible, refuses clearly (kNN can't be
  re-weighted) and **never changes the audit**; the local explanation finds the deciding feature.
- `tests/test_evaluators.py` -- the contract (formula, mechanism, trace, determinism) now covers
  13 tools; with `pip install -r requirements.txt` on Python 3.12 all 13 are available and pass,
  including Giskard, ART and drift detection.
- Server + CLI: `/api/whatif`, `/api/explain`, `ethikit whatif`, `ethikit explain`.
- e2e: the what-if panel is labelled a counterfactual and shows before/after; inspect states
  each tool's limits and explains one row; charts never draw negative heights.

**New tools** (13 in total; each says what it does *not* measure):

| Tool | Principles | Metric(s) |
|---|---|---|
| calibration | safety, fairness | expected_calibration_error; calibration_gap_between_groups |
| subgroup_performance | inclusiveness, fairness | worst_group_accuracy; accuracy_gap_between_groups |
| permutation_importance | explainability, transparency | importance_concentration; sensitive_feature_reliance |
| attribute_inference | privacy | attribute_inference_gain (second privacy attack) |

**Bugs found and fixed while building M4**
- Charts drew negative values as invisible negative-height bars; they now draw below a zero line in red.
- A long "not installed — pip install …" hint widened a tool chip past a phone screen.
- `giskard` 3.x (allowed by the old `>=2.0` pin) pulls in numba, which breaks ART and alibi-detect
  on NumPy 2, and was not deterministic; pinned `giskard<3` (2.19 verified).

**Manual / visual**

| # | Do | You must see |
|---|---|---|
| M4.1 | `python -m ethikit list tools` | 13 tools; with the full install none is "not installed" |
| M4.2 | `python -m ethikit run --trace --models logistic_regression --tools calibration,subgroup_performance,permutation_importance,attribute_inference` | each tool's formula and worked steps, with real numbers |
| M4.3 | `python -m ethikit whatif --models logistic_regression,knn` | a "counterfactual -- NOT part of the audit" header; parity gap falling (about 0.45 -> 0.04) with an accuracy cost (about 0.79 -> 0.74); kNN *skipped* with the reason |
| M4.4 | same with `--method threshold_optimizer` | an extra warning that it uses the sensitive attribute at prediction time |
| M4.5 | `python -m ethikit explain --models decision_tree --row 4` | a signed bar per feature and the method's limits |
| M4.6 | Console: run, scroll **results** to "what if we mitigate?" | a gold *counterfactual* badge; pick a model, method, constraint -> before/after table with green/red changes and caveats; the comparison table above does **not** change |
| M4.7 | Console **inspect**: any tool | a "what this does not measure" box under the maths |
| M4.8 | Console **inspect** -> "explain row" with row 5 | a bar chart; negative contributions in red below a zero line |
| M4.9 | Console **inspect** -> `art_robustness` | the first worked step lists the attack budget |
| M4.10 | Model card -> "How each number was computed" | every tool shows "What this does not measure" |

## M5 -- Deep learning  *(built)*

Setup for the torch checks (the others need nothing extra):
`pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install captum opacus`

**Automated**

| Test | What it guarantees |
|---|---|
| `pytest tests/test_deep.py` | scikit-learn `mlp` is in the **deep** family and passes every tool; the Keras-style adapter (a `.predict` that returns probabilities, 1 or 2 columns) and `adapt()` behave; runs without torch |
| `pytest tests/test_deep_torch.py` (skips without torch) | `torch_mlp` clones and is deterministic for a fixed seed; a raw `nn.Module` and a saved `.pt` file are evaluated and give identical probabilities |
| ... `test_captum_attributions_satisfy_completeness` | integrated gradients add up to f(x) − f(baseline) (mean error < 0.01) |
| ... `test_captum_does_not_apply_to_non_torch_models` | Captum is skipped (not faked) for models without a network |
| ... `test_dp_training_reports_its_budget...` | the DP model records epsilon ≤ target and delta; its membership-inference advantage is at least 0.1 below the plain overfitted network's |
| ... `test_captum_works_on_a_dp_trained_model` | regression: Opacus hooks are removed after training, so Captum runs on DP models |
| `pytest tests/test_evaluators.py` | the contract test now includes `captum_ig` |

Measured on `--synthetic n=500 noise=2.5 n_features=12 seed=7`, network (256, 256), 150 epochs:

| model | epsilon | train acc | test acc | membership-inference advantage |
|---|---|---|---|---|
| plain | -- | 1.000 | 0.620 | 0.407 |
| DP | 8 | 0.717 | 0.680 | 0.100 |
| DP | 1 | 0.680 | 0.673 | 0.053 |

**Manual / visual**

| # | Do this | You should see |
|---|---|---|
| M5.1 | `python -m ethikit list models` | a **deep** family: `mlp`, `mlp_reg`, and (with torch) `torch_mlp`, `torch_mlp_dp` |
| M5.2 | Console without torch installed: open the **deep** family | `mlp` selectable; `torch_mlp`/`torch_mlp_dp` greyed out, hover says the pip command |
| M5.3 | Console with torch: select `torch_mlp`, open **advanced** | hidden_layers, learning_rate, epochs, batch_size, dropout, dp, target_epsilon editable |
| M5.4 | the DP showcase (one command, ~10 s): `python -m ethikit run --synthetic n=500 noise=2.5 n_features=12 seed=7 --seed 7 --models torch_mlp,torch_mlp_dp --model-param torch_mlp.hidden_layers="(256,256)" --model-param torch_mlp.epochs=150 --model-param torch_mlp.learning_rate=0.003 --model-param torch_mlp_dp.hidden_layers="(256,256)" --model-param torch_mlp_dp.epochs=150 --model-param torch_mlp_dp.learning_rate=0.003 --model-param torch_mlp_dp.target_epsilon=1 --tools performance,membership_inference` | `torch_mlp` advantage ≈ 0.41, `torch_mlp_dp` ≈ 0.05; accuracy 0.62 vs 0.67 |
| M5.5 | open `ethikit_out/model_cards/torch_mlp_dp.html` | "differential privacy: epsilon = 1.00 (smaller = stronger guarantee), delta = …" in the privacy section |
| M5.6 | Console: same run (paste the settings, or **import config** after `--export-config`), **inspect** -> `torch_mlp_dp` × `membership_inference` | the first worked step names the DP budget |
| M5.7 | `python -m ethikit run --models torch_mlp --tools captum_ig` then console **inspect** -> `captum_ig` | a per-feature bar chart and a "completeness" line with an error near 0 (≈ 1e-3) |
| M5.8 | Save a network: `python -c "import torch,ethikit.deep as d,ethikit.data as D; from ethikit.data import loaders as L; ds=D.split_dataset(L.make_synthetic_fairness_raw()); torch.save(d.TorchMLPClassifier(epochs=5).fit(ds.X_train,ds.y_train).module_,'net.pt')"` then `python -m ethikit run --model-file net=net.pt --tools performance,captum_ig` | `net` is audited like any zoo model, Captum included |

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

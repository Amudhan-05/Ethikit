# ethikit user guide

How to set up ethikit, run it from the web console and the command line, and what every feature does.
For the checklist of things to verify, see [TESTING.md](TESTING.md). For project status, see [ROADMAP.md](ROADMAP.md).

## 0. What ethikit is, in one paragraph

ethikit helps you **select and audit** machine-learning models on principles of ethicality
(fairness, explainability, safety, privacy, inclusiveness) next to ordinary utility. It runs a battery of
open-source tools against each candidate model, turns every result into a 0–1 score (1 = best), and lets you
weight the principles to rank models. It is **not** a model ranker: it never claims a "best" or "most ethical"
model. It shows evidence, trade-offs, the Pareto front (models nobody beats on every principle), and, for
each tool, what the tool does *not* measure. Everything runs on your machine; there is no hosting and no
account.

## 1. Setup

You need **Python 3.12**.

```bash
git clone https://github.com/AmudhanS-05/Ethikit && cd Ethikit
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # everything (pins numpy<2, giskard>=2,<3, folktables)
pip install -e . --no-deps                 # makes `ethikit` / `python -m ethikit` use this checkout
```

Optional extras:

| You want | Install |
|---|---|
| The test suite | `pip install -r requirements-dev.txt` then `playwright install chromium` |
| Deep learning models (M5) | `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install captum opacus` |

Without PyTorch you will see a harmless "None of PyTorch, TensorFlow... have been found" line from a
dependency; ignore it. Tools or models that need a missing package are shown greyed out with the install command.

Environment variables (both optional): `ETHIKIT_RUNS` (where run history is saved, default `./ethikit_runs`)
and `ETHIKIT_CACHE` (downloaded datasets, default `~/.cache/ethikit`).

## 2. Start the web console

```bash
python -m ethikit serve --open        # http://127.0.0.1:8765 ; the docs are at /docs/
```

Keep the terminal open; closing it stops the console. Use `--port N` if 8765 is taken (see `serve --help`).
The console only listens on your own machine.

The page has two parts: **01 configure** (inputs) and **02 execution graph** (live run), followed by seven result tabs.

## 3. The configure panel

**Every control is a command-line flag.** The `»` line at the bottom always shows the equivalent
`ethikit run ...` command. Click **copy** to paste it into a terminal and get the same run.

### 3.1 `--mode`
- **my data & models**: audit models on your data (a built-in dataset or your CSV). This is the normal mode.
- **reference suite**: for when you have no data of your own yet. Runs model families across several public datasets of a domain and shows evidence for choosing a *starting point* (section 7).

### 3.2 Data (`--data`)
| Choice | What it is |
|---|---|
| `synthetic` | Offline, 4,000 rows, deliberately biased: the group leaks into the label. Best for learning the tool. |
| `synthetic_custom` | Same, with knobs (below). |
| `regression` | Synthetic biased regression data. |
| `adult` | UCI Adult income (sensitive = sex). Needs network once; **silently falls back to synthetic when offline**, so check the dataset name in the results. |
| `german_credit` | Statlog German Credit (sensitive = age ≥ 25). Needs network. |
| your csv | Upload a file, then set the columns below. |

Synthetic knobs (`--synthetic key=value`): `n`=4000, `n_features`=4, `bias`=0.9, `proxy_strength`=0.8,
`minority_share`=0.5, `positive_rate`=0.5, `noise`=1.0, `n_categorical`=0, `seed`=0.
`bias` is how strongly the group affects the label; `proxy_strength` is how well another feature stands in for the group.

### 3.3 Your CSV
- `--target`: the column to predict.
- `--sensitive`: the column whose groups are compared (sex, race, age band, ...).
- `--positive-label`: which target value counts as the favourable outcome (=1).
- `--sensitive-threshold`: for a numeric sensitive column (e.g. age), split groups at this value.
- `--exclude-sensitive`: do not give the sensitive column to the model as a feature. This is **not** the same as fairness: proxies can still carry it. Compare runs with and without.

### 3.4 Models
- `--models`: tick any of the built-in models, grouped in families. Classification: linear (`logistic_regression`), tree_based (`decision_tree`, `random_forest`, `gradient_boosting`), instance_based (`knn`), probabilistic (`naive_bayes`), kernel (`svc`), deep (`mlp`, plus `torch_mlp` and `torch_mlp_dp` with PyTorch). Regression has its own set.
- `--estimator name=import.path`: add any scikit-learn-style class, e.g. `xt=sklearn.ensemble.ExtraTreesClassifier`.
- `--model-file`: audit your own **pre-trained** model (joblib/pickle/`.pt`/`.keras`) as-is. Only load files you trust, because loading can run code.
- **advanced**: every hyperparameter of every selected model and every tool parameter. Values are Python literals (`200`, `0.1`, `None`, `'balanced'`, `(64, 32)`). Changed values become `--model-param` / `--tool-param` flags.

### 3.5 Requirements (`--require`)
Hard limits on a raw metric, for example `demographic_parity_diff<=0.1` or `aif360.theil_index<0.2`.
Type one and press Enter (or **add**). A metric reported by two tools needs the tool prefix. Models that miss a limit are
**listed last and struck through**, with the margin ("0.16 above the limit"). A metric that was not measured counts as a failure.
Typos give an error that lists the metrics that exist.

### 3.6 Tools (`--tools`)
Chips to enable/disable each tool. See section 6 for what each does. A greyed chip is not installed; hover for the install command.

### 3.7 Weights (`--weights`)
How much each principle matters in the weighted score. Defaults: fairness 0.35, explainability 0.20, safety 0.20, privacy 0.15,
inclusiveness 0.10 (utility is reported alongside). Weights are normalised. **Changing them re-orders the results table immediately, without re-running.**
Weights are *your value judgement*; the stability check (section 5.3) shows how sensitive the order is to them.

### 3.8 Run settings
- `--seed`: makes a run reproducible. The same config and seed give identical numbers.
- `--test-size`: fraction held out for evaluation.
- `--cv 5x3`: repeated k-fold (5 folds × 3 repeats) instead of one split. Every score becomes a mean with a spread (±). Slower: runtime multiplies.
- `--no-scale`: skip standardising features (use with a `--model-file` trained on raw data).

### 3.9 Buttons
- **copy**: copies the command line.
- **export config / import config**: save the whole setup as a JSON file and load it later (also: `ethikit run --export-config` / `--config`).
- **⏎ run**: starts the run.

## 4. The execution graph (what you see while it runs)

Data and models on the left, tools grouped by principle, results per principle on the right. Models pulse "fitting…" then "evaluating";
tools go blue (running) then green (done) with a count such as `2/3`; dots travel along the edges. The log underneath streams the same events,
and you can scroll it with the keyboard. With cross-validation the graph animates the first fold only. With "reduce motion" on in your OS, nothing animates but everything still fills in.

## 5. The result tabs

### 5.1 results
- **Comparison table**: one row per model, one column per principle, heat-shaded 0–1 (1 = best). Feasible models first, failing ones struck through.
- **Weighted column**: your weights applied. Re-orders live when you change weights.
- **Radar chart**: one polygon per model; hover a legend name to highlight it.
- **Pareto list**: models that no other model beats on every principle. These are the defensible candidates; the rest are strictly dominated.
- **Requirements**: who meets each limit, who fails and by how much, and an "excluded, and why" box.
- **How stable is the order?**: one plain sentence per weighted principle: how far you can move that weight before the leading model changes, and who takes over. "Stable for any weight" means one model dominates.
- **What if we mitigate?**: a gold *counterfactual* panel. Pick a model, a method, and a constraint to see before/after numbers (for instance the parity gap falling at an accuracy cost). It is **never** part of the audit: the table above does not change. Methods that use the sensitive attribute at prediction time are flagged. Models that cannot be re-weighted (kNN) are skipped with the reason.
- **Export audit bundle (.zip)**: see section 8.

### 5.2 inspect
Pick a model and a tool to see exactly how a number was produced: the **formula**, the **worked steps with this run's real numbers**, charts (for example the member vs non-member histograms for membership inference), and a **"what this does not measure"** box.
The "explain row" box shows, for one test row, which features pushed the decision up (green) or down (red, below the zero line).

### 5.3 model-card
A model card per model: what was measured, quantitative analysis, requirements you set, what the data says before any model, which tools did not run (with install commands), and **how each number was computed**.
Downloads: HTML card, and `.mct.json` (Google Model Card Toolkit format, schema v0.0.2).
Printing (Ctrl/Cmd+P) gives a white, title-bar-free page with every section expanded, ready to save as PDF.

### 5.4 data-profile
Shown before the models finish. Group sizes and base rates (flags under-represented groups and large base-rate gaps), a **proxy screen** (can the other features predict the sensitive attribute? AUC and mutual information; a proxy such as `proxy_income` shows at the top), and missing values before imputation.
Below it is the **datasheet for the dataset** (Gebru et al.): composition answers are pre-filled from the profile; questions only a human can answer (motivation, collection, consent, licence, "should NOT be used for") stay open for you. Type answers, **save answers**, **download .md**; saved answers go into the audit bundle.

### 5.5 reference-suite
The output of suite mode (section 7).

### 5.6 history
Every run is saved under `ethikit_runs/`. Tick two runs and **compare selected** to see what changed: config differences, score differences (green/red deltas), raw-metric differences. **load config** restores every control to that run's values so you can re-run it exactly.

### 5.7 governance
Four checklists to answer yes / partial / no / n-a: Deon (21 items), ALTAI (19), Datasheets (11), NIST AI RMF (13). Each shows its edition, date, and whether it is verbatim or paraphrased. **Only Deon is verified verbatim against upstream; compare ALTAI, Datasheets and NIST items with the originals yourself.**
**Attach to model cards** adds your answers as a "Governance checklist" section on every card.

## 6. The tools (14)

Each reports raw metrics, converts them to 0–1 scores per principle, and states its limits.

| Tool | Principle(s) | What it computes |
|---|---|---|
| performance | utility | Accuracy, macro-F1 and ROC-AUC for classification; R² and RMSE for regression. |
| fairness | fairness, inclusiveness | Demographic-parity difference and equalized-odds gaps between groups (fairlearn, with a numpy fallback). |
| aif360 | fairness, inclusiveness | IBM AIF360 metrics: disparate impact (the "80% rule"), Theil index, etc. Binarises the sensitive attribute (privileged = highest training base rate). |
| calibration | safety, fairness | Expected calibration error, and the calibration gap between groups. Needs probabilities. |
| subgroup_performance | inclusiveness, fairness | Worst-group accuracy and the accuracy gap between groups. |
| explainability | explainability, transparency | How well a simple surrogate tree mimics the model, plus sampled SHAP-style attributions. |
| permutation_importance | explainability, transparency | How much shuffling each feature hurts; importance concentration and **reliance on the sensitive feature**. |
| robustness | safety | Accuracy under random Gaussian noise. A weak test by design. |
| art_robustness | safety | Adversarial robustness with a small query budget (HopSkipJump via IBM ART). The first worked step lists the attack budget. |
| drift_detection | safety | Kolmogorov–Smirnov drift per feature under a manufactured shift (alibi-detect). |
| giskard | fairness, safety | Giskard's automated vulnerability scan, on a sample. Heuristic. |
| membership_inference | privacy | A black-box attack asking "was this row in the training data?". Reports the attacker's advantage. Overfit models leak more. |
| attribute_inference | privacy | Can the attacker recover the sensitive attribute from the model's outputs? A second privacy attack. |
| captum_ig | explainability | Integrated-gradient attributions for PyTorch models, with a completeness check. Needs torch + captum. |

Every tool follows a contract (tested): finite numbers, a formula, a run-specific mechanism, a trace, a stated limit, and identical results for a fixed seed.

## 7. Reference suites (no data of your own)

In suite mode pick a domain: **census** (suite of 16 datasets), **lending** (3), **health** (1), **justice** (1), and **hiring**, which is an explicit *gap* (no public datasets, so the run button is disabled and you are told to bring your own data).
Controls: `--max-datasets`, `--max-rows`, `--states` (census). Each dataset is marked *verified* or *first run on your machine*; downloads are cached.
Output (the **reference-suite** tab): a relative 0–1 table per model **family** (families are averaged inside the family before the suite mean, so a family with more members does not count more), a "real units" table, a "does the ordering hold up?" (Friedman) table, and caveats. A one-dataset domain says it is not cross-dataset evidence. Standard deviations show `–` when there is only one dataset.
**Use this shortlist on my data** returns you to "my data & models" with the shortlisted models ticked and nothing run yet. The suite never names a winner; it gives a shortlist and shows its weights.
CLI: `python -m ethikit domains list | info <id> | run <id>`.

## 8. Audit bundle

`ethikit run ... --bundle audit.zip` or the console button writes one folder/zip containing: config, results, `metrics.csv`, traces, report, model cards, datasheet, checklist, `environment.json`, README and `MANIFEST.sha256`.
`python -m ethikit bundle verify audit.zip` **re-runs the config and compares every metric**. It prints `✓ bundle verified`, or names what changed (checksum and re-run both catch tampering) and exits 1.

## 9. Command line reference

| Command | Purpose |
|---|---|
| `run` | Evaluate models and write report, cards and results. Flags are in section 3. Extra: `--out DIR`, `--trace` (print every tool's formula and worked steps), `--no-save`, `--export-config PATH`, `--config RUN_ID_OR_FILE`, `--checklist ANSWERS.json`, `--datasheet-answers JSON`, `--bundle PATH`, `-v`. |
| `profile` | Data profile only, no training. |
| `whatif` | Mitigation counterfactual (`--models`, `--method exponentiated_gradient\|threshold_optimizer`, `--constraint demographic_parity\|equalized_odds`). Not part of the audit. |
| `explain` | Why a model scored one test row as it did (`--models`, `--row N`). |
| `datasheet` | Dataset datasheet pre-filled from the profile (`--answers` to add yours). |
| `bundle make \| verify` | Build or verify an audit bundle. |
| `runs list \| show \| diff` | Saved-run history. |
| `list datasets \| models \| tools \| params <name> \| checklists` | See what exists. |
| `checklist <id>` | Blank answers template, or `--markdown`. |
| `domains list \| info \| run` | Reference suites. |
| `serve` | Start the web console (`--host`, `--port`, `--open`). |
| `export-site [dir]` | Copy the console into the docs site with a recorded replay. |

Errors are always one line starting with `error:`, never a stack trace.

## 10. A suggested first session (10 minutes)

1. `python -m ethikit serve --open`. Leave defaults, pick `synthetic`, tick `logistic_regression`, `decision_tree`, `random_forest`, `naive_bayes`. Press **⏎ run**.
2. In **results**, find the Pareto list; change a weight and watch the order change without re-running.
3. In **requirements**, add `demographic_parity_diff<=0.3` and run again; see who is struck through and why.
4. **inspect** → `random_forest` → `membership_inference`: read the formula, the worked steps and the histograms. Then `aif360`.
5. **data-profile**: note the proxy warning. Re-run with `--exclude-sensitive` and compare in **history**.
6. **model-card**: open one and read "How each number was computed" and the limits.
7. **governance**: answer three Deon items, attach them, and see them on the card.
8. Export the bundle and run `python -m ethikit bundle verify` on it.

## 11. Things to keep in mind

- Scores are relative to the models in *this* run. A 1.0 means "best among these candidates on this principle", not "good".
- Group fairness metrics on one declared attribute cannot see intersections or explain *why* a gap exists, and demographic parity and equalized odds cannot generally both be zero.
- Offline, `adult` quietly becomes synthetic data. Check the dataset name in the card.
- Python 3.13 runs but the drift tool will not install; keep `giskard` below version 3.

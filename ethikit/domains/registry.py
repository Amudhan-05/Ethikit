"""The reference suites. `get_domains()` builds them fresh so options (e.g. which US states)
can vary. Suites are honest about their size: see Domain.strength.

Status of each loader (see DatasetSpec.verified):
  - COMPAS      : fetched and cleaned from the real file (6,172 rows, ProPublica's own count).
  - ACS census  : adapter checked on a PUMS-shaped fixture; the census.gov download itself is
                  done by folktables and has not been run from the build sandbox.
  - OpenML/fairlearn datasets (Adult, German Credit, credit-card default, bank marketing,
                  diabetes hospital): written against the documented schemas; first real run
                  happens on your machine and fails with a precise message if a column differs.
"""
from __future__ import annotations

from typing import Dict, Sequence

from . import sources as S
from .spec import DatasetSpec, Domain

DEFAULT_STATES = ("CA", "TX", "NY")

_ACS_TITLES = {
    "ACSIncome": "Will income exceed $50k?",
    "ACSEmployment": "Is the person employed?",
    "ACSPublicCoverage": "Does the person have public health insurance?",
    "ACSMobility": "Did the person live at the same address a year ago?",
    "ACSTravelTime": "Is the commute longer than 20 minutes?",
}


def get_domains(states: Sequence[str] = DEFAULT_STATES) -> Dict[str, Domain]:
    census = [
        DatasetSpec(
            id=f"{task}_{st}_2018", title=f"{task} · {st} 2018 — {_ACS_TITLES[task]}",
            loader=S.acs_loader(task, st), verified=False,
            source="US Census ACS PUMS 2018 1-year, via folktables (Ding et al., 'Retiring Adult', NeurIPS 2021)",
            sensitive="sex (Male / Female)",
            note="Features are the task's integer-coded ACS columns used as numbers; sex stays a feature.")
        for st in states for task in S.ACS_TASKS
    ] + [DatasetSpec(
        id="adult_income", title="UCI Adult — will income exceed $50k? (1994 census)",
        loader=S.load_adult_strict, verified=False,
        source="UCI Adult via OpenML", sensitive="sex (Male / Female)",
        note="Included for comparability with older papers; the literature recommends the ACS tasks instead.")]

    return {d.id: d for d in [
        Domain("census", "Income, employment & benefits (US census)",
               "Population-scale tabular data about people: earnings, jobs, insurance, mobility, commuting.",
               census,
               ["US-only, 2018 survey data; a model that suits it may not suit another country or year.",
                "Sensitive attribute is sex only here; race (RAC1P) is available in the data but not audited by default.",
                "The 'Retiring Adult' authors argue Adult is a weak benchmark; ACS tasks are preferred."],
               "Public-sector or financial teams deciding on eligibility, outreach or income-related scoring."),
        Domain("lending", "Lending & credit",
               "Credit risk and marketing decisions about individuals.",
               [DatasetSpec("german_credit", "German Credit — good vs bad credit risk (1,000 rows)",
                            S.load_german_credit, "Statlog German Credit via OpenML (credit-g)",
                            "age >= 25 vs < 25", verified=False,
                            note="Small, dated (1994), single German bank; widely used but widely criticised."),
                DatasetSpec("credit_card_default", "Credit-card default (Taiwan, 2005)",
                            S.load_credit_card_default, "fairlearn.datasets.fetch_credit_card (UCI)",
                            "sex", verified=False),
                DatasetSpec("bank_marketing", "Bank marketing — term-deposit subscription",
                            S.load_bank_marketing, "fairlearn.datasets.fetch_bank_marketing (UCI)",
                            "age (median split)", verified=False,
                            note="A marketing outcome, not a credit decision; included for its sensitive age attribute.")],
               ["Only three datasets: results are indicative, and significance tests have little power.",
                "None is a modern, representative lending dataset; real lending data is rarely public."],
               "Banks, lenders and fintechs choosing a starting model family for credit-style decisions."),
        Domain("health", "Health",
               "Clinical outcomes and care decisions.",
               [DatasetSpec("diabetes_hospital", "Diabetes hospital readmission within 30 days",
                            S.load_diabetes_hospital, "fairlearn.datasets.fetch_diabetes_hospital (UCI)",
                            "race", verified=False)],
               ["One dataset: a starting point, not cross-dataset evidence.",
                "Clinical decisions need clinical validation that no public benchmark replaces."],
               "Hospitals and payers exploring readmission-style risk models."),
        Domain("justice", "Criminal justice",
               "Risk assessment in the justice system.",
               [DatasetSpec("compas", "COMPAS two-year recidivism (Broward County, FL)",
                            S.load_compas, "ProPublica compas-analysis (GitHub)",
                            "race (African-American / Caucasian / Hispanic / Other)", needs="network",
                            note="Cleaned as in ProPublica's analysis; race excluded from features; the COMPAS score itself is not used.")],
               ["One dataset: a starting point, not cross-dataset evidence.",
                "The very notion of risk scoring people in this setting is contested; an audit cannot settle it."],
               "Researchers and agencies studying recidivism-style risk tools."),
        Domain("hiring", "Hiring & HR",
               "Screening and ranking candidates.",
               [],
               ["No standard public benchmark was found, and a synthetic stand-in would only show the "
                "synthetic data's properties, so none is shipped.",
                "Use your own historical data: ethikit run --data candidates.csv --target hired --sensitive gender"],
               "HR teams: bring your own data."),
    ]}

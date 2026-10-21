"""A fake ACS PUMS table shaped like the real one, with just enough variety that every
folktables task finds both classes after its own population filter. For offline tests of the
census adapter only -- the numbers mean nothing."""
import numpy as np
import pandas as pd

COLS = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P", "PINCP",
        "PWGTP", "ESR", "PUBCOV", "MIG", "JWMNP", "DIS", "ESP", "CIT", "MIL", "ANC", "NATIVITY",
        "DEAR", "DEYE", "DREM", "GCL", "MIGSP", "MIGPUMA", "ST", "FER", "POWPUMA", "POVPIP",
        "JWTR", "PUMA"]


def fake_pums(task: str, n: int = 1500, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    df = pd.DataFrame({c: r.integers(1, 6, n) for c in COLS})
    df["AGEP"] = r.integers(18, 34, n)                 # inside every task's age window
    df["WKHP"] = r.integers(1, 60, n)
    df["PWGTP"] = r.integers(1, 200, n)
    df["SEX"] = r.integers(1, 3, n)
    df["JWMNP"] = r.integers(1, 60, n)                 # ~1/3 above the 20-minute cut
    df["PINCP"] = r.integers(100, 100000, n)
    df["ESR"] = r.choice([1, 2, 3], n)
    df["PUBCOV"] = r.integers(1, 3, n)
    df["MIG"] = r.integers(1, 4, n)
    if task == "ACSPublicCoverage":                    # filter keeps PINCP <= 30000
        df["PINCP"] = r.integers(100, 30000, n)
    if task == "ACSTravelTime":                        # filter keeps employed (ESR == 1)
        df["ESR"] = 1
    return df

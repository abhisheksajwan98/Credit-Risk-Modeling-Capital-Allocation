"""Shared fixtures.

Everything is built from the synthetic generator at a size that keeps the whole suite under a
minute. The point of these tests is the *invariants* -- no future information, no post-origination
column reaching a model, no split overlap -- not model quality, so a small sample is sufficient
and a large one would only slow the feedback loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.prepare import prepare_loans
from credit_risk.data.synthetic import SyntheticSpec, write_synthetic_raw
from credit_risk.features.financial import add_derived_features

SMALL = 12_000


@pytest.fixture(scope="session")
def raw_path(tmp_path_factory) -> str:
    directory = tmp_path_factory.mktemp("raw")
    return str(write_synthetic_raw(directory, SyntheticSpec(n_loans=SMALL, seed=11)))


@pytest.fixture(scope="session")
def prepared(raw_path):
    df, report = prepare_loans(raw_path, chunksize=5_000)
    return df.reset_index(drop=True), report


@pytest.fixture(scope="session")
def loans(prepared) -> pd.DataFrame:
    return prepared[0]


@pytest.fixture(scope="session")
def prepare_report(prepared):
    return prepared[1]


@pytest.fixture(scope="session")
def enriched(loans) -> pd.DataFrame:
    return add_derived_features(loans)


@pytest.fixture(scope="session")
def split_frames(loans):
    return {s: loans[loans["split"] == s] for s in ("train", "valid", "test", "monitor", "history")}


@pytest.fixture(scope="session")
def labels(split_frames):
    return {
        s: split_frames[s]["default"].astype(int).to_numpy()
        for s in ("train", "valid", "test")
    }


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)

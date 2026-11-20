import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RECORD = str(ROOT / "data" / "sample" / "e0103")
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def golden():
    return json.loads((FIXTURES / "golden_e0103.json").read_text())


@pytest.fixture(scope="session")
def golden_features():
    return pd.read_csv(FIXTURES / "golden_e0103_features.csv")

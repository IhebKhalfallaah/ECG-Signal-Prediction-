import subprocess

import pytest

from tests.conftest import ROOT


@pytest.mark.parametrize("path", [
    "data/work/ECG_Data/Normal/s0273lre.dat",   # extracted PhysioNet records (notebooks 01-04)
    "data/work/fd3.csv",                         # intermediate feature CSVs
    "notebooks/svm_model_C60_gamma001.pkl",      # model re-exported by notebook 05
    "data/raw/ptb-diagnostic-ecg-database-1.0.0.zip",
])
def test_generated_data_is_gitignored(path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert result.returncode == 0, f"{path} would be committed"

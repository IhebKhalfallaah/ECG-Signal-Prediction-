import numpy as np
import pandas as pd
import pytest
import wfdb

from ecg.features import extract_features_from_ecg
from ecg.plotting import generate_plots
from ecg.predict import predict_majority_class_from_file
from tests.conftest import ROOT, SAMPLE_RECORD

MODEL_PATH = str(ROOT / "models" / "svm_model_C60_gamma001.pkl")


def test_features_match_golden(golden_features):
    df = extract_features_from_ecg(SAMPLE_RECORD)
    pd.testing.assert_frame_equal(df, golden_features, check_exact=False, rtol=1e-9)


def test_prediction_matches_golden(golden):
    assert predict_majority_class_from_file(SAMPLE_RECORD, MODEL_PATH) == golden["prediction"]


def test_plots_match_golden_metrics(golden, tmp_path):
    plots = generate_plots(SAMPLE_RECORD, "e0103", label="x", plots_dir=str(tmp_path))
    for lead in ("metrics_iii", "metrics_v4"):
        assert plots[lead] == pytest.approx(golden[lead], rel=1e-9)
    for key in ("lead_iii", "lead_v4", "segment_iii", "segment_v4"):
        assert (tmp_path / plots[key]).is_file()


def test_missing_leads_returns_error_string(tmp_path):
    wfdb.wrsamp("onelead", fs=250, units=["mV"], sig_name=["I"],
                p_signal=np.random.default_rng(0).normal(size=(5000, 1)),
                fmt=["16"], write_dir=str(tmp_path))
    result = predict_majority_class_from_file(str(tmp_path / "onelead"), MODEL_PATH)
    assert result == "Error: No features extracted"

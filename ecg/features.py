"""Per-beat statistical and entropy features for leads V4 and III (D3)."""
import logging

import numpy as np
import pandas as pd
import wfdb
from scipy.stats import kurtosis, skew

from ecg.detection import pan_tompkins
from ecg.entropy import approximate_entropy, permutation_entropy, sample_entropy
from ecg.filters import preprocess_ecg

logger = logging.getLogger(__name__)

D3_LEAD_NAMES = ("mliii", "iii", "d3")
ANALYSIS_WINDOW_SECONDS = 60

SEGMENT_FEATURES = [
    "A_max", "A_min", "mean", "std", "rms", "kurtosis", "skewness",
    "sample_entropy", "permutation_entropy", "approximate_entropy",
]

# Column order expected by the trained SVM model.
FEATURE_ORDER = (
    ["segment_index"]
    + [f"{name}_v4" for name in SEGMENT_FEATURES]
    + [f"{name}_d3" for name in SEGMENT_FEATURES]
)


def find_lead_indices(sig_names):
    """Return (idx_v4, idx_d3); either is None when the lead is absent."""
    channels = [name.lower() for name in sig_names]
    idx_v4 = next((i for i, ch in enumerate(channels) if ch == "v4"), None)
    idx_d3 = next((i for i, ch in enumerate(channels) if ch in D3_LEAD_NAMES), None)
    return idx_v4, idx_d3


def extract_stat_entropy_features(segment):
    """Ten features of one beat segment, in SEGMENT_FEATURES order."""
    if len(segment) == 0:
        return [0] * len(SEGMENT_FEATURES)
    std = np.std(segment)
    return [
        np.max(segment), np.min(segment), np.mean(segment), std,
        np.sqrt(np.mean(segment ** 2)), kurtosis(segment), skew(segment),
        sample_entropy(segment, m=2, r=0.2 * std),
        permutation_entropy(segment, m=3),
        approximate_entropy(segment, m=2, r=0.2 * std),
    ]


def extract_features_from_ecg(record_path):
    """Feature table (one row per R-R segment of the first minute), or None on failure."""
    try:
        record = wfdb.rdrecord(record_path)
        fs = record.fs
        idx_v4, idx_d3 = find_lead_indices(record.sig_name)
        if idx_v4 is None or idx_d3 is None:
            logger.warning("Leads V4/III not found in %s", record_path)
            return None

        v4 = preprocess_ecg(record.p_signal[:, idx_v4], fs)
        d3 = preprocess_ecg(record.p_signal[:, idx_d3], fs)
        r_peaks, _ = pan_tompkins(v4, fs)
        max_samples = ANALYSIS_WINDOW_SECONDS * fs

        rows = []
        for i in range(len(r_peaks) - 1):
            start, end = r_peaks[i], r_peaks[i + 1]
            if end > max_samples:
                break
            row = {"segment_index": i}
            row.update({f"{k}_v4": v for k, v in zip(SEGMENT_FEATURES, extract_stat_entropy_features(v4[start:end]))})
            row.update({f"{k}_d3": v for k, v in zip(SEGMENT_FEATURES, extract_stat_entropy_features(d3[start:end]))})
            rows.append(row)

        return pd.DataFrame(rows)

    except Exception:
        logger.exception("Error extracting features from %s", record_path)
        return None

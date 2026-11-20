import numpy as np
import wfdb

from ecg.detection import compute_ecg_metrics, pan_tompkins
from ecg.entropy import approximate_entropy, permutation_entropy, sample_entropy
from ecg.filters import preprocess_ecg
from tests.conftest import SAMPLE_RECORD


def _amplitude_at(signal, fs, freq):
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / fs)
    return spectrum[np.argmin(np.abs(freqs - freq))]


def test_preprocess_removes_50hz():
    fs = 250
    t = np.arange(0, 10, 1 / fs)
    raw = np.sin(2 * np.pi * 1.2 * t) + np.sin(2 * np.pi * 50 * t)
    filtered = preprocess_ecg(raw, fs)
    assert _amplitude_at(filtered, fs, 50) < 0.05 * _amplitude_at(raw, fs, 50)


def test_entropy_short_input_returns_zero():
    x = np.array([1.0])
    assert sample_entropy(x, m=2, r=0.2) == 0
    assert permutation_entropy(x, m=3) == 0
    assert approximate_entropy(x, m=2, r=0.2) == 0


def test_pan_tompkins_on_sample():
    record = wfdb.rdrecord(SAMPLE_RECORD)
    fs = record.fs
    v4 = record.p_signal[: 60 * fs, record.sig_name.index("V4")]
    r_peaks, _ = pan_tompkins(preprocess_ecg(v4, fs), fs)
    assert 50 <= len(r_peaks) <= 150


def test_metrics_none_when_few_peaks():
    signal = np.zeros(100)
    peaks = np.array([10, 20])
    assert compute_ecg_metrics(signal, peaks, peaks, peaks, peaks, 250) is None

"""R-peak detection (Pan-Tompkins), P/Q/S/T localisation and interval metrics."""
import logging

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def pan_tompkins(ecg_signal, fs):
    """Detect R-peaks with a simplified Pan-Tompkins algorithm.

    Returns the R-peak indices and the moving-window-integrated signal.
    """
    diff = np.diff(ecg_signal, prepend=ecg_signal[0])
    squared = diff ** 2

    window_size = int(0.15 * fs)  # ~150 ms integration window
    mwa = np.convolve(squared, np.ones(window_size) / window_size, mode="same")

    threshold = np.mean(mwa) * 1.5
    raw_peaks, _ = find_peaks(mwa, height=threshold, distance=int(0.6 * fs))

    # Refine: local maximum of the original signal within +/-100 ms of each peak
    search_radius = int(0.1 * fs)
    r_peaks = []
    for peak in raw_peaks:
        start = max(0, peak - search_radius)
        end = min(len(ecg_signal), peak + search_radius)
        r_peaks.append(start + np.argmax(ecg_signal[start:end]))

    return np.array(r_peaks), mwa


def detect_pqrst(signal, r_peaks, fs):
    """Locate P, Q, S and T waves around each R-peak using fixed search windows."""
    p_waves, q_waves, s_waves, t_waves = [], [], [], []
    p_win = int(0.2 * fs)
    q_win = int(0.05 * fs)
    s_win = int(0.05 * fs)
    t_win = int(0.4 * fs)

    for r in r_peaks:
        # Q wave = minimum before R
        if r - q_win >= 0:
            q_waves.append(r - q_win + np.argmin(signal[r - q_win:r]))

        # S wave = minimum after R
        if r + s_win < len(signal):
            s_waves.append(r + np.argmin(signal[r:r + s_win]))

        # P wave = maximum before Q (or before R if the window is too short)
        p_start = max(0, r - p_win)
        p_segment = signal[p_start:r - q_win] if r - q_win > p_start else signal[p_start:r]
        if len(p_segment) > 0:
            p_waves.append(p_start + np.argmax(p_segment))

        # T wave = maximum after S
        t_start = r + s_win
        t_segment = signal[t_start:min(len(signal), r + t_win)]
        if len(t_segment) > 0:
            t_waves.append(t_start + np.argmax(t_segment))

    return np.array(p_waves), np.array(q_waves), np.array(s_waves), np.array(t_waves)


def compute_ecg_metrics(signal, r_peaks, q_waves, s_waves, t_waves, fs):
    """Interval (seconds) and amplitude metrics measured on the first beats."""
    if len(r_peaks) < 3:
        logger.warning("Not enough R-peaks detected.")
        return None

    r1, r2, r3 = r_peaks[:3]
    q1, q2 = q_waves[0], q_waves[1]
    s2 = s_waves[1]
    t2 = t_waves[1]

    return {
        "duration_segment2": (r3 - r2) / fs,
        "st_duration": (t2 - s2) / fs,
        "s2_relative_to_r2": (s2 - r2) / fs,
        "t2_relative_to_r2": (t2 - r2) / fs,
        "qt_interval": (t2 - q1) / fs,
        "amp_q2": signal[q2],
        "amp_r2": signal[r2],
        "amp_s2": signal[s2],
        "qs_duration": (s2 - q2) / fs,
    }

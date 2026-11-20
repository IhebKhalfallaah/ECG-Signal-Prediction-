"""Diagnostic plots (filtered leads, PQRST detection) and interval metrics for one record."""
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wfdb  # noqa: E402

from ecg.detection import compute_ecg_metrics, detect_pqrst, pan_tompkins  # noqa: E402
from ecg.features import find_lead_indices  # noqa: E402
from ecg.filters import preprocess_ecg  # noqa: E402

logger = logging.getLogger(__name__)

FILTERED_PLOT_SAMPLES = 5000
PQRST_PLOT_SAMPLES = 6000


def _save(fig, plots_dir, filename):
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, filename))
    plt.close(fig)
    return filename


def _plot_filtered(signal, lead_label, title, plots_dir, filename):
    fig = plt.figure(figsize=(15, 10))
    plt.plot(np.arange(FILTERED_PLOT_SAMPLES), signal[:FILTERED_PLOT_SAMPLES], color="blue", label=f"Filtered Lead {lead_label}")
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return _save(fig, plots_dir, filename)


def _analyse_and_plot_pqrst(signal, fs, title, plots_dir, filename):
    r_peaks, _ = pan_tompkins(signal, fs)
    p_waves, q_waves, s_waves, t_waves = detect_pqrst(signal, r_peaks, fs)
    metrics = compute_ecg_metrics(signal, r_peaks, q_waves, s_waves, t_waves, fs)

    n = PQRST_PLOT_SAMPLES
    fig = plt.figure(figsize=(16, 10))
    plt.plot(signal[:n], label="ECG")
    for points, style, name in ((r_peaks, "ro", "R"), (q_waves, "ko", "Q"), (s_waves, "go", "S"),
                                (p_waves, "mo", "P"), (t_waves, "co", "T")):
        visible = points[points < n]
        plt.plot(visible, signal[visible], style, label=name)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return _save(fig, plots_dir, filename), metrics


def generate_plots(record_path, record_name, label, plots_dir):
    """Write four PNGs to plots_dir; return their filenames plus metrics per lead (None if leads missing)."""
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    idx_v4, idx_d3 = find_lead_indices(record.sig_name)
    if idx_v4 is None or idx_d3 is None:
        logger.warning("Leads V4/III not found in %s", record_path)
        return None

    filtered_iii = preprocess_ecg(record.p_signal[:, idx_d3], fs)
    filtered_v4 = preprocess_ecg(record.p_signal[:, idx_v4], fs)

    filtered_title = f"Filtered | {record_name} - {label}"
    lead_iii = _plot_filtered(filtered_iii, "III", filtered_title, plots_dir, f"{record_name}_lead_iii.png")
    lead_v4 = _plot_filtered(filtered_v4, "V4", filtered_title, plots_dir, f"{record_name}_lead_v4.png")

    segment_iii, metrics_iii = _analyse_and_plot_pqrst(
        filtered_iii, fs, f"ECG with PQRST detection III | {record_name} : {label}",
        plots_dir, f"{record_name}_segment_iii.png")
    segment_v4, metrics_v4 = _analyse_and_plot_pqrst(
        filtered_v4, fs, f"ECG with PQRST detection V4 | {record_name} : {label}",
        plots_dir, f"{record_name}_segment_v4.png")

    return {
        "lead_iii": lead_iii,
        "lead_v4": lead_v4,
        "segment_iii": segment_iii,
        "segment_v4": segment_v4,
        "metrics_iii": metrics_iii,
        "metrics_v4": metrics_v4,
    }

"""Butterworth / notch filtering used both for training features and in the app."""
from scipy.signal import butter, filtfilt, iirnotch

HIGHPASS_CUTOFF_HZ = 0.5
NOTCH_FREQ_HZ = 50.0
LOWPASS_CUTOFF_HZ = 100.0


def highpass_filter(signal, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, signal)


def lowpass_filter(signal, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def notch_filter(signal, notch_freq, fs, quality=30):
    nyq = 0.5 * fs
    b, a = iirnotch(notch_freq / nyq, quality)
    return filtfilt(b, a, signal)


def preprocess_ecg(signal, fs):
    """High-pass 0.5 Hz (baseline wander) -> notch 50 Hz (power line) -> low-pass 100 Hz."""
    signal = highpass_filter(signal, HIGHPASS_CUTOFF_HZ, fs)
    signal = notch_filter(signal, NOTCH_FREQ_HZ, fs)
    signal = lowpass_filter(signal, LOWPASS_CUTOFF_HZ, fs)
    return signal

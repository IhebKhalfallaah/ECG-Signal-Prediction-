"""Record-level prediction: classify every beat segment, then take a majority vote."""
from collections import Counter

import joblib

from ecg.features import FEATURE_ORDER, extract_features_from_ecg

LABELS = {0: "Sain", 1: "angine", 2: "infactus"}


def predict_majority_class_from_file(record_path, model_path):
    df = extract_features_from_ecg(record_path)
    if df is None or df.empty:
        return "Error: No features extracted"

    model = joblib.load(model_path)
    predictions = model.predict(df[FEATURE_ORDER])
    majority_class = Counter(predictions).most_common(1)[0][0]
    return LABELS.get(majority_class, str(majority_class))

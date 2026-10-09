#!/usr/bin/env python3
"""Evaluate disaster-message labels using the publisher-provided data splits."""

from __future__ import annotations

import io
import json
import re
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SOURCE_COMMIT = "1657e1c4e11acbfa56241d2b5cd627adbd989ca2"
SOURCE_BASE_URL = (
    "https://raw.githubusercontent.com/rmunro/disaster_response_messages/"
    f"{SOURCE_COMMIT}/"
)
SPLITS = ("training", "validation", "test")
SOURCE_SPLIT_NAMES = {"training": "train", "validation": "validation", "test": "test"}
THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
REGULARIZATION_VALUES = (0.5, 1.0, 2.0, 4.0)
METADATA_COLUMNS = {
    "id",
    "split",
    "message",
    "original",
    "genre",
    "event",
    "actionable_haiti",
    "date_haiti",
}


def load_split(name: str) -> pd.DataFrame:
    url = f"{SOURCE_BASE_URL}disaster_response_{name}.csv"
    request = urllib.request.Request(
        url, headers={"User-Agent": "ResQ-AI-disaster-response-experiment/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        frame = pd.read_csv(io.BytesIO(response.read()), low_memory=False)
    required = {"id", "split", "message", "related", "direct_report"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} split is missing required columns: {sorted(missing)}")
    observed_splits = set(frame["split"].dropna().astype(str).unique())
    expected_split = SOURCE_SPLIT_NAMES[name]
    if observed_splits != {expected_split}:
        raise ValueError(
            f"Expected only split {expected_split!r}; found {sorted(observed_splits)}."
        )
    return frame


def normalize_message(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def remove_duplicate_messages(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Keep an exact normalized message in its earliest supplied split only."""
    seen: set[str] = set()
    prepared: dict[str, pd.DataFrame] = {}
    dropped: dict[str, int] = {}
    for split in SPLITS:
        frame = frames[split].copy()
        if frame["message"].isna().any():
            raise ValueError(f"{split} split contains messages with no text.")
        keys = frame["message"].astype(str).map(normalize_message)
        keep = ~keys.isin(seen)
        prepared[split] = frame.loc[keep].reset_index(drop=True)
        dropped[split] = int((~keep).sum())
        seen.update(keys.loc[keep])
    return prepared, dropped


def get_label_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    if "related" not in columns or "direct_report" not in columns:
        raise ValueError("Dataset does not contain the expected category columns.")
    start = columns.index("related")
    end = columns.index("direct_report")
    if end < start:
        raise ValueError("Dataset category columns are not in the expected order.")
    labels = columns[start : end + 1]
    if set(labels) & METADATA_COLUMNS:
        raise ValueError("Dataset metadata unexpectedly overlaps the label columns.")
    return labels, [name for name in labels if name != "related"]


def validate_labels(
    frames: dict[str, pd.DataFrame], labels: list[str], multilabels: list[str]
) -> None:
    for split, frame in frames.items():
        related = set(pd.to_numeric(frame["related"], errors="raise").unique())
        if not related <= {0, 1, 2}:
            raise ValueError(f"{split} has unexpected `related` labels: {related}.")
        for name in multilabels:
            values = set(pd.to_numeric(frame[name], errors="raise").unique())
            if not values <= {0, 1}:
                raise ValueError(
                    f"{split} has non-binary values for {name!r}: {values}."
                )
        missing = set(labels) - set(frame.columns)
        if missing:
            raise ValueError(f"{split} split is missing labels: {sorted(missing)}")


def multilabel_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    macro_label_indices: list[int],
) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, average=None, zero_division=0
    )
    macro_truth = truth[:, macro_label_indices]
    macro_prediction = prediction[:, macro_label_indices]
    return {
        "subset_accuracy_exact_match": float(accuracy_score(truth, prediction)),
        "labelwise_accuracy": float(np.mean(truth == prediction)),
        "micro_precision": float(
            precision_score(truth, prediction, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(truth, prediction, average="micro", zero_division=0)
        ),
        "micro_f1": float(f1_score(truth, prediction, average="micro", zero_division=0)),
        "macro_f1_trainable_labels": float(
            f1_score(
                macro_truth,
                macro_prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "per_label": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(precision))
        },
    }


def choose_threshold(
    probabilities: np.ndarray,
    truth: np.ndarray,
    macro_label_indices: list[int],
) -> tuple[float, float]:
    best_threshold = THRESHOLDS[0]
    best_f1 = -1.0
    for threshold in THRESHOLDS:
        prediction = (probabilities >= threshold).astype(np.int8)
        macro_f1 = f1_score(
            truth[:, macro_label_indices],
            prediction[:, macro_label_indices],
            average="macro",
            zero_division=0,
        )
        if macro_f1 > best_f1:
            best_threshold, best_f1 = threshold, float(macro_f1)
    return best_threshold, best_f1


def fit_multilabel(
    x_train,
    y_train: np.ndarray,
    x_validation,
    y_validation: np.ndarray,
    macro_label_indices: list[int],
) -> tuple[OneVsRestClassifier, float, list[dict]]:
    candidates = []
    for c_value in REGULARIZATION_VALUES:
        model = OneVsRestClassifier(
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=500,
                solver="liblinear",
                random_state=42,
            ),
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)
        threshold, macro_f1 = choose_threshold(
            probabilities, y_validation, macro_label_indices
        )
        prediction = (probabilities >= threshold).astype(np.int8)
        candidates.append(
            (
                macro_f1,
                model,
                threshold,
                {
                    "C": c_value,
                    "threshold": threshold,
                    "validation_macro_f1_trainable_labels": macro_f1,
                    "validation_metrics": multilabel_metrics(
                        y_validation, prediction, macro_label_indices
                    ),
                },
            )
        )
    selected = max(candidates, key=lambda row: row[0])
    return selected[1], selected[2], [candidate[3] for candidate in candidates]


def fit_related_classifier(
    x_train,
    y_train: np.ndarray,
    x_validation,
    y_validation: np.ndarray,
) -> tuple[LogisticRegression, list[dict]]:
    candidates = []
    labels = [0, 1, 2]
    for c_value in REGULARIZATION_VALUES:
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
            random_state=42,
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        macro_f1 = float(
            f1_score(
                y_validation, prediction, labels=labels, average="macro", zero_division=0
            )
        )
        candidates.append(
            (
                macro_f1,
                model,
                {
                    "C": c_value,
                    "validation_accuracy": float(
                        accuracy_score(y_validation, prediction)
                    ),
                    "validation_macro_f1": macro_f1,
                },
            )
        )
    selected = max(candidates, key=lambda row: row[0])
    return selected[1], [candidate[2] for candidate in candidates]


def run_experiment() -> dict:
    source_frames = {split: load_split(split) for split in SPLITS}
    original_counts = {split: len(frame) for split, frame in source_frames.items()}
    frames, duplicate_counts = remove_duplicate_messages(source_frames)
    labels, binary_labels = get_label_columns(list(frames["training"].columns))
    validate_labels(frames, labels, binary_labels)

    constants = {
        name: int(frames["training"][name].iloc[0])
        for name in binary_labels
        if frames["training"][name].nunique(dropna=False) == 1
    }
    trainable_labels = [name for name in binary_labels if name not in constants]
    if not trainable_labels:
        raise ValueError("No non-constant binary message labels are available.")

    vectorizer = TfidfVectorizer(
        dtype=np.float32,
        max_features=250_000,
        min_df=2,
        ngram_range=(1, 2),
        strip_accents="unicode",
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(frames["training"]["message"].astype(str))
    x_validation = vectorizer.transform(frames["validation"]["message"].astype(str))
    x_test = vectorizer.transform(frames["test"]["message"].astype(str))

    y_train = frames["training"][binary_labels].to_numpy(dtype=np.int8)
    y_validation = frames["validation"][binary_labels].to_numpy(dtype=np.int8)
    y_test = frames["test"][binary_labels].to_numpy(dtype=np.int8)
    macro_label_indices = [binary_labels.index(name) for name in trainable_labels]
    multilabel_model, threshold, multilabel_candidates = fit_multilabel(
        x_train,
        y_train[:, macro_label_indices],
        x_validation[:, :],
        y_validation[:, macro_label_indices],
        list(range(len(macro_label_indices))),
    )
    # Constant training targets are restored at their observed value below.
    multilabel_probabilities = multilabel_model.predict_proba(x_test)
    multilabel_prediction = np.zeros_like(y_test)
    multilabel_prediction[:, macro_label_indices] = (
        multilabel_probabilities >= threshold
    ).astype(np.int8)
    for name, value in constants.items():
        multilabel_prediction[:, binary_labels.index(name)] = value

    related_model, related_candidates = fit_related_classifier(
        x_train,
        frames["training"]["related"].to_numpy(dtype=np.int8),
        x_validation,
        frames["validation"]["related"].to_numpy(dtype=np.int8),
    )
    related_test_truth = frames["test"]["related"].to_numpy(dtype=np.int8)
    related_test_prediction = related_model.predict(x_test)
    related_class_labels = [0, 1, 2]
    related_precision, related_recall, related_f1, related_support = (
        precision_recall_fscore_support(
            related_test_truth,
            related_test_prediction,
            labels=related_class_labels,
            zero_division=0,
        )
    )

    return {
        "source": {
            "repository": "https://github.com/rmunro/disaster_response_messages",
            "commit": SOURCE_COMMIT,
            "license": "Creative Commons Attribution (per source README)",
            "citation": (
                "Robert Munro. 2012. Processing short message communications in "
                "low-resource languages. PhD dissertation, Stanford University."
            ),
            "feature": "message text only",
            "excluded_fields": [
                "id",
                "split",
                "original",
                "genre",
                "event",
                "actionable_haiti",
                "date_haiti",
            ],
        },
        "dataset": {
            "original_rows_by_split": original_counts,
            "duplicate_messages_removed_by_split": duplicate_counts,
            "rows_after_deduplication": {
                split: len(frame) for split, frame in frames.items()
            },
            "vocabulary_size": len(vectorizer.vocabulary_),
            "binary_label_count": len(binary_labels),
            "trainable_binary_label_count": len(trainable_labels),
            "constant_binary_labels_in_training": constants,
            "related_classes": [0, 1, 2],
        },
        "evaluation": {
            "split_policy": (
                "Publisher-provided training/validation/test split; text duplicates "
                "are retained only in the earliest split."
            ),
            "multilabel_target": {
                "labels": binary_labels,
                "selection_metric": "validation macro F1 over trainable labels",
                "selected_regularization": multilabel_candidates[
                    max(
                        range(len(multilabel_candidates)),
                        key=lambda i: multilabel_candidates[i][
                            "validation_macro_f1_trainable_labels"
                        ],
                    )
                ]["C"],
                "selected_threshold": threshold,
                "validation_candidates": multilabel_candidates,
                "test_metrics": multilabel_metrics(
                    y_test, multilabel_prediction, macro_label_indices
                ),
            },
            "related_target": {
                "selection_metric": "validation macro F1",
                "selected_regularization": related_candidates[
                    max(
                        range(len(related_candidates)),
                        key=lambda i: related_candidates[i]["validation_macro_f1"],
                    )
                ]["C"],
                "validation_candidates": related_candidates,
                "test_accuracy": float(
                    accuracy_score(related_test_truth, related_test_prediction)
                ),
                "test_macro_f1": float(
                    f1_score(
                        related_test_truth,
                        related_test_prediction,
                        labels=related_class_labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "test_per_class": {
                    str(label): {
                        "precision": float(related_precision[index]),
                        "recall": float(related_recall[index]),
                        "f1": float(related_f1[index]),
                        "support": int(related_support[index]),
                    }
                    for index, label in enumerate(related_class_labels)
                },
            },
        },
    }


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ARTIFACTS / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = run_experiment()
    metrics["run_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path = output_dir / "experiment_metrics.json"
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Results written to {output_path}")
    print(
        "Multi-label exact-match accuracy: "
        f"{metrics['evaluation']['multilabel_target']['test_metrics']['subset_accuracy_exact_match']:.4f}"
    )
    print(
        "Multi-label elementwise accuracy: "
        f"{metrics['evaluation']['multilabel_target']['test_metrics']['labelwise_accuracy']:.4f}"
    )
    print(
        "Related-label accuracy: "
        f"{metrics['evaluation']['related_target']['test_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()

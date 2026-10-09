#!/usr/bin/env python3
"""Profile NYC EMS dispatch data and run leakage-safe structured experiments."""

from __future__ import annotations

import csv
import gzip
import http.client
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
RESOURCE_URL = "https://data.cityofnewyork.us/resource/76xm-jjuj.json"
METADATA_URL = "https://data.cityofnewyork.us/api/views/76xm-jjuj"
OFFICIAL_URL = "https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj/data"
DATAGOV_URL = "https://catalog.data.gov/dataset/ems-incident-dispatch-data"
USER_AGENT = "ResQ-AI-NYC-EMS-experiment/1.0"
SAMPLE_PER_MONTH = 5_000


class CategoryCardinalityError(ValueError):
    pass


CATEGORICAL_FEATURES = [
    "initial_call_type",
    "borough",
    "incident_dispatch_area",
]
NUMERIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]
SEVERITY_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
RESPONSE_FEATURES = CATEGORICAL_FEATURES + [
    "initial_severity_level_code",
] + NUMERIC_FEATURES
CLASS_TARGET = "initial_severity_level_code"
RESPONSE_TARGETS = {
    "dispatch_response_seconds_qy": "valid_dispatch_rspns_time_indc",
    "incident_response_seconds_qy": "valid_incident_rspns_time_indc",
}

LEAKAGE_AUDIT = [
    {
        "field": "incident_id",
        "severity_experiment": "CONTEXT_ONLY",
        "response_experiment": "CONTEXT_ONLY",
        "used_as_feature": False,
        "reason": "Unique incident identifier; never a predictive input.",
    },
    {
        "field": "incident_datetime",
        "severity_experiment": "VALID_AT_PREDICTION_TIME",
        "response_experiment": "VALID_AT_PREDICTION_TIME",
        "used_as_feature": True,
        "reason": "Call creation time; transformed into hour, weekday, and month cycles.",
    },
    {
        "field": "initial_call_type",
        "severity_experiment": "VALID_AT_PREDICTION_TIME",
        "response_experiment": "VALID_AT_PREDICTION_TIME",
        "used_as_feature": True,
        "reason": "Initial call taker code, available at incident intake.",
    },
    {
        "field": "initial_severity_level_code",
        "severity_experiment": "TARGET",
        "response_experiment": "VALID_AT_PREDICTION_TIME",
        "used_as_feature": True,
        "reason": "Target for coded-severity experiment; prior intake triage input for response-time models.",
    },
    {
        "field": "final_call_type",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "POST_DISPATCH",
        "used_as_feature": False,
        "reason": "May reflect information learned after initial dispatch; excluded.",
    },
    {
        "field": "final_severity_level_code",
        "severity_experiment": "TARGET_DERIVED",
        "response_experiment": "POST_DISPATCH",
        "used_as_feature": False,
        "reason": "Final severity can be revised using later information; excluded.",
    },
    {
        "field": "first_assignment_datetime",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "TARGET_DERIVED",
        "used_as_feature": False,
        "reason": "Occurs after intake; dispatch response time derives from assignment timing.",
    },
    {
        "field": "valid_dispatch_rspns_time_indc",
        "severity_experiment": "TARGET_DERIVED",
        "response_experiment": "TARGET_DERIVED",
        "used_as_feature": False,
        "reason": "Used only to filter records with valid dispatch-response labels.",
    },
    {
        "field": "dispatch_response_seconds_qy",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "TARGET",
        "used_as_feature": False,
        "reason": "Observed response-duration label; never a feature.",
    },
    {
        "field": "first_activation_datetime",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "POST_DISPATCH",
        "used_as_feature": False,
        "reason": "Resource activation occurs after assignment.",
    },
    {
        "field": "first_on_scene_datetime",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "TARGET_DERIVED",
        "used_as_feature": False,
        "reason": "Outcome timestamp; incident response duration is derived from it.",
    },
    {
        "field": "valid_incident_rspns_time_indc",
        "severity_experiment": "TARGET_DERIVED",
        "response_experiment": "TARGET_DERIVED",
        "used_as_feature": False,
        "reason": "Used only to filter records with valid incident-response labels.",
    },
    {
        "field": "incident_response_seconds_qy",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "TARGET",
        "used_as_feature": False,
        "reason": "Observed incident-to-scene duration; never a feature.",
    },
    {
        "field": "incident_travel_tm_seconds_qy",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "LEAKAGE",
        "used_as_feature": False,
        "reason": "Post-activation travel duration overlaps the response-time outcome.",
    },
    {
        "field": "first_to_hosp_datetime",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "POST_DISPATCH",
        "used_as_feature": False,
        "reason": "Occurs after dispatch and transport.",
    },
    {
        "field": "first_hosp_arrival_datetime",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "POST_DISPATCH",
        "used_as_feature": False,
        "reason": "Occurs after dispatch and transport.",
    },
    {
        "field": "incident_close_datetime",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "POST_DISPATCH",
        "used_as_feature": False,
        "reason": "Incident closure is an operational outcome.",
    },
    {
        "field": "held_indicator",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "Timing/availability at intake is not documented; excluded.",
    },
    {
        "field": "incident_disposition_code",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "POST_DISPATCH",
        "used_as_feature": False,
        "reason": "Outcome/disposition is known after response; excluded.",
    },
    {
        "field": "borough",
        "severity_experiment": "VALID_AT_PREDICTION_TIME",
        "response_experiment": "VALID_AT_PREDICTION_TIME",
        "used_as_feature": True,
        "reason": "Aggregated incident location available at intake.",
    },
    {
        "field": "incident_dispatch_area",
        "severity_experiment": "VALID_AT_PREDICTION_TIME",
        "response_experiment": "VALID_AT_PREDICTION_TIME",
        "used_as_feature": True,
        "reason": "Dispatch-area location input; used as an intake-time area feature.",
    },
    {
        "field": "zipcode",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "Aggregated location; provenance/timing not established, excluded.",
    },
    {
        "field": "policeprecinct",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "Aggregated location; provenance/timing not established, excluded.",
    },
    {
        "field": "citycouncildistrict",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "Aggregated location; provenance/timing not established, excluded.",
    },
    {
        "field": "communitydistrict",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "Aggregated location; provenance/timing not established, excluded.",
    },
    {
        "field": "communityschooldistrict",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "Aggregated location; provenance/timing not established, excluded.",
    },
    {
        "field": "congressionaldistrict",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "Aggregated location; provenance/timing not established, excluded.",
    },
    {
        "field": "reopen_indicator",
        "severity_experiment": "POST_DISPATCH",
        "response_experiment": "POST_DISPATCH",
        "used_as_feature": False,
        "reason": "Reopen status is an after-the-fact operational outcome.",
    },
    {
        "field": "special_event_indicator",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "Availability at call intake is not documented; excluded.",
    },
    {
        "field": "standby_indicator",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "May reflect later resource handling; excluded.",
    },
    {
        "field": "transfer_indicator",
        "severity_experiment": "UNKNOWN",
        "response_experiment": "UNKNOWN",
        "used_as_feature": False,
        "reason": "May reflect later resource handling; excluded.",
    },
]


def get_json(url: str, timeout: int = 180) -> object:
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(
                    f"NYC Open Data request failed for {url}: HTTP {exc.code}"
                ) from exc
            detail = f"HTTP {exc.code}"
        except (
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            if attempt == 2:
                raise RuntimeError(
                    f"NYC Open Data request failed for {url}: {exc}"
                ) from exc
            detail = str(exc)
        delay = 2**attempt
        print(
            f"Transient NYC Open Data error ({detail}); retrying in {delay}s.",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)
    raise RuntimeError("NYC Open Data request exhausted all retry attempts.")


def query(params: dict[str, str]) -> object:
    return get_json(f"{RESOURCE_URL}?{urllib.parse.urlencode(params)}")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot write an empty CSV without field names: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_dataset() -> dict:
    metadata = get_json(METADATA_URL)
    columns = metadata["columns"]
    field_names = [column["fieldName"] for column in columns]
    categorical_fields = [
        "initial_call_type",
        "initial_severity_level_code",
        "final_call_type",
        "final_severity_level_code",
        "valid_dispatch_rspns_time_indc",
        "valid_incident_rspns_time_indc",
        "held_indicator",
        "incident_disposition_code",
        "borough",
        "incident_dispatch_area",
        "reopen_indicator",
        "special_event_indicator",
        "standby_indicator",
        "transfer_indicator",
    ]
    timestamp_fields = [
        column["fieldName"]
        for column in columns
        if column["dataTypeName"] == "calendar_date"
    ]
    numeric_fields = [
        column["fieldName"]
        for column in columns
        if column["dataTypeName"] == "number"
    ]

    coverage_exprs = ["count(*) as total_records"]
    coverage_exprs += [
        f"count({column['fieldName']}) as {column['fieldName']}_nonnull"
        for column in columns
    ]
    coverage_exprs += [
        f"count(distinct {field}) as {field}_distinct"
        for field in ("incident_id", "initial_call_type", "initial_severity_level_code")
    ]
    coverage = query({"$select": ",".join(coverage_exprs)})[0]
    total_records = int(coverage["total_records"])

    year_distribution = query(
        {
            "$select": "date_extract_y(incident_datetime) as year,count(*) as records",
            "$group": "year",
            "$order": "year",
        }
    )
    monthly_distribution = query(
        {
            "$select": "date_extract_y(incident_datetime) as year,date_extract_m(incident_datetime) as month,count(*) as records",
            "$group": "year,month",
            "$order": "year,month",
        }
    )

    category_rows = []
    category_counts = {}
    for field in categorical_fields:
        groups = query(
            {
                "$select": f"{field},count(*) as records",
                "$group": field,
                "$order": "records DESC",
                "$limit": "1000",
            }
        )
        category_counts[field] = len(groups)
        for group in groups:
            category_rows.append(
                {
                    "field": field,
                    "value": group.get(field, ""),
                    "records": group["records"],
                }
            )

    timestamp_summary_exprs = []
    for field in timestamp_fields:
        timestamp_summary_exprs.extend(
            (f"min({field}) as {field}_min", f"max({field}) as {field}_max")
        )
    timestamp_summary = query({"$select": ",".join(timestamp_summary_exprs)})[0]

    numeric_summary_exprs = []
    for field in numeric_fields:
        numeric_summary_exprs.extend(
            (
                f"min({field}) as {field}_min",
                f"max({field}) as {field}_max",
                f"avg({field}) as {field}_mean",
            )
        )
    numeric_summary = query({"$select": ",".join(numeric_summary_exprs)})[0]

    duplicate_summary = query(
        {
            "$select": "count(*) as records,count(distinct incident_id) as unique_incident_ids"
        }
    )[0]
    distinct_ids = int(duplicate_summary["unique_incident_ids"])
    duplicate_summary["duplicate_incident_ids"] = total_records - distinct_ids

    inspection = {
        "dataset_name": metadata["name"],
        "dataset_id": metadata["id"],
        "source_description": metadata["description"],
        "rows_updated_at_epoch": metadata.get("rowsUpdatedAt"),
        "rows_updated_by": metadata.get("rowsUpdatedBy"),
        "record_count": total_records,
        "first_incident_datetime": timestamp_summary.get("incident_datetime_min"),
        "last_incident_datetime": timestamp_summary.get("incident_datetime_max"),
        "duplicate_summary": duplicate_summary,
        "non_null_counts": coverage,
        "year_distribution": year_distribution,
        "monthly_distribution": monthly_distribution,
        "categorical_cardinalities": category_counts,
        "timestamp_summary": timestamp_summary,
        "numeric_summary": numeric_summary,
        "columns": columns,
        "categorical_fields": categorical_fields,
        "timestamp_fields": timestamp_fields,
        "numeric_fields": numeric_fields,
    }
    return {
        "metadata": metadata,
        "columns": columns,
        "inspection": inspection,
        "category_rows": category_rows,
        "monthly_counts": {
            (int(row["year"]), int(row["month"])): int(row["records"])
            for row in monthly_distribution
        },
    }


def window_bounds(year: int, month: int) -> tuple[str, str]:
    start = f"{year:04d}-{month:02d}-01T00:00:00"
    if month == 12:
        end = f"{year + 1:04d}-01-01T00:00:00"
    else:
        end = f"{year:04d}-{month + 1:02d}-01T00:00:00"
    return start, end


def fetch_model_sample(monthly_counts: dict[tuple[int, int], int]) -> pd.DataFrame:
    fields = [
        "incident_id",
        "incident_datetime",
        "initial_call_type",
        "initial_severity_level_code",
        "final_call_type",
        "final_severity_level_code",
        "first_assignment_datetime",
        "valid_dispatch_rspns_time_indc",
        "dispatch_response_seconds_qy",
        "first_on_scene_datetime",
        "valid_incident_rspns_time_indc",
        "incident_response_seconds_qy",
        "incident_travel_tm_seconds_qy",
        "borough",
        "incident_dispatch_area",
    ]
    records = []
    for year in (2024, 2025):
        for month in range(1, 13):
            count = monthly_counts.get((year, month), 0)
            if count < SAMPLE_PER_MONTH:
                raise RuntimeError(
                    f"Not enough records to sample {year}-{month:02d}: {count}"
                )
            start, end = window_bounds(year, month)
            offset = (count - SAMPLE_PER_MONTH) // 2
            result = query(
                {
                    "$select": ",".join(fields),
                    "$where": (
                        f'incident_datetime >= "{start}" AND '
                        f'incident_datetime < "{end}"'
                    ),
                    "$order": "incident_datetime,incident_id",
                    "$limit": str(SAMPLE_PER_MONTH),
                    "$offset": str(offset),
                }
            )
            if len(result) != SAMPLE_PER_MONTH:
                raise RuntimeError(
                    f"Expected {SAMPLE_PER_MONTH} rows for {year}-{month:02d} "
                    f"at offset {offset}; received {len(result)}."
                )
            records.extend(result)
            print(
                f"Fetched {len(records):,} of 120,000 sample rows "
                f"through {year}-{month:02d}.",
                flush=True,
            )
    sample = pd.DataFrame.from_records(records)
    if sample["incident_id"].duplicated().any():
        duplicates = int(sample["incident_id"].duplicated().sum())
        raise RuntimeError(f"Stratified sample contains {duplicates} duplicate IDs.")
    return sample


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    timestamps = pd.to_datetime(result["incident_datetime"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Sample contains invalid incident_datetime values.")
    hour = timestamps.dt.hour + timestamps.dt.minute / 60
    weekday = timestamps.dt.dayofweek
    month = timestamps.dt.month - 1
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)
    for field in set(CATEGORICAL_FEATURES + ["initial_severity_level_code"]):
        result[field] = result[field].fillna("UNKNOWN").astype(str)
    return result


def make_logistic_pipeline(c_value: float) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", preprocessing),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=300,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def encode_for_histogram(
    train: pd.DataFrame,
    other_frames: list[pd.DataFrame],
    categorical_fields: list[str],
    numeric_fields: list[str],
) -> tuple[np.ndarray, list[np.ndarray], list[bool]]:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    train_categories = encoder.fit_transform(train[categorical_fields])
    other_categories = [
        encoder.transform(frame[categorical_fields]) for frame in other_frames
    ]
    cardinalities = [len(categories) for categories in encoder.categories_]
    if any(size > 255 for size in cardinalities):
        raise CategoryCardinalityError(
            "Native categorical gradient boosting supports at most 255 categories "
            f"per feature; observed {cardinalities}."
        )
    train_numeric = train[numeric_fields].to_numpy(dtype=float)
    other_numeric = [
        frame[numeric_fields].to_numpy(dtype=float) for frame in other_frames
    ]
    feature_mask = [True] * len(categorical_fields) + [False] * len(numeric_fields)
    train_encoded = np.concatenate((train_categories, train_numeric), axis=1)
    other_encoded = [
        np.concatenate((categories, numeric), axis=1)
        for categories, numeric in zip(other_categories, other_numeric)
    ]
    return train_encoded, other_encoded, feature_mask


def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    probabilities: np.ndarray | None = None,
) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class_metrics": {
            label: report.get(label, {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            for label in labels
        },
        "class_distribution": {
            "actual_test": y_true.value_counts().sort_index().to_dict(),
            "predicted_test": pd.Series(y_pred).value_counts().sort_index().to_dict(),
        },
    }
    if probabilities is not None:
        confidence = probabilities.max(axis=1)
        result["confidence"] = {
            "interpretation": "Raw model confidence; not calibrated.",
            "mean_max_probability": float(np.mean(confidence)),
            "median_max_probability": float(np.median(confidence)),
            "p10_max_probability": float(np.quantile(confidence, 0.10)),
            "fraction_below_0_5": float(np.mean(confidence < 0.5)),
            "selective_accuracy": {
                str(threshold): {
                    "coverage": float(np.mean(confidence >= threshold)),
                    "accuracy_on_covered": (
                        float(np.mean(y_true.to_numpy()[confidence >= threshold] == y_pred[confidence >= threshold]))
                        if np.any(confidence >= threshold)
                        else None
                    ),
                }
                for threshold in (0.5, 0.7, 0.9)
            },
        }
    return result


def run_severity_experiment(
    data: pd.DataFrame,
    full_source_label_counts: dict[str, int] | None = None,
) -> tuple[dict, list[dict]]:
    processed = add_time_features(data)
    train = processed[processed["incident_datetime"].str.startswith("2024")].copy()
    validation = processed[
        processed["incident_datetime"].str.startswith("2025")
        & (pd.to_datetime(processed["incident_datetime"]).dt.month <= 6)
    ].copy()
    test = processed[
        processed["incident_datetime"].str.startswith("2025")
        & (pd.to_datetime(processed["incident_datetime"]).dt.month >= 7)
    ].copy()
    if not len(train) or not len(validation) or not len(test):
        raise RuntimeError("Train/validation/test split is empty.")

    y_train = train[CLASS_TARGET].astype(str)
    y_validation = validation[CLASS_TARGET].astype(str)
    y_test = test[CLASS_TARGET].astype(str)
    labels = sorted(set(y_train) | set(y_validation) | set(y_test))
    candidates = []
    validation_results = []

    dummy = DummyClassifier(strategy="most_frequent")
    dummy_train = np.zeros((len(train), 1))
    dummy_validation = np.zeros((len(validation), 1))
    dummy.fit(dummy_train, y_train)
    dummy_validation_pred = dummy.predict(dummy_validation)
    dummy_metrics = classification_metrics(y_validation, dummy_validation_pred, labels)
    validation_results.append(
        {
            "model": "majority_class_baseline",
            "validation_metrics": dummy_metrics,
            "hyperparameters": {"strategy": "most_frequent"},
        }
    )
    candidates.append(("majority_class_baseline", dummy_metrics["macro_f1"], dummy))

    for c_value in (0.1, 1.0, 10.0):
        model = make_logistic_pipeline(c_value)
        model.fit(train[SEVERITY_FEATURES], y_train)
        validation_pred = model.predict(validation[SEVERITY_FEATURES])
        validation_probabilities = model.predict_proba(validation[SEVERITY_FEATURES])
        metrics = classification_metrics(
            y_validation, validation_pred, labels, validation_probabilities
        )
        name = f"onehot_logistic_C_{c_value:g}"
        validation_results.append(
            {
                "model": name,
                "validation_metrics": metrics,
                "hyperparameters": {
                    "C": c_value,
                    "class_weight": "balanced",
                    "max_iter": 300,
                    "solver": "lbfgs",
                },
            }
        )
        candidates.append((name, metrics["macro_f1"], model))

    try:
        x_train, (x_validation, x_test), categorical_mask = encode_for_histogram(
            train,
            [validation, test],
            CATEGORICAL_FEATURES,
            NUMERIC_FEATURES,
        )
        model = HistGradientBoostingClassifier(
            categorical_features=categorical_mask,
            class_weight="balanced",
            max_iter=120,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42,
        )
        model.fit(x_train, y_train)
        validation_pred = model.predict(x_validation)
        validation_probabilities = model.predict_proba(x_validation)
        metrics = classification_metrics(
            y_validation, validation_pred, labels, validation_probabilities
        )
        name = "hist_gradient_boosting_structured"
        validation_results.append(
            {
                "model": name,
                "validation_metrics": metrics,
                "hyperparameters": {
                    "max_iter": 120,
                    "max_leaf_nodes": 31,
                    "min_samples_leaf": 50,
                    "l2_regularization": 1.0,
                    "class_weight": "balanced",
                    "early_stopping": True,
                    "random_state": 42,
                },
            }
        )
        candidates.append((name, metrics["macro_f1"], model))
    except CategoryCardinalityError as exc:
        validation_results.append(
            {
                "model": "hist_gradient_boosting_structured",
                "status": "not_run",
                "reason": str(exc),
            }
        )
        x_train = x_validation = x_test = categorical_mask = None

    selected_name, _, _ = max(candidates, key=lambda item: item[1])
    test_candidate_metrics = {}
    selected_test_pred = None
    selected_probabilities = None
    for name, _, model in candidates:
        if name == "majority_class_baseline":
            predictions = model.predict(np.zeros((len(test), 1)))
            probabilities = None
        elif name.startswith("onehot_logistic"):
            predictions = model.predict(test[SEVERITY_FEATURES])
            probabilities = model.predict_proba(test[SEVERITY_FEATURES])
        else:
            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)
        test_candidate_metrics[name] = classification_metrics(
            y_test, predictions, labels, probabilities
        )
        if name == selected_name:
            selected_test_pred = predictions
            selected_probabilities = probabilities
    test_pred = selected_test_pred
    test_metrics = test_candidate_metrics[selected_name]

    test_metrics["class_distribution"] = {
        "train": y_train.value_counts().sort_index().to_dict(),
        "validation": y_validation.value_counts().sort_index().to_dict(),
        "test": y_test.value_counts().sort_index().to_dict(),
        "predicted_test": pd.Series(test_pred).value_counts().sort_index().to_dict(),
    }
    result = {
        "target": CLASS_TARGET,
        "prediction_unit": "NYC initial EMS severity code; not mapped to ResQ severity labels.",
        "features": SEVERITY_FEATURES,
        "split": {
            "train": "2024 calendar year",
            "validation": "2025-01 through 2025-06",
            "test": "2025-07 through 2025-12",
            "counts": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
        },
        "validation_candidates": validation_results,
        "full_source_label_counts": full_source_label_counts or {},
        "source_labels_not_observed_in_sample": sorted(
            set(full_source_label_counts or {}) - set(labels)
        ),
        "selected_model": selected_name,
        "test_candidate_metrics": test_candidate_metrics,
        "selected_test_metrics": test_metrics,
    }
    prediction_rows = [
        {
            "incident_id": incident_id,
            "actual_initial_severity_code": actual,
            "predicted_initial_severity_code": predicted,
            "max_probability": (
                float(probability.max()) if selected_probabilities is not None else ""
            ),
        }
        for incident_id, actual, predicted, probability in zip(
            test["incident_id"].tolist(),
            y_test.tolist(),
            test_pred.tolist(),
            selected_probabilities
            if selected_probabilities is not None
            else [None] * len(test),
        )
    ]
    return result, prediction_rows


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "mae_seconds": mean_absolute_error(y_true, predictions),
        "rmse_seconds": math.sqrt(mean_squared_error(y_true, predictions)),
        "r2": r2_score(y_true, predictions),
        "median_absolute_error_seconds": float(
            np.median(np.abs(y_true - predictions))
        ),
    }


def run_response_experiment(
    data: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    processed = add_time_features(data)
    train_mask = processed["incident_datetime"].str.startswith("2024")
    date = pd.to_datetime(processed["incident_datetime"])
    validation_mask = (
        processed["incident_datetime"].str.startswith("2025") & (date.dt.month <= 6)
    )
    test_mask = (
        processed["incident_datetime"].str.startswith("2025") & (date.dt.month >= 7)
    )
    outputs = {}
    all_predictions = []

    for target, valid_field in RESPONSE_TARGETS.items():
        valid = processed[valid_field].fillna("").str.upper().eq("Y")
        target_values = pd.to_numeric(processed[target], errors="coerce")
        valid &= target_values.notna() & np.isfinite(target_values) & (target_values >= 0)
        task_train = processed[train_mask & valid].copy()
        task_validation = processed[validation_mask & valid].copy()
        task_test = processed[test_mask & valid].copy()
        y_train = np.log1p(target_values[train_mask & valid].to_numpy(dtype=float))
        y_validation = target_values[validation_mask & valid].to_numpy(dtype=float)
        y_test = target_values[test_mask & valid].to_numpy(dtype=float)
        if not len(task_train) or not len(task_validation) or not len(task_test):
            raise RuntimeError(f"Empty valid time target split for {target}.")

        validation_baseline = DummyRegressor(strategy="median")
        dummy_train = np.zeros((len(task_train), 1))
        dummy_validation = np.zeros((len(task_validation), 1))
        validation_baseline.fit(dummy_train, y_train)
        baseline_validation_prediction = np.expm1(
            validation_baseline.predict(dummy_validation)
        )
        candidates = [
            (
                "training_median_baseline",
                mean_absolute_error(y_validation, baseline_validation_prediction),
                None,
            )
        ]
        validation_metrics = {
            "training_median_baseline": regression_metrics(
                y_validation, baseline_validation_prediction
            )
        }

        try:
            x_train, (x_validation, x_test), categorical_mask = encode_for_histogram(
                task_train,
                [task_validation, task_test],
                CATEGORICAL_FEATURES + ["initial_severity_level_code"],
                NUMERIC_FEATURES,
            )
            model = HistGradientBoostingRegressor(
                categorical_features=categorical_mask,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=2.0,
                early_stopping=True,
                random_state=42,
            )
            model.fit(x_train, y_train)
            validation_prediction = np.maximum(
                0, np.expm1(model.predict(x_validation))
            )
            validation_metrics["hist_gradient_boosting_log_target"] = (
                regression_metrics(y_validation, validation_prediction)
            )
            candidates.append(
                (
                    "hist_gradient_boosting_log_target",
                    mean_absolute_error(y_validation, validation_prediction),
                    model,
                )
            )
        except CategoryCardinalityError as exc:
            outputs[target] = {
                "status": "not_run",
                "reason": str(exc),
            }
            continue

        selected_name, _, selected_model = min(candidates, key=lambda item: item[1])
        baseline_test = DummyRegressor(strategy="median")
        baseline_test.fit(dummy_train, y_train)
        baseline_test_prediction = np.maximum(
            0, np.expm1(baseline_test.predict(np.zeros((len(task_test), 1))))
        )
        test_results = {
            "training_median_baseline": regression_metrics(
                y_test, baseline_test_prediction
            )
        }
        gradient_boosting_test_prediction = None
        if selected_model is not None:
            gradient_boosting_test_prediction = np.maximum(
                0, np.expm1(selected_model.predict(x_test))
            )
            test_results["hist_gradient_boosting_log_target"] = regression_metrics(
                y_test, gradient_boosting_test_prediction
            )
        if selected_name == "training_median_baseline":
            selected_prediction = baseline_test_prediction
        else:
            selected_prediction = gradient_boosting_test_prediction
        outputs[target] = {
            "target": target,
            "label_filter": f"{valid_field} == 'Y', finite non-negative seconds",
            "features": RESPONSE_FEATURES,
            "split": {
                "train": "2024 calendar year",
                "validation": "2025-01 through 2025-06",
                "test": "2025-07 through 2025-12",
                "counts": {
                    "train": len(task_train),
                    "validation": len(task_validation),
                    "test": len(task_test),
                },
            },
            "models_tested": [
                {
                    "name": "training_median_baseline",
                    "hyperparameters": {"strategy": "median"},
                },
                {
                    "name": "hist_gradient_boosting_log_target",
                    "hyperparameters": {
                        "target_transform": "log1p",
                        "max_iter": 150,
                        "max_leaf_nodes": 31,
                        "min_samples_leaf": 40,
                        "l2_regularization": 2.0,
                        "early_stopping": True,
                        "random_state": 42,
                    },
                },
            ],
            "validation_metrics": validation_metrics,
            "selected_model_by_validation_mae": selected_name,
            "test_metrics": test_results,
            "target_distribution_seconds": {
                "train_median": float(
                    np.expm1(np.median(y_train))
                ),
                "test_median": float(np.median(y_test)),
                "test_p95": float(np.quantile(y_test, 0.95)),
            },
        }
        for incident_id, actual, prediction in zip(
            task_test["incident_id"], y_test, selected_prediction
        ):
            all_predictions.append(
                {
                    "target": target,
                    "incident_id": incident_id,
                    "actual_seconds": actual,
                    "predicted_seconds": prediction,
                }
            )
    return outputs, all_predictions


def metric(value: float) -> str:
    return f"{value:.4f}"


def write_report(
    output_path: Path,
    run_date: str,
    inspection: dict,
    severity: dict,
    response: dict,
    sample_rows: int,
) -> None:
    non_null = inspection["non_null_counts"]
    missing_table_rows = []
    for field in inspection["columns"]:
        field_name = field["fieldName"]
        missing = int(inspection["record_count"]) - int(
            non_null[f"{field_name}_nonnull"]
        )
        missing_table_rows.append(
            f"| `{field_name}` | {field['dataTypeName']} | {missing:,} |"
        )
    missing_table = "\n".join(missing_table_rows)
    years = "\n".join(
        f"| {row['year']} | {int(row['records']):,} |"
        for row in inspection["year_distribution"]
    )
    severity_metrics = severity["selected_test_metrics"]
    severity_candidate_rows = []
    for candidate in severity["validation_candidates"]:
        name = candidate["model"]
        validation_metrics = candidate.get("validation_metrics", {})
        test_metrics = severity["test_candidate_metrics"].get(name, {})
        if not validation_metrics or not test_metrics:
            severity_candidate_rows.append(
                f"| `{name}` | not run | not run | not run | not run |"
            )
            continue
        severity_candidate_rows.append(
            f"| `{name}` | {metric(validation_metrics['macro_f1'])} | "
            f"{metric(test_metrics['accuracy'])} | "
            f"{metric(test_metrics['macro_f1'])} | "
            f"{metric(test_metrics['weighted_f1'])} |"
        )
    response_rows = []
    for target, result in response.items():
        if result.get("status") == "not_run":
            response_rows.append(
                f"| `{target}` | not run | {result['reason']} |"
            )
            continue
        baseline = result["test_metrics"]["training_median_baseline"]
        selected = result["selected_model_by_validation_mae"]
        validation_baseline = result["validation_metrics"]["training_median_baseline"]
        validation_candidate = result["validation_metrics"].get(
            "hist_gradient_boosting_log_target"
        )
        candidate = result["test_metrics"].get("hist_gradient_boosting_log_target")
        if candidate is None:
            candidate_mae = candidate_rmse = candidate_r2 = "not run"
        else:
            candidate_mae = metric(candidate["mae_seconds"])
            candidate_rmse = metric(candidate["rmse_seconds"])
            candidate_r2 = metric(candidate["r2"])
        response_rows.append(
            f"| `{target}` | {selected} | "
            f"{metric(validation_baseline['mae_seconds'])} | "
            f"{metric(validation_candidate['mae_seconds']) if validation_candidate else 'not run'} | "
            f"{metric(baseline['mae_seconds'])} | "
            f"{candidate_mae} | {candidate_rmse} | {candidate_r2} |"
        )

    report = f"""# NYC EMS Incident Dispatch Data: ResQ External Experiment

## Source and scope

- Dataset: NYC Fire Department EMS Incident Dispatch Data (Socrata ID `76xm-jjuj`).
- Official dataset: {OFFICIAL_URL}
- Data.gov catalog: {DATAGOV_URL}
- API used: `{RESOURCE_URL}` and `{METADATA_URL}`.
- Accessed: {run_date} UTC.
- Source description: {inspection['source_description']}
- Full source record count at access: **{inspection['record_count']:,}**.
- Source incident time range: **{inspection['first_incident_datetime']}** through **{inspection['last_incident_datetime']}**.
- This is a full-source API aggregate inspection plus a separate, bounded model sample; the 30.4M source rows were not copied locally.
- Training/evaluation sample: **{sample_rows:,} unique incidents**, downloaded from the official API and saved beside this report as `nyc_ems_model_sample.csv.gz`.
- Sample method: one deterministic 5,000-row window within each month, ordered by incident time and ID and centered in that month's sorted rows. This is reproducible temporal/seasonal coverage, but it is clustered within each window and is not a random sample.
- Original synthetic ResQ data remains separate and unchanged.

## Full-data inspection

Full-source query counts and distributions are saved in `dataset_inspection.json`, `field_inventory.csv`, `categorical_distributions.csv`, and `monthly_record_counts.csv`.

| Year | Records |
|---:|---:|
{years}

| Field | Source type | Missing/null rows |
|---|---|---:|
{missing_table}

- Incident ID uniqueness: {inspection['duplicate_summary']['unique_incident_ids']:,} unique IDs for {inspection['record_count']:,} rows; duplicate IDs: {int(inspection['duplicate_summary']['duplicate_incident_ids']):,}.
- The source schema has {len(inspection['columns'])} fields: initial/final call type and severity codes; assignment, activation, on-scene and hospital/closure timestamps; dispatch/incident/travel durations and validity indicators; disposition; borough, dispatch area and aggregated geography; and operational flags.
- The source does **not** contain caller/dispatcher free-text narratives, dispatched unit identifiers/types, responder skill labels, agency/unit assignment fields, or responder outcomes. Geography is deliberately aggregated in the source description for privacy.
- Full-source initial-severity counts are `{severity['full_source_label_counts']}`. Source codes not observed in this model sample are `{severity['source_labels_not_observed_in_sample']}`; the fitted model cannot predict those unseen codes. The full-code counts remain available in `categorical_distributions.csv`.
- Response-time fields and timestamp extrema are in `dataset_inspection.json`; every categorical level and its full-source frequency is in `categorical_distributions.csv`.

## Leakage audit

`leakage_audit.csv` labels each source field separately for severity and response-time tasks. It uses only these categories: `VALID_AT_PREDICTION_TIME`, `TARGET`, `TARGET_DERIVED`, `POST_DISPATCH`, `LEAKAGE`, `CONTEXT_ONLY`, and `UNKNOWN`.

For the severity task, only `initial_call_type`, intake timestamp features, borough, and dispatch area are model features. `initial_severity_level_code` is the target. For response-time tasks, those same intake fields plus the initial severity code are features. Final-call/final-severity fields, assignment/on-scene/activation/transport/closure fields, actual durations, and response-validity flags never enter the feature matrix. Validity flags are used only to select rows with valid labels. The NYC coded severity target is not converted into ResQ's Low/Medium/High/Critical labels.

## Models and evaluation

- Temporal split: 2024 for training, Jan–Jun 2025 for validation/model selection, Jul–Dec 2025 for the untouched test.
- Classification candidate models: most-frequent baseline, one-hot logistic regression (`C` in 0.1/1/10, balanced class weights), and structured HistGradientBoosting (120 iterations, 31 leaves, 50 minimum leaf samples, L2 1.0, balanced class weights). Candidate selection uses validation macro F1.
- Response-time candidates: training-median baseline and HistGradientBoosting regression on `log1p(seconds)` (150 iterations, 31 leaves, 40 minimum leaf samples, L2 2.0). Selection uses validation MAE.
- No text exists for TF-IDF, embeddings, or transformer experiments; using one would require fabricating narratives. XGBoost/LightGBM/CatBoost and deep models were not run because the source has no narrative or unit-level target and those models cannot fix that target/schema mismatch.
- Exact split distributions, per-class scores, confusion matrix, confidence summaries, and every candidate's validation metrics are in `experiment_metrics.json`.

### Initial severity-code test set

- Selected model: `{severity['selected_model']}`.
- Accuracy: **{metric(severity_metrics['accuracy'])}**
- Macro F1: **{metric(severity_metrics['macro_f1'])}**
- Weighted F1: **{metric(severity_metrics['weighted_f1'])}**
- Macro precision / recall: **{metric(severity_metrics['macro_precision'])} / {metric(severity_metrics['macro_recall'])}**
- Per-class metrics and confusion matrix: `experiment_metrics.json`.
- Confidence values are raw model outputs, not calibrated probabilities; low-confidence coverage diagnostics are included.

| Candidate | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 |
|---|---:|---:|---:|---:|
{chr(10).join(severity_candidate_rows)}

### Response-time test set

| Target | Selected by validation | Validation median MAE (s) | Validation HistBoost MAE (s) | Test median MAE (s) | Test HistBoost MAE (s) | HistBoost RMSE (s) | HistBoost R2 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(response_rows)}

## ResQ mapping and model comparison

- **Directly trainable from NYC fields:** a standalone NYC initial-severity-code classifier, and standalone dispatch-response / incident-response duration regressors using intake-time structured data.
- **Indirectly validated:** response-time and coded-triage analyses demonstrate the kinds of operational targets ResQ could support if it captures compatible, time-stamped dispatcher/dispatch fields.
- **Not validated or trainable from this source:** ResQ free-text emergency category classification; ResQ skill/resource multi-label prediction; responder-type prediction/ranking; volunteer availability/dispatch assignment; escalation rules; and ResQ's four-level severity labels. No valid target or compatible text exists for these tasks.
- **No ResQ production model is replaced.** The current API consumes narrative text and ResQ skill labels, while this dataset has neither. Wiring the NYC model into those endpoints would require unsupported field/label mappings.
- Existing `FINAL_MODEL_SELECTION.md` reports synthetic prototype scores, but the corresponding training/evaluation artifacts are not available to reproduce those scores. They are not comparable to this temporal NYC structured experiment; no before/after ResQ improvement is claimed.
- As a reference only, the existing synthetic report lists category macro F1 0.79, severity macro F1 0.74, skill macro F1 0.72, and responder subset accuracy 0.76. The NYC coded initial-severity result uses different labels and dispatcher-coded inputs, so its score cannot be interpreted as an improvement over the ResQ severity score.
- The original synthetic dataset and prior experiment reports are preserved.

## Relevant public-data leads

- GitHub project [Venkata-Rathna-Anirudh-Mamidipaka/911-Call-Service-DPD](https://github.com/Venkata-Rathna-Anirudh-Mamidipaka/911-Call-Service-DPD) describes Detroit police calls with responding agency/unit, call type, dispatch/travel/response times. This is a useful unit/dispatch schema lead, but police-only and not a volunteer/EMS label substitute.
- GitHub project [revanthkumar1999/Analysis-on-Law-Enforcement-Dispatched-Calls](https://github.com/revanthkumar1999/Analysis-on-Law-Enforcement-Dispatched-Calls) points to San Francisco's official closed and real-time police-dispatch datasets. This can support future cross-city dispatch work, but it is police-focused and should be downloaded from the cited City of San Francisco source, not treated as EMS ground truth.
- No verified public GitHub-hosted dataset with volunteer responder availability/acceptance and completed emergency dispatch labels was identified. Synthetic ResQ responder-pool records must remain explicitly synthetic unless a governed real volunteer dataset is found.

## Limitations and next steps

- The NYC dataset describes EMS coded calls and operational outcomes, not caller speech/text or dispatched resource identity. Initial/final coded severities are operational labels, not universal clinical severity.
- Window sampling is clustered; test performance should be confirmed against a full-year/full-population batch before any operational use.
- Severity performance must be judged per code and macro F1, not accuracy alone. No 90% target is asserted unless the held-out metrics support it.
- The real, privacy-safe path to improving ResQ skill/responder tasks is to capture timestamped call intake text/structured fields alongside actual dispatched unit/resource/skill assignments and availability, with access controls and a prospective evaluation protocol.

## Reproduction

From the `ResQ_AI` directory:

```bash
.venv/bin/python experiments/nyc_ems/run_experiment.py
```

The script queries full-source aggregates, downloads the fixed 2024–2025 stratified sample, trains/evaluates the candidates, and creates a new timestamped artifact directory. It needs internet access to NYC Open Data. The downloaded model sample in this run directory is the exact snapshot used for the metrics.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ARTIFACTS / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Inspecting full NYC EMS source through Socrata aggregates...")
    full = aggregate_dataset()
    inspection = full["inspection"]
    sample = fetch_model_sample(full["monthly_counts"])
    sample_path = output_dir / "nyc_ems_model_sample.csv.gz"
    sample.to_csv(sample_path, index=False, compression="gzip")

    columns = full["columns"]
    non_null = inspection["non_null_counts"]
    field_inventory = []
    for column in columns:
        field = column["fieldName"]
        field_inventory.append(
            {
                "field": field,
                "display_name": column.get("name", ""),
                "data_type": column.get("dataTypeName", ""),
                "records": inspection["record_count"],
                "non_null": int(non_null[f"{field}_nonnull"]),
                "missing_null": inspection["record_count"]
                - int(non_null[f"{field}_nonnull"]),
                "distinct_if_profiled": non_null.get(f"{field}_distinct", ""),
                "description": column.get("description") or "",
            }
        )
    write_csv(output_dir / "field_inventory.csv", field_inventory)
    write_csv(
        output_dir / "categorical_distributions.csv",
        full["category_rows"],
        ["field", "value", "records"],
    )
    write_csv(
        output_dir / "monthly_record_counts.csv",
        inspection["monthly_distribution"],
        ["year", "month", "records"],
    )
    write_csv(
        output_dir / "leakage_audit.csv",
        LEAKAGE_AUDIT,
        [
            "field",
            "severity_experiment",
            "response_experiment",
            "used_as_feature",
            "reason",
        ],
    )
    (output_dir / "dataset_inspection.json").write_text(
        json.dumps(inspection, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Training structured severity and response-time experiments...")
    full_severity_counts = {
        str(row["value"]): int(row["records"])
        for row in full["category_rows"]
        if row["field"] == CLASS_TARGET and row["value"] is not None
    }
    severity, severity_predictions = run_severity_experiment(
        sample, full_severity_counts
    )
    response, response_predictions = run_response_experiment(sample)
    metrics = {
        "source": {
            "official_url": OFFICIAL_URL,
            "datagov_url": DATAGOV_URL,
            "resource_api": RESOURCE_URL,
            "accessed_at_utc": run_date,
            "inspection": inspection,
            "sample": {
                "rows": len(sample),
                "unique_incident_ids": int(sample["incident_id"].nunique()),
                "method": (
                    "one deterministic 5,000-row window per month, centered in "
                    "sorted monthly rows"
                ),
            },
        },
        "severity_experiment": severity,
        "response_time_experiments": response,
    }
    (output_dir / "experiment_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(
        output_dir / "severity_test_predictions.csv",
        severity_predictions,
        [
            "incident_id",
            "actual_initial_severity_code",
            "predicted_initial_severity_code",
            "max_probability",
        ],
    )
    write_csv(
        output_dir / "response_time_test_predictions.csv",
        response_predictions,
        ["target", "incident_id", "actual_seconds", "predicted_seconds"],
    )
    write_report(
        output_dir / "report.md",
        run_date,
        inspection,
        severity,
        response,
        len(sample),
    )
    print(f"Created experiment artifacts in {output_dir}")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "source_rows": inspection["record_count"],
                "sample_rows": len(sample),
                "severity_model": severity["selected_model"],
                "severity_test": {
                    key: severity["selected_test_metrics"][key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "macro_precision",
                        "macro_recall",
                    )
                },
                "response_test": {
                    target: {
                        name: scores
                        for name, scores in result.get("test_metrics", {}).items()
                    }
                    for target, result in response.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

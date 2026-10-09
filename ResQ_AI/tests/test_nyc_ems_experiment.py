import pandas as pd

from experiments.nyc_ems.run_experiment import (
    CATEGORICAL_FEATURES,
    LEAKAGE_AUDIT,
    NUMERIC_FEATURES,
    RESPONSE_FEATURES,
    SEVERITY_FEATURES,
    add_time_features,
)


def test_time_features_are_derived_from_incident_creation_time():
    frame = pd.DataFrame(
        {
            "incident_datetime": [
                "2024-01-01T00:00:00.000",
                "2025-12-31T23:00:00.000",
            ],
            "initial_call_type": ["INJURY", "SICK"],
            "initial_severity_level_code": ["5", "2"],
            "borough": ["BROOKLYN", "QUEENS"],
            "incident_dispatch_area": ["K6", "Q1"],
        }
    )

    result = add_time_features(frame)

    assert len(result) == 2
    assert all(feature in result for feature in NUMERIC_FEATURES)
    assert result.loc[0, "hour_sin"] == 0
    assert result.loc[0, "month_sin"] == 0
    assert set(result["initial_severity_level_code"]) == {"2", "5"}


def test_model_features_exclude_post_dispatch_and_outcome_fields():
    audited = {row["field"]: row for row in LEAKAGE_AUDIT}
    severity_sources = {
        "incident_datetime",
        "initial_call_type",
        "borough",
        "incident_dispatch_area",
    }
    response_sources = severity_sources | {"initial_severity_level_code"}

    assert set(CATEGORICAL_FEATURES + NUMERIC_FEATURES) == set(SEVERITY_FEATURES)
    assert audited["initial_severity_level_code"]["severity_experiment"] == "TARGET"
    assert audited["initial_severity_level_code"]["response_experiment"] == (
        "VALID_AT_PREDICTION_TIME"
    )
    assert {
        field
        for field, row in audited.items()
        if row["used_as_feature"]
    } == response_sources
    assert not audited["incident_travel_tm_seconds_qy"]["used_as_feature"]
    assert audited["incident_travel_tm_seconds_qy"]["response_experiment"] == "LEAKAGE"
    assert "final_severity_level_code" not in RESPONSE_FEATURES
    assert "dispatch_response_seconds_qy" not in RESPONSE_FEATURES


def test_one_severity_target_is_not_silently_mapped_to_resq_labels():
    assert "initial_severity_level_code" not in SEVERITY_FEATURES
    assert "initial_severity_level_code" in RESPONSE_FEATURES
    assert "initial_severity_level_code" in {
        row["field"] for row in LEAKAGE_AUDIT
    }

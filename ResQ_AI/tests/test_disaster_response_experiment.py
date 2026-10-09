import numpy as np
import pandas as pd

from experiments.disaster_response.run_experiment import (
    get_label_columns,
    multilabel_metrics,
    remove_duplicate_messages,
)


def test_label_columns_keep_related_separate_from_binary_tags():
    labels, binary = get_label_columns(
        [
            "id",
            "split",
            "message",
            "related",
            "tag_a",
            "tag_b",
            "direct_report",
            "event",
        ]
    )

    assert labels == ["related", "tag_a", "tag_b", "direct_report"]
    assert binary == ["tag_a", "tag_b", "direct_report"]


def test_duplicate_text_is_removed_from_later_splits_only():
    frames = {
        "training": pd.DataFrame(
            {"message": ["Storm warning", "Need clean water"]}
        ),
        "validation": pd.DataFrame(
            {"message": [" storm   WARNING ", "Need shelter"]}
        ),
        "test": pd.DataFrame({"message": ["Need clean water", "Flood nearby"]}),
    }

    prepared, dropped = remove_duplicate_messages(frames)

    assert [len(prepared[name]) for name in ("training", "validation", "test")] == [
        2,
        1,
        1,
    ]
    assert dropped == {"training": 0, "validation": 1, "test": 1}


def test_multilabel_metrics_report_exact_and_elementwise_accuracy():
    truth = np.array([[1, 0], [0, 1]], dtype=np.int8)
    prediction = np.array([[1, 1], [0, 1]], dtype=np.int8)

    metrics = multilabel_metrics(truth, prediction, [0, 1])

    assert metrics["subset_accuracy_exact_match"] == 0.5
    assert metrics["labelwise_accuracy"] == 0.75
    assert metrics["micro_f1"] == 0.8

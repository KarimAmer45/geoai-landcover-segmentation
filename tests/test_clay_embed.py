import json

import numpy as np

from src.clay_embed import evaluate_predictions, run_downstream_experiment


def test_metrics_include_bootstrap_confidence_intervals():
    report = evaluate_predictions(
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0],
        bootstrap_iterations=100,
    )
    assert report["sample_count"] == 6
    assert report["accuracy"]["score"] == 4 / 6
    assert len(report["f1"]["ci95"]) == 2


def test_metrics_support_named_positive_class():
    report = evaluate_predictions(
        ["non-forest", "non-forest", "forest", "forest"],
        ["non-forest", "forest", "forest", "forest"],
        positive_label="forest",
        bootstrap_iterations=50,
    )
    assert report["f1"]["score"] == 0.8


def test_downstream_experiment_is_reproducible(tmp_path):
    rng = np.random.default_rng(7)
    labels = np.repeat([0, 1], 24)
    embeddings = rng.normal(size=(48, 12)).astype(np.float32)
    embeddings[:, 0] += labels * 4
    source = tmp_path / "embeddings.npz"
    np.savez_compressed(
        source,
        embeddings=embeddings,
        labels=labels,
        model_id=np.array("synthetic CI embeddings"),
    )

    report = run_downstream_experiment(
        source,
        tmp_path / "result",
        bootstrap_iterations=100,
    )
    assert report["embedding_source"] == "synthetic CI embeddings"
    assert report["accuracy"]["score"] >= 0.8
    assert (tmp_path / "result" / "classifier.joblib").exists()
    assert json.loads((tmp_path / "result" / "metrics.json").read_text())["random_state"] == 42

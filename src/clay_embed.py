"""Track B2: Clay v1.5 embeddings and a light downstream classifier."""

from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLAY_CHECKPOINT_URL = (
    "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
)


def load_clay_encoder(checkpoint: str | Path, device: str = "cpu") -> Any:
    """Load the official Clay v1.5 encoder from a Lightning checkpoint."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Clay checkpoint not found: {checkpoint_path}. Download it from {CLAY_CHECKPOINT_URL}"
        )
    try:
        from claymodel.module import ClayMAEModule
    except ImportError as exc:  # pragma: no cover - heavyweight optional extra
        raise RuntimeError(
            "The official Clay package is required; install requirements-clay.txt"
        ) from exc
    model = ClayMAEModule.load_from_checkpoint(str(checkpoint_path), map_location=device)
    model.eval()
    model.to(device)
    return model.encoder


def embed_chips(
    chips: np.ndarray,
    timestamps: np.ndarray,
    wavelengths: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    encoder: Any | None = None,
) -> np.ndarray:
    """Generate Clay embeddings for normalized chips.

    Inputs follow the official Clay contract: chips ``[N,C,H,W]``, timestamps
    ``[N,4]`` (week, hour, latitude, longitude), and wavelengths in nanometres
    as either ``[C]`` or ``[N,C]``. Callers are responsible for sensor-specific
    normalization using Clay's metadata file.
    """
    chips = np.asarray(chips, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32)
    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    if chips.ndim != 4:
        raise ValueError("chips must have shape [N, C, H, W]")
    if timestamps.shape != (chips.shape[0], 4):
        raise ValueError("timestamps must have shape [N, 4]")
    if wavelengths.ndim == 1:
        wavelengths = np.repeat(wavelengths[None, :], chips.shape[0], axis=0)
    if wavelengths.shape != chips.shape[:2]:
        raise ValueError("wavelengths must have shape [C] or [N, C]")
    if encoder is None and checkpoint is None:
        raise ValueError("checkpoint is required when encoder is not supplied")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Clay inference") from exc
    active_encoder = encoder or load_clay_encoder(checkpoint, device=device)
    chip_tensor = torch.from_numpy(chips).to(device)
    time_tensor = torch.from_numpy(timestamps).to(device)
    wavelength_tensor = torch.from_numpy(wavelengths).to(device)
    with torch.inference_mode():
        result = active_encoder(chip_tensor, time_tensor, wavelength_tensor)
    if isinstance(result, tuple | list):
        result = result[0]
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    embeddings = np.asarray(result, dtype=np.float32)
    if embeddings.ndim > 2:
        embeddings = embeddings.mean(axis=tuple(range(1, embeddings.ndim - 1)))
    if embeddings.ndim != 2 or embeddings.shape[0] != chips.shape[0]:
        raise RuntimeError(f"unexpected Clay encoder output shape: {embeddings.shape}")
    return embeddings


def train_classifier(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Any:
    """Fit a standardized logistic classifier on frozen Clay embeddings."""
    X, y = _validate_xy(X, y)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=random_state),
    )
    return classifier.fit(X, y)


def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be [samples, features] and y must be [samples]")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or np.min(counts) < 4:
        raise ValueError("binary classification requires at least four samples per class")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")
    return X, y


def _bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Any,
    *,
    iterations: int,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_true, sampled_pred = y_true[indices], y_pred[indices]
        if len(np.unique(sampled_true)) < 2:
            continue
        values.append(float(metric(sampled_true, sampled_pred)))
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def evaluate_predictions(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    positive_label: Any = 1,
    bootstrap_iterations: int = 1_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Report accuracy, balanced accuracy, F1, and percentile bootstrap CIs."""
    truth, predictions = np.asarray(list(y_true)), np.asarray(list(y_pred))
    if truth.shape != predictions.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("y_true and y_pred must be equally sized non-empty vectors")
    metrics = {
        "accuracy": lambda a, b: accuracy_score(a, b),
        "balanced_accuracy": lambda a, b: balanced_accuracy_score(a, b),
        "f1": lambda a, b: f1_score(a, b, pos_label=positive_label, zero_division=0),
    }
    report: dict[str, Any] = {"sample_count": int(len(truth))}
    for offset, (name, metric) in enumerate(metrics.items()):
        score = float(metric(truth, predictions))
        low, high = _bootstrap_interval(
            truth,
            predictions,
            metric,
            iterations=bootstrap_iterations,
            random_state=random_state + offset,
        )
        report[name] = {"score": score, "ci95": [low, high]}
    return report


def run_downstream_experiment(
    embeddings_file: str | Path,
    output_dir: str | Path = "outputs/clay",
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    bootstrap_iterations: int = 1_000,
) -> dict[str, Any]:
    """Train/evaluate forest-vs-non-forest classification from saved embeddings.

    The NPZ input must contain ``embeddings`` and ``labels`` arrays. A held-out,
    stratified split prevents train/test leakage; the output preserves split indices.
    """
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    with np.load(embeddings_file, allow_pickle=False) as payload:
        X, y = _validate_xy(payload["embeddings"], payload["labels"])
        embedding_source = (
            str(payload["model_id"].item())
            if "model_id" in payload.files
            else "unverified embeddings (model_id absent)"
        )
        checkpoint_sha256 = (
            str(payload["checkpoint_sha256"].item())
            if "checkpoint_sha256" in payload.files
            else None
        )
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    classifier = train_classifier(X[train_idx], y[train_idx], random_state=random_state)
    predictions = classifier.predict(X[test_idx])
    report = evaluate_predictions(
        y[test_idx],
        predictions,
        positive_label=np.unique(y)[-1],
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    report.update(
        {
            "task": "forest_vs_non_forest",
            "embedding_source": embedding_source,
            "checkpoint_sha256": checkpoint_sha256,
            "random_state": random_state,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
        }
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, destination / "classifier.joblib")
    (destination / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def embed_npz(
    input_file: str | Path,
    output_file: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> Path:
    """Embed prepared Clay inputs from NPZ and preserve optional labels."""
    with np.load(input_file, allow_pickle=False) as payload:
        embeddings = embed_chips(
            payload["chips"],
            payload["timestamps"],
            payload["wavelengths"],
            checkpoint=checkpoint,
            device=device,
        )
        labels = payload["labels"] if "labels" in payload.files else None
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    with Path(checkpoint).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance = {
        "model_id": np.array("Clay v1.5 frozen encoder"),
        "checkpoint_sha256": np.array(digest.hexdigest()),
    }
    if labels is None:
        np.savez_compressed(output, embeddings=embeddings, **provenance)
    else:
        np.savez_compressed(output, embeddings=embeddings, labels=labels, **provenance)
    return output

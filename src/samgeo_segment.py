"""Track A: zero-training, text-promptable satellite-image segmentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .postprocess import create_preview, mask_to_polygons, write_area_summary


@dataclass(frozen=True)
class SegmentationResult:
    mask_tif: str
    geojson: str
    summary_json: str
    preview_png: str
    class_name: str
    polygon_count: int
    total_area_ha: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_langsam(model_type: str) -> Any:
    try:
        from samgeo.text_sam import LangSAM
    except ImportError as exc:  # pragma: no cover - heavyweight optional extra
        raise RuntimeError(
            "SAMGeo text support is required; install requirements-models.txt"
        ) from exc
    return LangSAM(model_type=model_type)


def segment_text(
    image: str | Path,
    text_prompt: str,
    output_dir: str | Path = "outputs",
    *,
    box_threshold: float = 0.24,
    text_threshold: float = 0.24,
    model_type: str = "sam2-hiera-tiny",
    min_area_ha: float = 0.0,
    model_factory: Callable[[str], Any] | None = None,
) -> SegmentationResult:
    """Segment ``text_prompt`` and produce mask, polygons, area, and preview.

    Model weights are downloaded by SAMGeo on first use. ``model_factory`` is an
    explicit dependency-injection seam used by the CPU smoke test.
    """
    source = Path(image)
    if not source.is_file():
        raise FileNotFoundError(f"input image does not exist: {source}")
    prompt = text_prompt.strip()
    if not prompt or len(prompt) > 100:
        raise ValueError("text_prompt must contain between 1 and 100 characters")
    for name, value in (("box_threshold", box_threshold), ("text_threshold", text_threshold)):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    mask_path = destination / "mask.tif"
    vector_path = destination / "mask.geojson"
    summary_path = destination / "summary.json"
    preview_path = destination / "preview.png"

    factory = model_factory or _load_langsam
    model = factory(model_type)
    model.predict(
        str(source),
        prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        output=str(mask_path),
        mask_multiplier=1,
    )
    if not mask_path.is_file():
        raise RuntimeError("SAMGeo completed without writing the requested mask")

    polygons = mask_to_polygons(
        mask_path,
        vector_path,
        class_name=prompt,
        min_area_ha=min_area_ha,
    )
    summary = write_area_summary(polygons, prompt, summary_path)
    create_preview(source, mask_path, preview_path)
    return SegmentationResult(
        mask_tif=str(mask_path),
        geojson=str(vector_path),
        summary_json=str(summary_path),
        preview_png=str(preview_path),
        class_name=prompt,
        polygon_count=int(summary["polygon_count"]),
        total_area_ha=float(summary["total_area_ha"]),
    )


if __name__ == "__main__":  # pragma: no cover
    from .cli import main

    raise SystemExit(main(["segment"]))

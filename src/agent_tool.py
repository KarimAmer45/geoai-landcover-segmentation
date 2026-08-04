"""Schema-validated, bounded wrapper suitable for an LLM tool call."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from .data import fetch_tile, validate_bbox
from .samgeo_segment import SegmentationResult, segment_text


class SegmentRequest(BaseModel):
    """Strict tool schema; all coordinates are WGS84 decimal degrees."""

    class_name: Annotated[str, Field(min_length=1, max_length=100)]
    bbox: tuple[float, float, float, float]
    zoom: Annotated[int, Field(ge=8, le=20)] = 17

    @field_validator("class_name")
    @classmethod
    def normalize_class_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("class_name cannot be blank")
        return value

    @field_validator("bbox")
    @classmethod
    def check_bbox(cls, value: tuple[float, float, float, float]):
        return validate_bbox(value, max_area_deg2=0.005)


def segment(request: SegmentRequest, work_dir: str | Path = "outputs/agent") -> SegmentationResult:
    """Fetch a bounded tile and run the Track A pipeline."""
    destination = Path(work_dir)
    tile = fetch_tile(request.bbox, destination / "tile.tif", zoom=request.zoom)
    return segment_text(tile, request.class_name, destination)


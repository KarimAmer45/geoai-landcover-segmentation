"""Fetch and validate small georeferenced satellite tiles."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def validate_bbox(
    bbox: Sequence[float], max_area_deg2: float = 0.02
) -> tuple[float, float, float, float]:
    """Validate a WGS84 ``[west, south, east, north]`` bounding box.

    The area limit prevents accidental requests for very large web-map downloads.
    It is deliberately conservative because this project is a small demonstrator.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly [west, south, east, north]")
    west, south, east, north = (float(value) for value in bbox)
    if not (-180 <= west < east <= 180):
        raise ValueError("bbox longitudes must satisfy -180 <= west < east <= 180")
    if not (-85 <= south < north <= 85):
        raise ValueError("bbox latitudes must satisfy -85 <= south < north <= 85")
    if (east - west) * (north - south) > max_area_deg2:
        raise ValueError(
            f"bbox is too large for this demonstrator (limit: {max_area_deg2} square degrees)"
        )
    return west, south, east, north


def fetch_tile(
    bbox: Sequence[float],
    out: str | Path = "data/sample/tile.tif",
    zoom: int = 17,
    source: str = "Satellite",
) -> Path:
    """Download a web-map tile as a GeoTIFF using SAMGeo.

    Network access is required and the provider's terms and attribution still apply.
    """
    checked_bbox = validate_bbox(bbox)
    if not 1 <= zoom <= 22:
        raise ValueError("zoom must be between 1 and 22")
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from samgeo import tms_to_geotiff
    except ImportError as exc:  # pragma: no cover - depends on optional heavyweight extra
        raise RuntimeError(
            "SAMGeo is required to fetch tiles; install requirements-models.txt"
        ) from exc
    tms_to_geotiff(
        output=str(destination),
        bbox=list(checked_bbox),
        zoom=zoom,
        source=source,
        overwrite=True,
    )
    return destination

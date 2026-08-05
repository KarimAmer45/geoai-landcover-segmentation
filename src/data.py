"""Fetch and validate small georeferenced satellite tiles."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


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


# Human-readable source names mapped to contextily providers. Extend as needed;
# a raw XYZ URL template containing "{z}/{x}/{y}" is also accepted directly.
_TILE_SOURCES = {
    "Satellite": "Esri.WorldImagery",
    "Esri.WorldImagery": "Esri.WorldImagery",
    "OpenStreetMap": "OpenStreetMap.Mapnik",
}


def _resolve_source(source: str) -> Any:
    """Return a contextily provider (or pass a raw XYZ URL template through)."""
    if "{z}" in source and "{x}" in source and "{y}" in source:
        return source
    import contextily as cx

    provider_key = _TILE_SOURCES.get(source, source)
    provider: Any = cx.providers
    for part in provider_key.split("."):
        try:
            provider = provider[part]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"unknown tile source {source!r}; use a known name or an XYZ URL template"
            ) from exc
    return provider


def fetch_tile(
    bbox: Sequence[float],
    out: str | Path = "data/sample/tile.tif",
    zoom: int = 17,
    source: str = "Satellite",
) -> Path:
    """Download a web-map basemap tile as a georeferenced GeoTIFF.

    Uses contextily + rasterio (no ``osgeo``/GDAL bindings required, so it works in a
    plain pip environment on Windows). Network access is required and the provider's
    terms and attribution still apply. The output is written in EPSG:3857.
    """
    checked_bbox = validate_bbox(bbox)
    if not 1 <= zoom <= 22:
        raise ValueError("zoom must be between 1 and 22")
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import contextily as cx
    except ImportError as exc:  # pragma: no cover - depends on optional heavyweight extra
        raise RuntimeError(
            "contextily is required to fetch tiles; install requirements-models.txt"
        ) from exc
    west, south, east, north = checked_bbox
    cx.bounds2raster(
        west,
        south,
        east,
        north,
        str(destination),
        zoom=zoom,
        source=_resolve_source(source),
        ll=True,
    )
    return destination

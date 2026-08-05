"""Convert binary raster masks to polygons, statistics, and preview figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def _metric_crs(gdf: gpd.GeoDataFrame) -> Any:
    if gdf.crs is None:
        raise ValueError("mask has no CRS; area cannot be measured")
    if not gdf.crs.is_geographic:
        return gdf.crs
    estimated = gdf.estimate_utm_crs()
    if estimated is None:
        raise ValueError("could not estimate a local metric CRS for the mask")
    return estimated


def mask_to_polygons(
    mask_tif: str | Path,
    out_geojson: str | Path = "outputs/mask.geojson",
    *,
    class_name: str = "feature",
    min_area_ha: float = 0.0,
) -> gpd.GeoDataFrame:
    """Polygonize non-zero mask pixels and calculate geodesically sensible area.

    Areas are computed in a local UTM CRS when the source is geographic. Output is
    always GeoJSON-compatible EPSG:4326 while retaining an ``area_ha`` attribute.
    """
    if min_area_ha < 0:
        raise ValueError("min_area_ha cannot be negative")
    with rasterio.open(mask_tif) as src:
        if src.crs is None:
            raise ValueError("mask has no CRS; cannot create georeferenced polygons")
        band = src.read(1)
        valid = band > 0
        records = [
            {"geometry": shape(geometry), "class_name": class_name}
            for geometry, value in shapes(
                valid.astype(np.uint8), mask=valid, transform=src.transform
            )
            if value == 1
        ]
        source_crs = src.crs

    gdf = gpd.GeoDataFrame(
        records,
        columns=["geometry", "class_name"],
        geometry="geometry",
        crs=source_crs,
    )
    if not gdf.empty:
        metric = gdf.to_crs(_metric_crs(gdf))
        gdf["area_ha"] = metric.geometry.area.to_numpy() / 10_000.0
        gdf = gdf.loc[gdf["area_ha"] >= min_area_ha].copy()
    else:
        gdf["area_ha"] = np.array([], dtype=float)

    output = Path(out_geojson)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_gdf = gdf.to_crs(epsg=4326)
    output_gdf.to_file(output, driver="GeoJSON")
    return output_gdf


def area_summary(gdf: gpd.GeoDataFrame, class_name: str) -> dict[str, float | int | str]:
    """Return a stable, JSON-serializable summary of polygon areas."""
    total = float(gdf["area_ha"].sum()) if "area_ha" in gdf else 0.0
    return {
        "class_name": class_name,
        "polygon_count": int(len(gdf)),
        "total_area_ha": round(total, 6),
    }


def write_area_summary(
    gdf: gpd.GeoDataFrame,
    class_name: str,
    output: str | Path = "outputs/summary.json",
) -> dict[str, float | int | str]:
    summary = area_summary(gdf, class_name)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def create_preview(
    image_tif: str | Path,
    mask_tif: str | Path,
    output: str | Path = "outputs/preview.png",
) -> Path:
    """Create an RGB image with a translucent green segmentation overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(image_tif) as src:
        image = src.read(list(range(1, min(src.count, 3) + 1)))
    with rasterio.open(mask_tif) as src:
        mask = src.read(1) > 0
    if image.shape[1:] != mask.shape:
        raise ValueError("image and mask dimensions differ")
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    elif image.shape[0] == 2:
        image = np.concatenate([image, image[:1]], axis=0)
    rgb = np.moveaxis(image[:3], 0, -1).astype(np.float32)
    for channel in range(3):
        low, high = np.percentile(rgb[..., channel], (2, 98))
        rgb[..., channel] = np.clip((rgb[..., channel] - low) / max(high - low, 1e-6), 0, 1)
    # A high-contrast red reads on both vegetation and bare ground; a translucent
    # fill plus a crisp boundary keeps the mask legible even over green land, which a
    # green overlay could not. See docs note on Track A limits for spectrally similar classes.
    fill = np.array([1.0, 0.30, 0.20])
    alpha = 0.35
    overlay = rgb.copy()
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * fill

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(overlay)
    if mask.any() and not mask.all():
        axis.contour(mask.astype(float), levels=[0.5], colors="#c00000", linewidths=1.1)
    axis.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path

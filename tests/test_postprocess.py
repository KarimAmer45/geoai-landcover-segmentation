import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.postprocess import mask_to_polygons, write_area_summary


def test_polygonize_and_measure_exact_hectare(tmp_path):
    mask_path = tmp_path / "mask.tif"
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:12, 3:13] = 1  # 100 pixels * 10m * 10m = 1 ha
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="uint8",
        crs="EPSG:32647",
        transform=from_origin(740_000, 55_000, 10, 10),
    ) as destination:
        destination.write(mask, 1)

    geojson_path = tmp_path / "mask.geojson"
    gdf = mask_to_polygons(mask_path, geojson_path, class_name="trees")
    assert geojson_path.exists()
    assert len(gdf) == 1
    assert gdf.crs.to_epsg() == 4326
    assert gdf.iloc[0].area_ha == pytest.approx(1.0)

    summary_path = tmp_path / "summary.json"
    summary = write_area_summary(gdf, "trees", summary_path)
    assert summary == {"class_name": "trees", "polygon_count": 1, "total_area_ha": 1.0}
    assert json.loads(summary_path.read_text()) == summary


def test_empty_mask_writes_empty_feature_collection(tmp_path):
    mask_path = tmp_path / "empty.tif"
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint8",
        crs="EPSG:32647",
        transform=from_origin(740_000, 55_000, 10, 10),
    ) as destination:
        destination.write(np.zeros((4, 4), dtype=np.uint8), 1)

    geojson_path = tmp_path / "empty.geojson"
    gdf = mask_to_polygons(mask_path, geojson_path, class_name="water")
    assert gdf.empty
    assert json.loads(geojson_path.read_text())["features"] == []

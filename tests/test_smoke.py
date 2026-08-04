from pathlib import Path

import numpy as np
import rasterio

from src.samgeo_segment import segment_text


class FakeLangSAM:
    """Offline contract double: model inference only, real geospatial I/O afterward."""

    def predict(self, image, text_prompt, **kwargs):
        with rasterio.open(image) as source:
            rgb = source.read()
            mask = (rgb[1] > rgb[0]).astype(np.uint8)
            profile = source.profile
        profile.update(count=1, dtype="uint8")
        with rasterio.open(kwargs["output"], "w", **profile) as destination:
            destination.write(mask, 1)


def test_cpu_pipeline_on_bundled_fixture(tmp_path):
    fixture = Path("data/sample/tile.tif")
    assert fixture.exists()
    result = segment_text(
        fixture,
        "trees",
        tmp_path,
        model_factory=lambda _: FakeLangSAM(),
    )
    assert Path(result.mask_tif).exists()
    assert Path(result.geojson).exists()
    assert Path(result.preview_png).exists()
    assert result.total_area_ha > 0

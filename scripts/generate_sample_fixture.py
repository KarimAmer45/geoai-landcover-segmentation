"""Generate the tiny deterministic GeoTIFF fixture committed with the project."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def main() -> None:
    output = Path("data/sample/tile.tif")
    output.parent.mkdir(parents=True, exist_ok=True)
    height = width = 64
    rows, cols = np.mgrid[:height, :width]
    vegetation = ((rows - 31) ** 2 + (cols - 25) ** 2 < 18**2) | (
        (rows - 18) ** 2 + (cols - 49) ** 2 < 9**2
    )
    image = np.empty((3, height, width), dtype=np.uint8)
    image[0] = np.where(vegetation, 45, 175) + (cols % 9).astype(np.uint8)
    image[1] = np.where(vegetation, 145, 125) + (rows % 7).astype(np.uint8)
    image[2] = np.where(vegetation, 55, 85)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32647",
        "transform": from_origin(740_000, 55_000, 10, 10),
        "compress": "deflate",
    }
    with rasterio.open(output, "w", **profile) as destination:
        destination.write(image)
        destination.update_tags(
            fixture="synthetic",
            purpose="offline CI only; not a model-quality example",
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.moveaxis(image, 0, -1).astype(np.float32) / 255.0
    overlay = rgb.copy()
    overlay[vegetation] = 0.65 * overlay[vegetation] + 0.35 * np.array([1.0, 0.30, 0.20])
    preview = Path("docs/fixture-preview.png")
    preview.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(overlay)
    axis.contour(vegetation.astype(float), levels=[0.5], colors="#c00000", linewidths=1.1)
    axis.set_axis_off()
    figure.tight_layout(pad=0)
    figure.savefig(preview, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


if __name__ == "__main__":
    main()

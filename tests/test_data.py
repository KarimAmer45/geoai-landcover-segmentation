import pytest

from src.data import validate_bbox


def test_validate_bbox_accepts_small_wgs84_area():
    assert validate_bbox([101.435, 0.45, 101.445, 0.46]) == (
        101.435,
        0.45,
        101.445,
        0.46,
    )


@pytest.mark.parametrize(
    "bbox",
    [
        [10, 1, 9, 2],
        [-181, 1, 2, 3],
        [1, -90, 2, 3],
        [1, 2, 3],
        [0, 0, 10, 10],
    ],
)
def test_validate_bbox_rejects_invalid_or_unbounded_requests(bbox):
    with pytest.raises(ValueError):
        validate_bbox(bbox)


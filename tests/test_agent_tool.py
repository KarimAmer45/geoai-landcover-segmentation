import pytest
from pydantic import ValidationError

from src.agent_tool import SegmentRequest


def test_agent_schema_normalizes_and_bounds_request():
    request = SegmentRequest(class_name=" trees ", bbox=(101.435, 0.45, 101.445, 0.46))
    assert request.class_name == "trees"


def test_agent_schema_rejects_large_execution():
    with pytest.raises(ValidationError):
        SegmentRequest(class_name="trees", bbox=(100, 0, 101, 1))


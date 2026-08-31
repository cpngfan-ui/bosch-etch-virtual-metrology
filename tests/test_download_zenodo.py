"""Contract tests for the public-data download helper."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

download_url = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "download_zenodo.py")
)["download_url"]


def test_download_url_accepts_current_zenodo_self_link() -> None:
    record = {
        "links": {
            "self": "https://zenodo.org/api/records/17122442/files/example.nc/content"
        }
    }

    assert download_url(record) == record["links"]["self"]


def test_download_url_keeps_compatibility_with_content_link() -> None:
    record = {
        "links": {
            "content": "https://zenodo.org/api/records/17122442/files/example.nc/content",
            "self": "https://zenodo.org/api/records/17122442/files/example.nc",
        }
    }

    assert download_url(record) == record["links"]["content"]


@pytest.mark.parametrize("record", [{}, {"links": {}}, {"links": "invalid"}])
def test_download_url_rejects_records_without_a_content_url(record: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="links mapping|content URL"):
        download_url(record)

#!/usr/bin/env python3
"""Download and checksum the public BOSCH plasma-etch dataset from Zenodo.

The default command downloads only the compact modeling-core files used by the
first Virtual Metrology benchmark.  Pass ``--include-oes`` to also download the
roughly 7.9 GB daily optical-emission files.

Existing, checksum-valid files are never rewritten.  A checksum mismatch is a
hard failure so that ``data/raw`` remains an immutable source snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

RECORD_ID = "17122442"
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
CORE_FILES = {
    "Readme.pdf",
    "Wafer_layout.pdf",
    "Lot_status.xlsx",
    "Si_Oxide_etch_89_points.csv",
    "Si_Oxide_etch_9_points.csv",
    "Process_data.nc",
    "Dictionary_process.nc",
    "Dictionary_OES.nc",
}


def md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the MD5 digest used by Zenodo's file manifest."""

    digest = hashlib.md5()  # noqa: S324 - required to match the published manifest
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str) -> dict[str, Any]:
    """Fetch JSON with an explicit user agent."""

    request = urllib.request.Request(url, headers={"User-Agent": "bosch-etch-vm/0.1"})
    with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed HTTPS URL
        return json.load(response)


def expected_md5(file_record: dict[str, Any]) -> str:
    """Normalize Zenodo's ``md5:<digest>`` checksum representation."""

    checksum = str(file_record["checksum"])
    algorithm, digest = checksum.split(":", maxsplit=1)
    if algorithm.lower() != "md5":
        raise ValueError(f"Unexpected checksum algorithm: {algorithm}")
    return digest.lower()


def download_url(file_record: dict[str, Any]) -> str:
    """Return the file-content URL across supported Zenodo API schemas."""

    links = file_record.get("links")
    if not isinstance(links, dict):
        raise ValueError("Zenodo file record does not contain a links mapping")
    url = links.get("content") or links.get("self")
    if not isinstance(url, str) or not url:
        raise ValueError("Zenodo file record does not contain a content URL")
    return url


def download_file(file_record: dict[str, Any], destination: Path) -> None:
    """Download one file atomically and verify it before publication."""

    expected = expected_md5(file_record)
    if destination.exists():
        actual = md5(destination)
        if actual == expected:
            print(f"verified existing  {destination.name}")
            return
        raise RuntimeError(
            f"Refusing to overwrite {destination}: MD5 {actual} != published {expected}"
        )

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    url = download_url(file_record)
    request = urllib.request.Request(url, headers={"User-Agent": "bosch-etch-vm/0.1"})
    print(f"downloading        {destination.name}")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)

    actual = md5(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 failure for {destination.name}: {actual} != {expected}")
    temporary.replace(destination)
    print(f"verified download  {destination.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw") / f"zenodo_{RECORD_ID}",
        help="Immutable raw-data directory.",
    )
    parser.add_argument(
        "--include-oes",
        action="store_true",
        help="Also download the large daily OES files (about 7.9 GB).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = fetch_json(RECORD_API)
    selected = [
        item
        for item in record["files"]
        if args.include_oes or item["key"] in CORE_FILES
    ]
    missing_core = CORE_FILES - {item["key"] for item in selected}
    if missing_core:
        raise RuntimeError(
            f"Zenodo manifest is missing expected core files: {sorted(missing_core)}"
        )

    metadata_path = args.output_dir / "zenodo_record.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("id") != record.get("id"):
            raise RuntimeError(f"Refusing to overwrite unrelated metadata at {metadata_path}")
    else:
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    for file_record in sorted(selected, key=lambda item: item["key"]):
        download_file(file_record, args.output_dir / file_record["key"])

    print(f"complete: {len(selected)} files in {args.output_dir}")


if __name__ == "__main__":
    main()

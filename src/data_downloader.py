"""Utility functions for downloading and extracting the NOAA Global Hourly dataset."""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Optional

import requests

from . import config


def stream_download(url: str, destination: Path, chunk_size: int = 1_048_576) -> None:
    """Download a file from *url* to *destination* streaming the response."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fp.write(chunk)


def ensure_archive(url: str = config.DATASET_URL, archive_path: Optional[Path] = None) -> Path:
    """Ensure the compressed archive is available locally."""
    archive_path = archive_path or config.RAW_ARCHIVE_PATH
    if archive_path.exists():
        return archive_path
    stream_download(url, archive_path)
    return archive_path


def ensure_extracted(archive_path: Optional[Path] = None, extract_dir: Optional[Path] = None) -> Path:
    """Ensure the NOAA archive has been extracted and return the directory containing CSV files."""
    archive_path = archive_path or config.RAW_ARCHIVE_PATH
    extract_dir = extract_dir or config.EXTRACTED_DIR
    if extract_dir.exists() and any(extract_dir.rglob("*.csv")):
        return extract_dir

    archive_path = ensure_archive(archive_path=archive_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path=extract_dir)
    return extract_dir

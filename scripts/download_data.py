"""
scripts/download_data.py
------------------------
Downloads the UCI Student Performance dataset (ID 320) and extracts
student-mat.csv and student-por.csv into the configured raw data directory.

Usage
-----
    python scripts/download_data.py

The script resolves all paths relative to the **project root** (the directory
that contains this file's parent directory), so it works correctly regardless
of the current working directory.
"""

from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import requests
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_URL: str = (
    "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
)
EXPECTED_CSV_FILES: list[str] = ["student-mat.csv", "student-por.csv"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the project root directory (two levels up from this script).

    ``scripts/download_data.py``  ->  ``scripts/``  ->  project root
    """
    return Path(__file__).resolve().parent.parent


def load_config(project_root: Path) -> dict:
    """Load ``config.yaml`` from *project_root* and return its contents.

    Parameters
    ----------
    project_root:
        Absolute path to the project root directory.

    Returns
    -------
    dict
        Parsed YAML configuration.

    Raises
    ------
    FileNotFoundError
        If ``config.yaml`` does not exist at the given root.
    """
    config_path: Path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Run the scaffold step first."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def download_zip(url: str) -> bytes:
    """Download a ZIP archive from *url* and return its raw bytes.

    A simple progress indicator (dots every 1 MiB) is printed to stdout.

    Parameters
    ----------
    url:
        Full URL of the ZIP file to download.

    Returns
    -------
    bytes
        Raw ZIP content.

    Raises
    ------
    requests.HTTPError
        If the server returns a non-2xx status code.
    """
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    chunks: list[bytes] = []
    downloaded: int = 0
    chunk_size: int = 1024 * 1024  # 1 MiB

    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            chunks.append(chunk)
            downloaded += len(chunk)
            # Progress: one dot per MiB
            print(".", end="", flush=True)

    print(f"\nDownloaded {downloaded / 1024:.1f} KiB total.")
    return b"".join(chunks)


def _find_member(zf: zipfile.ZipFile, target: str, depth: int = 0) -> bytes | None:
    """Return the bytes of *target* inside *zf*, descending into nested ZIPs.

    The UCI archive currently ships ``student.zip`` (which itself contains the
    CSVs) instead of the CSVs directly, so a one-level unpack is not enough.
    """
    names: list[str] = zf.namelist()

    for name in names:
        if name.endswith(target):
            return zf.read(name)

    if depth >= 2:
        return None

    for name in names:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(BytesIO(zf.read(name))) as inner:
                data = _find_member(inner, target, depth + 1)
            if data is not None:
                return data

    return None


def extract_csv_files(zip_bytes: bytes, raw_dir: Path) -> None:
    """Extract only the target CSV files from an in-memory ZIP archive.

    Handles the CSVs being nested inside a further ZIP inside the archive.

    Parameters
    ----------
    zip_bytes:
        Raw bytes of the ZIP file.
    raw_dir:
        Destination directory where CSV files will be written.

    Raises
    ------
    KeyError
        If an expected CSV file is not found inside the ZIP.
    """
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        all_names: list[str] = zf.namelist()
        print(f"ZIP contents: {all_names}")

        for target in EXPECTED_CSV_FILES:
            data: bytes | None = _find_member(zf, target)
            if data is None:
                raise KeyError(
                    f"Expected '{target}' in ZIP but found only: {all_names}"
                )

            dest_path: Path = raw_dir / target
            dest_path.write_bytes(data)
            print(f"  Extracted -> {dest_path}")


def verify_files(raw_dir: Path) -> None:
    """Verify that all expected CSV files exist and are non-empty.

    Parameters
    ----------
    raw_dir:
        Directory containing the extracted CSV files.

    Raises
    ------
    FileNotFoundError
        If a required file is missing or empty.
    """
    for name in EXPECTED_CSV_FILES:
        path: Path = raw_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Verification failed: '{path}' is missing or empty."
            )
        print(f"  Verified  OK  {path}  ({path.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full dataset download pipeline."""
    root: Path = _project_root()
    print(f"Project root: {root}")

    # 1. Load configuration
    config: dict = load_config(root)
    raw_dir_rel: str = config.get("data", {}).get("raw_dir", "data/raw")
    raw_dir: Path = root / raw_dir_rel
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Raw data directory: {raw_dir}")

    # 2. Download
    zip_bytes: bytes = download_zip(DATASET_URL)

    # 3. Extract
    print("Extracting CSV files...")
    extract_csv_files(zip_bytes, raw_dir)

    # 4. Verify
    print("Verifying extracted files...")
    verify_files(raw_dir)

    print("\nDataset download complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

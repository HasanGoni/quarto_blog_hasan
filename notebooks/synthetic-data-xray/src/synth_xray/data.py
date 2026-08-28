"""Downloading and locating real GDXray images and ground truth.

The upstream GDXray dataset (https://github.com/computervision-xray-testing/GDXray)
was originally distributed via Dropbox links. As of this writing all of those
Dropbox links return Dropbox's "File Deleted" page (confirmed for every group,
not just Welds) -- see repo issues #1, #2, #4, #5. The repo maintainer's
current fix, posted directly on issues #4 and #5, is a replacement Google
Drive folder:
https://drive.google.com/drive/folders/1p8HGgwWj9l_j-6yWybMPEccQPEjAjqqr
GDXRAY_URLS below therefore points at the per-file Google Drive links resolved
from that folder, and downloading uses `gdown` (rather than plain `requests`)
because Google Drive requires a virus-scan-bypass/confirm-token dance for
files this large that `requests` alone doesn't handle.
"""
import pathlib
import zipfile

import gdown

GDXRAY_URLS = {
    "Welds": "https://drive.google.com/uc?id=1hKDBI-76cjwP-aUSFXuaTu2unpoyc-lg",
    "Castings": "https://drive.google.com/uc?id=1AcXT_E2-z_gBm3eSx_2VqsCHfdyIelrc",
}


def download_and_extract(group: str, dest_dir: pathlib.Path) -> pathlib.Path:
    """Download and extract a GDXray group zip into `dest_dir`, skipping if already extracted."""
    dest_dir = pathlib.Path(dest_dir)
    group_dir = dest_dir / group
    if group_dir.exists():
        return group_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{group}.zip"
    gdown.download(GDXRAY_URLS[group], str(zip_path), quiet=False)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    return group_dir


def find_series_dirs(group_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find every series subdirectory in an extracted GDXray group."""
    return sorted(p for p in pathlib.Path(group_dir).iterdir() if p.is_dir())


def find_images_in_series(series_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find every image file in a series directory."""
    return sorted(pathlib.Path(series_dir).glob("*.png"))


def find_groundtruth_file(series_dir: pathlib.Path) -> pathlib.Path | None:
    """Find the GDXray ground-truth text file in a series directory, if present.

    Real GDXray series ship this as plain `ground_truth.txt` (verified against
    the real Welds group, series W0001/W0002), not `ground_truth_<series>.txt`
    as originally assumed -- the glob below matches both spellings so it still
    recognizes a per-series-named file if one is ever present.
    """
    matches = sorted(pathlib.Path(series_dir).glob("ground_truth*.txt"))
    return matches[0] if matches else None

import zipfile
import pytest
from synth_xray.data import (
    GDXRAY_URLS,
    download_and_extract,
    find_series_dirs,
    find_images_in_series,
    find_groundtruth_file,
)


def _make_fake_group_zip(zip_path):
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Welds/W0001/W0001_0001.png", b"fake-png-bytes")
        zf.writestr("Welds/W0001/W0001_0002.png", b"fake-png-bytes")
        zf.writestr("Welds/W0001/ground_truth_W0001.txt", "1 10 10 20 20\n")
        zf.writestr("Welds/W0002/W0002_0001.png", b"fake-png-bytes")


def test_download_and_extract_skips_existing(tmp_path, monkeypatch):
    dest = tmp_path / "cache"
    dest.mkdir()
    group_dir = dest / "Welds"
    group_dir.mkdir()
    (group_dir / "already_here.txt").write_text("x")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not attempt download when group dir already exists")

    monkeypatch.setattr("synth_xray.data.requests.get", fail_if_called)
    result = download_and_extract("Welds", dest)
    assert result == group_dir


def test_find_series_and_images_and_groundtruth(tmp_path):
    zip_path = tmp_path / "Welds.zip"
    _make_fake_group_zip(zip_path)
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    group_dir = extract_dir / "Welds"

    series_dirs = find_series_dirs(group_dir)
    assert {d.name for d in series_dirs} == {"W0001", "W0002"}

    w0001 = group_dir / "W0001"
    images = find_images_in_series(w0001)
    assert len(images) == 2
    assert all(p.suffix == ".png" for p in images)

    gt = find_groundtruth_file(w0001)
    assert gt is not None
    assert gt.name == "ground_truth_W0001.txt"

    w0002 = group_dir / "W0002"
    assert find_groundtruth_file(w0002) is None


def test_gdxray_urls_are_https_links():
    # As of this writing, every Dropbox link in the upstream GDXray repo
    # (https://github.com/computervision-xray-testing/GDXray) returns
    # Dropbox's "File Deleted" page -- confirmed for all five groups, not
    # just Welds (see repo issues #1, #2, #4, #5). The maintainer's current
    # fix, posted directly on issues #4/#5, is a replacement Google Drive
    # folder, so GDXRAY_URLS now points there instead of Dropbox.
    assert "Welds" in GDXRAY_URLS
    assert GDXRAY_URLS["Welds"].startswith("https://drive.google.com/")

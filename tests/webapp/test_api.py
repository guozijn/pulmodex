from __future__ import annotations

from pathlib import Path
import json

from fastapi.testclient import TestClient
import SimpleITK as sitk

from src.webapp import api


def test_predict_accepts_zip_and_converts_before_enqueue(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path / "outputs"))

    converted = tmp_path / "uploads" / "converted.nii.gz"
    observed = {}

    def fake_prepare(upload_path: Path, stored_file: Path, seriesuid: str) -> Path:
        observed["prepare_args"] = (upload_path, stored_file, seriesuid)
        return converted

    class FakeTask:
        id = "job-zip"

    def fake_delay(scan_path: str, output_dir: str, seriesuid: str) -> FakeTask:
        observed["delay_args"] = (scan_path, output_dir, seriesuid)
        return FakeTask()

    monkeypatch.setattr(api, "_prepare_scan_input", fake_prepare)
    monkeypatch.setattr(api.predict_task, "delay", fake_delay)

    client = TestClient(api.app)
    response = client.post(
        "/predict",
        files={"file": ("series.zip", b"PK\x03\x04", "application/zip")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-zip"

    upload_path, stored_file, seriesuid = observed["prepare_args"]
    assert stored_file == upload_path / "series.zip"
    assert observed["delay_args"] == (str(converted), str(tmp_path / "outputs"), seriesuid)


def test_predict_accepts_nifti_gz_and_converts_before_enqueue(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path / "outputs"))

    uploaded = tmp_path / "uploads" / "scan.nii.gz"
    observed = {}

    def fake_prepare(upload_path: Path, stored_file: Path, seriesuid: str) -> Path:
        observed["prepare_args"] = (upload_path, stored_file, seriesuid)
        return uploaded

    class FakeTask:
        id = "job-nifti"

    def fake_delay(scan_path: str, output_dir: str, seriesuid: str) -> FakeTask:
        observed["delay_args"] = (scan_path, output_dir, seriesuid)
        return FakeTask()

    monkeypatch.setattr(api, "_prepare_scan_input", fake_prepare)
    monkeypatch.setattr(api.predict_task, "delay", fake_delay)

    client = TestClient(api.app)
    response = client.post(
        "/predict",
        files={"file": ("scan.nii.gz", b"nifti-data", "application/gzip")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-nifti"

    upload_path, stored_file, seriesuid = observed["prepare_args"]
    assert stored_file == upload_path / "scan.nii.gz"
    assert observed["delay_args"] == (str(uploaded), str(tmp_path / "outputs"), seriesuid)


def test_predict_rejects_unsupported_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", str(tmp_path / "uploads"))

    client = TestClient(api.app)
    response = client.post(
        "/predict",
        files={"file": ("scan.dcm", b"not-supported-here", "application/dicom")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .zip or .nii.gz uploads are supported"


def test_prepare_scan_input_passes_through_nifti_gz(monkeypatch, tmp_path):
    captured = {}
    expected = tmp_path / "upload" / "scan.nii.gz"

    def fake_validate_nifti(stored_file: Path) -> Path:
        captured["args"] = stored_file
        return expected

    monkeypatch.setattr(api, "_validate_nifti_upload", fake_validate_nifti)

    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    stored_file = upload_dir / "scan.nii.gz"
    stored_file.write_bytes(b"gz")

    resolved = api._prepare_scan_input(upload_dir, stored_file, "scan-123")

    assert resolved == expected
    assert captured["args"] == stored_file


def test_validate_nifti_upload_rejects_unreadable_file(tmp_path):
    bad_nifti = tmp_path / "scan.nii.gz"
    bad_nifti.write_bytes(b"not-a-real-nifti")

    try:
        api._validate_nifti_upload(bad_nifti)
    except api.HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "Uploaded .nii.gz file could not be read"
    else:
        raise AssertionError("Expected unreadable .nii.gz upload to raise HTTPException")


def test_body_crop_image_reduces_obvious_air_background():
    image = sitk.Image([32, 32, 12], sitk.sitkFloat32)
    image.SetSpacing((1.0, 1.0, 2.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image = sitk.Add(image, -1000.0)

    roi = sitk.Image([10, 12, 6], sitk.sitkFloat32)
    roi = sitk.Add(roi, 50.0)
    image = sitk.Paste(image, roi, roi.GetSize(), destinationIndex=[9, 8, 3])

    cropped = api._body_crop_image(image, margin_mm=2.0)

    assert cropped.GetSize()[0] < image.GetSize()[0]
    assert cropped.GetSize()[1] < image.GetSize()[1]
    assert cropped.GetOrigin() != image.GetOrigin()
    arr = sitk.GetArrayFromImage(cropped)
    assert arr.max() > -500.0


def test_list_scans_returns_empty_when_output_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path / "missing_outputs"))

    client = TestClient(api.app)
    response = client.get("/scans")

    assert response.status_code == 200
    assert response.json() == []


def test_delete_scan_removes_output_and_upload_dirs(monkeypatch, tmp_path):
    output_dir = tmp_path / "outputs"
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(api, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(api, "UPLOAD_DIR", str(upload_dir))

    seriesuid = "scan-123"
    scan_dir = output_dir / seriesuid
    staged_dir = upload_dir / seriesuid
    scan_dir.mkdir(parents=True)
    staged_dir.mkdir(parents=True)
    (scan_dir / "meta.json").write_text(json.dumps({"seriesuid": seriesuid}))
    (scan_dir / "report.json").write_text(json.dumps({"n_candidates_final": 1}))
    (staged_dir / "scan.zip").write_bytes(b"zip")

    client = TestClient(api.app)
    response = client.delete(f"/scans/{seriesuid}")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "seriesuid": seriesuid}
    assert not scan_dir.exists()
    assert not staged_dir.exists()


def test_delete_scan_404_for_missing_scan(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path / "outputs"))

    client = TestClient(api.app)
    response = client.delete("/scans/missing-scan")

    assert response.status_code == 404
    assert response.json()["detail"] == "Scan not found"

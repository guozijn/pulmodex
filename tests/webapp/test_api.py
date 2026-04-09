from __future__ import annotations

from pathlib import Path
import json

from fastapi.testclient import TestClient
import SimpleITK as sitk

from src.webapp import api


def test_predict_accepts_zip_and_converts_before_enqueue(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path / "outputs"))

    converted = tmp_path / "uploads" / "converted.mhd"
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


def test_predict_rejects_unsupported_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", str(tmp_path / "uploads"))

    client = TestClient(api.app)
    response = client.post(
        "/predict",
        files={"file": ("scan.dcm", b"not-supported-here", "application/dicom")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .zip uploads are supported"


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

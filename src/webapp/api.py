"""FastAPI application.

Endpoints:
  POST /predict          — upload DICOM/MHD, enqueue Celery job → {job_id}
  GET  /status/{job_id}  — poll job status → {state, progress, result}
  GET  /slices/{uid}/{view} → list of PNG slice paths or streamed image
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.webapp.tasks import celery_app, predict_task

app = FastAPI(title="Pulmodex API", version="0.1.0")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
UPLOAD_DIR = "/tmp/pulmodex_uploads"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a .mhd or .raw file and enqueue inference.

    For .mhd uploads, the corresponding .raw must be uploaded together
    or the .mhd must reference a self-contained format. For simplicity,
    upload the .mhd; the worker expects both files to be in the same dir.

    Returns:
        {"job_id": str}
    """
    seriesuid = str(uuid.uuid4())
    upload_path = Path(UPLOAD_DIR) / seriesuid
    upload_path.mkdir(parents=True, exist_ok=True)

    dest = upload_path / Path(file.filename).name  # prevent path traversal
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Enqueue
    task = predict_task.delay(str(dest), OUTPUT_DIR, seriesuid)
    return {"job_id": task.id, "seriesuid": seriesuid}


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Poll Celery task status.

    Returns:
        {"state": str, "progress": dict|None, "result": dict|None}
    """
    result = celery_app.AsyncResult(job_id)
    info = result.info if result.state == "PROGRESS" else None
    return {
        "state": result.state,
        "progress": info,
        "result": result.result if result.successful() else None,
    }


@app.get("/slices/{uid}/{view}")
async def get_slices(uid: str, view: str, idx: int = 0):
    """Return a rendered PNG slice.

    Args:
        uid: seriesuid
        view: axial | coronal | sagittal
        idx: slice index

    Returns:
        PNG image response
    """
    if view not in ("axial", "coronal", "sagittal"):
        raise HTTPException(400, "view must be axial|coronal|sagittal")

    img_path = Path(OUTPUT_DIR) / uid / "slices" / f"{view}_{idx:04d}.png"
    if not img_path.exists():
        raise HTTPException(404, f"Slice not found: {img_path}")

    return FileResponse(str(img_path), media_type="image/png")


@app.get("/slices/{uid}/{view}/index")
async def list_slices(uid: str, view: str):
    """List all available slice indices for a view."""
    if view not in ("axial", "coronal", "sagittal"):
        raise HTTPException(400, "view must be axial|coronal|sagittal")

    slice_dir = Path(OUTPUT_DIR) / uid / "slices"
    if not slice_dir.exists():
        raise HTTPException(404, "Slices not yet generated")

    files = sorted(slice_dir.glob(f"{view}_*.png"))
    indices = [int(f.stem.split("_")[1]) for f in files]
    return {"view": view, "indices": indices, "count": len(indices)}


@app.get("/report/{uid}")
async def get_report(uid: str):
    """Return the inference report JSON for a scan."""
    report_path = Path(OUTPUT_DIR) / uid / "report.json"
    if not report_path.exists():
        raise HTTPException(404, "Report not found")
    import json
    return JSONResponse(json.loads(report_path.read_text()))

import asyncio
import csv
import logging
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
INPUT_VIDEO_PATH = BASE_DIR / "input.mp4"
FINAL_VIDEO_PATH = BASE_DIR / "final_output.mp4"
RESULTS_CSV_PATH = BASE_DIR / "results.csv"
INDEX_PATH = BASE_DIR / "index.html"

app = FastAPI(title="ANPR Web API", version="1.0.0")
PROCESS_LOCK = asyncio.Lock()
JOBS: dict[str, dict[str, Any]] = {}
MAX_LOG_LINES = 400


@app.get("/")
def index() -> FileResponse:
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(
        INDEX_PATH,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)) -> dict[str, Any]:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 uploads are supported")

    content = await video.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    run_id = uuid.uuid4().hex[:12]

    JOBS[run_id] = {
        "run_id": run_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Queued",
        "logs": [],
        "error": None,
        "created_at": time.time(),
        "duration_seconds": None,
        "rows": 0,
        "video_url": "/video/final",
        "results_url": "/results",
    }

    asyncio.create_task(_process_job(run_id, content))

    return {
        "status": "queued",
        "run_id": run_id,
        "status_url": f"/jobs/{run_id}",
        "video_url": "/video/final",
        "results_url": "/results",
    }


@app.get("/jobs/{run_id}")
def get_job_status(run_id: str) -> JSONResponse:
    job = JOBS.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown run_id")

    return JSONResponse(
        content=job,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/video/final")
def get_final_video() -> FileResponse:
    if not FINAL_VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail="final_output.mp4 not found")
    return FileResponse(
        FINAL_VIDEO_PATH,
        media_type="video/mp4",
        filename="final_output.mp4",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/results")
def get_results() -> JSONResponse:
    if not RESULTS_CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="results.csv not found")

    rows: list[dict[str, Any]] = []
    with RESULTS_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_raw = str(row.get("frame_nmr", "")).strip()
            car_raw = str(row.get("car_id", "")).strip()
            number_raw = str(row.get("license_number", "")).strip()
            score_raw = str(row.get("license_number_score", "")).strip()

            if not frame_raw or not car_raw:
                continue

            try:
                frame_nmr = int(float(frame_raw))
                car_id = int(float(car_raw))
            except ValueError:
                continue

            try:
                license_number_score = float(score_raw) if score_raw else None
            except ValueError:
                license_number_score = None

            rows.append(
                {
                    "frame_nmr": frame_nmr,
                    "car_id": car_id,
                    "license_number": number_raw,
                    "license_number_score": license_number_score,
                }
            )

    return JSONResponse(
        content={"rows": rows, "count": len(rows)},
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _append_job_log(job: dict[str, Any], line: str) -> None:
    line = line.rstrip("\n")
    if not line:
        return
    logs = job["logs"]
    logs.append(line)
    if len(logs) > MAX_LOG_LINES:
        del logs[: len(logs) - MAX_LOG_LINES]


def _extract_progress(line: str) -> float | None:
    import re

    m = re.search(r"\[\s*(\d+(?:\.\d+)?)%\]", line)
    if m:
        try:
            return max(0.0, min(100.0, float(m.group(1))))
        except ValueError:
            return None
    return None


def _find_ffmpeg_exe() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _reencode_for_web(path: Path) -> tuple[bool, str]:
    ffmpeg_exe = _find_ffmpeg_exe()
    if not ffmpeg_exe:
        return False, "ffmpeg not found; kept original video encoding"

    tmp_path = path.with_name(f"{path.stem}.webtmp{path.suffix}")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]

    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp_path.exists():
        detail = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
        return False, f"web re-encode failed: {detail[-300:]}"

    tmp_path.replace(path)
    return True, "web re-encode successful (H.264 yuv420p)"


async def _process_job(run_id: str, video_bytes: bytes) -> None:
    job = JOBS[run_id]
    job["status"] = "running"
    job["message"] = "Starting pipeline"

    async with PROCESS_LOCK:
        started_at = time.time()
        try:
            INPUT_VIDEO_PATH.write_bytes(video_bytes)

            for stale in (FINAL_VIDEO_PATH, RESULTS_CSV_PATH, BASE_DIR / "results_raw.csv", BASE_DIR / "output.mp4"):
                if stale.exists():
                    stale.unlink()

            command = [
                sys.executable,
                "-u",
                "pipeline.py",
                "--source",
                str(INPUT_VIDEO_PATH),
                "--output",
                str(BASE_DIR / "output.mp4"),
                "--csv",
                str(RESULTS_CSV_PATH),
                "--final",
                str(FINAL_VIDEO_PATH),
            ]

            def _run_and_stream() -> int:
                proc = subprocess.Popen(
                    command,
                    cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    _append_job_log(job, line)
                    prog = _extract_progress(line)
                    if prog is not None:
                        job["progress"] = prog
                        job["message"] = f"Processing... {prog:.1f}%"
                return proc.wait()

            return_code = await asyncio.to_thread(_run_and_stream)
            if return_code != 0:
                job["status"] = "failed"
                job["error"] = "Pipeline exited with non-zero status"
                job["message"] = "Pipeline failed"
                return

            if not FINAL_VIDEO_PATH.exists() or not RESULTS_CSV_PATH.exists():
                job["status"] = "failed"
                job["error"] = "Expected outputs not found"
                job["message"] = "Pipeline failed"
                return

            ok, msg = _reencode_for_web(FINAL_VIDEO_PATH)
            _append_job_log(job, f"[VIDEO] {msg}")
            if not ok:
                _append_job_log(job, "[VIDEO] Warning: browser playback may fail with original codec")

            # Count rows for visibility in UI.
            row_count = 0
            with RESULTS_CSV_PATH.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for _ in reader:
                    row_count += 1

            job["rows"] = row_count
            job["progress"] = 100.0
            job["status"] = "completed"
            job["message"] = "Processing complete"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["message"] = "Pipeline failed"
        finally:
            job["duration_seconds"] = round(time.time() - started_at, 2)


if __name__ == "__main__":
    import uvicorn

    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("uvicorn").setLevel(logging.ERROR)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False,
        log_level="error",
    )

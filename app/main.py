import json
import math
import os
import re
import statistics
import subprocess
import traceback
import uuid
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Any

import numpy as np
import soundfile as sf
import torch
import whisper
from demucs.apply import apply_model
from demucs.pretrained import get_model
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.chord.chroma_engine import infer_chroma_for_timeline
from app.musicxml_export import build_measure_plan, write_musicxml
from app.stem_support import load_stem_manifest, prepare_multitrack_stems

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("APP_DATA_DIR", BASE_DIR / "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4"}
TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 2

YOUTUBE_URL_PATTERN = re.compile(r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/", re.IGNORECASE)
YTDLP_AUDIO_FORMAT = os.getenv("YTDLP_AUDIO_FORMAT", "bestaudio[ext=m4a]/bestaudio/best")

CHORD_WINDOW_SECONDS = float(os.getenv("CHORD_WINDOW_SECONDS", "0.5"))
MIN_ACTIVE_NOTES_FOR_CHORD = int(os.getenv("MIN_ACTIVE_NOTES_FOR_CHORD", "2"))
MIN_NOTE_DURATION_SECONDS = float(os.getenv("MIN_NOTE_DURATION_SECONDS", "0.12"))
MIN_SEGMENT_SECONDS = float(os.getenv("MIN_SEGMENT_SECONDS", "0.35"))
MAX_SEGMENT_SECONDS = float(os.getenv("MAX_SEGMENT_SECONDS", "8.0"))
STAGE6_CHROMA_ENABLED = os.getenv("STAGE6_CHROMA_ENABLED", "1") != "0"
STAGE6_CHROMA_CONFIDENCE_THRESHOLD = float(os.getenv("STAGE6_CHROMA_CONFIDENCE_THRESHOLD", "0.78"))
STAGE6_CHROMA_OVERRIDE_THRESHOLD = float(os.getenv("STAGE6_CHROMA_OVERRIDE_THRESHOLD", "0.88"))

RUNTIME_STAGE6_CHROMA_CONFIDENCE_THRESHOLD = STAGE6_CHROMA_CONFIDENCE_THRESHOLD
RUNTIME_STAGE6_CHROMA_OVERRIDE_THRESHOLD = STAGE6_CHROMA_OVERRIDE_THRESHOLD

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Demucs + Whisper + Hybrid Chord Inference + Stage 6 Chroma Fusion")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

_DEMUCS_MODEL: Any = None
_WHISPER_MODEL: Any = None

JOB_STORE: dict[str, dict[str, Any]] = {}
JOB_STORE_LOCK = threading.Lock()


def create_job_state(job_id: str) -> dict[str, Any]:
    state = {
        "job_id": job_id,
        "created_at": time.time(),
        "history": [],
        "queue": queue.Queue(),
        "done": False,
        "result": None,
        "error": None,
    }
    with JOB_STORE_LOCK:
        JOB_STORE[job_id] = state
    return state


def get_job_state(job_id: str) -> dict[str, Any] | None:
    with JOB_STORE_LOCK:
        return JOB_STORE.get(job_id)


def publish_job_event(job_id: str, stage: str, message: str, percent: float | None = None, status: str = "running") -> None:
    state = get_job_state(job_id)
    if not state:
        return
    event = {
        "seq": len(state["history"]) + 1,
        "time": round(time.time(), 3),
        "stage": stage,
        "message": message,
        "percent": percent,
        "status": status,
    }
    state["history"].append(event)
    state["queue"].put(event)


def mark_job_complete(job_id: str, result: dict[str, Any]) -> None:
    state = get_job_state(job_id)
    if not state:
        return
    state["result"] = result
    state["done"] = True
    state["queue"].put(None)


def mark_job_failed(job_id: str, error_payload: dict[str, Any]) -> None:
    state = get_job_state(job_id)
    if not state:
        return
    state["error"] = error_payload
    state["done"] = True
    state["queue"].put(None)


def report_progress(progress_cb, stage: str, message: str, percent: float | None = None, status: str = "running") -> None:
    if progress_cb:
        progress_cb(stage=stage, message=message, percent=percent, status=status)


def get_demucs_model() -> Any:
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is None:
        model = get_model(name="htdemucs")
        model.cpu()
        model.eval()
        _DEMUCS_MODEL = model
    return _DEMUCS_MODEL


def get_whisper_model() -> Any:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("base")
    return _WHISPER_MODEL


def get_stage6_runtime_settings() -> dict[str, float]:
    return {
        "confidence_threshold": round(float(RUNTIME_STAGE6_CHROMA_CONFIDENCE_THRESHOLD), 4),
        "override_threshold": round(float(RUNTIME_STAGE6_CHROMA_OVERRIDE_THRESHOLD), 4),
    }


class Stage6SettingsUpdate(BaseModel):
    confidence_threshold: float | None = None
    override_threshold: float | None = None


class MusicXMLExportRequest(BaseModel):
    time_signature: str = "4/4"
    title: str | None = None


def clamp_stage6_threshold(value: float, fallback: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return fallback
    return max(0.0, min(1.0, numeric))


@app.get("/stage6/settings")
def get_stage6_settings():
    return {
        "enabled": STAGE6_CHROMA_ENABLED,
        **get_stage6_runtime_settings(),
    }


@app.post("/stage6/settings")
def update_stage6_settings(payload: Stage6SettingsUpdate):
    global RUNTIME_STAGE6_CHROMA_CONFIDENCE_THRESHOLD, RUNTIME_STAGE6_CHROMA_OVERRIDE_THRESHOLD

    if payload.confidence_threshold is not None:
        RUNTIME_STAGE6_CHROMA_CONFIDENCE_THRESHOLD = clamp_stage6_threshold(
            payload.confidence_threshold,
            RUNTIME_STAGE6_CHROMA_CONFIDENCE_THRESHOLD,
        )

    if payload.override_threshold is not None:
        RUNTIME_STAGE6_CHROMA_OVERRIDE_THRESHOLD = clamp_stage6_threshold(
            payload.override_threshold,
            RUNTIME_STAGE6_CHROMA_OVERRIDE_THRESHOLD,
        )

    if RUNTIME_STAGE6_CHROMA_OVERRIDE_THRESHOLD < RUNTIME_STAGE6_CHROMA_CONFIDENCE_THRESHOLD:
        RUNTIME_STAGE6_CHROMA_OVERRIDE_THRESHOLD = RUNTIME_STAGE6_CHROMA_CONFIDENCE_THRESHOLD

    return {
        "ok": True,
        "enabled": STAGE6_CHROMA_ENABLED,
        **get_stage6_runtime_settings(),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/jobs/{job_id}/stems")
def get_job_stems(job_id: str):
    job_dir = OUTPUTS_DIR / Path(job_id).name
    manifest = load_stem_manifest(job_dir)
    if manifest:
        return JSONResponse(content=manifest)

    fallback_stems = []
    vocals_path = job_dir / "vocals.wav"
    instrumental_path = job_dir / "instrumental.wav"
    if vocals_path.is_file():
        fallback_stems.append({"id": "vocals", "label": "Vocals", "url": f"/media/{Path(job_id).name}/vocals.wav"})
    if instrumental_path.is_file():
        fallback_stems.append({"id": "instrumental", "label": "Instrumental", "url": f"/media/{Path(job_id).name}/instrumental.wav"})

    if not fallback_stems:
        raise HTTPException(status_code=404, detail="Stem manifest not found")

    return JSONResponse(content={
        "job_id": Path(job_id).name,
        "stems": fallback_stems,
        "instrumental_url": f"/media/{Path(job_id).name}/instrumental.wav",
        "vocals_url": f"/media/{Path(job_id).name}/vocals.wav",
        "manifest_url": f"/api/jobs/{Path(job_id).name}/stems",
    })


@app.get("/media/{job_id}/{filename:path}")
def serve_media_file(job_id: str, filename: str, request: Request):
    safe_job_id = Path(job_id).name
    relative_path = Path(filename)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise HTTPException(status_code=400, detail="Invalid media path")
    file_path = (OUTPUTS_DIR / safe_job_id / relative_path).resolve()
    job_root = (OUTPUTS_DIR / safe_job_id).resolve()
    try:
        file_path.relative_to(job_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid media path")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")

    suffix = file_path.suffix.lower()
    media_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".mid": "audio/midi",
        ".midi": "audio/midi",
        ".json": "application/json",
        ".txt": "text/plain; charset=utf-8",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    range_header = request.headers.get("range")
    if suffix not in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"} or not range_header:
        response = FileResponse(file_path, media_type=media_type)
        response.headers["Accept-Ranges"] = "bytes"
        return response

    file_size = file_path.stat().st_size
    try:
        units, range_spec = range_header.strip().split("=", 1)
        if units != "bytes":
            raise ValueError("Unsupported range unit")
        start_str, end_str = range_spec.split("-", 1)
        if start_str == "":
            suffix_length = int(end_str)
            if suffix_length <= 0:
                raise ValueError("Invalid suffix range")
            start = max(file_size - suffix_length, 0)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
    except Exception:
        raise HTTPException(status_code=416, detail="Invalid range header")

    if start < 0 or end < start or start >= file_size:
        raise HTTPException(status_code=416, detail="Requested range not satisfiable")

    end = min(end, file_size - 1)
    content_length = end - start + 1

    def iter_file_chunks():
        with open(file_path, "rb") as handle:
            handle.seek(start)
            remaining = content_length
            chunk_size = 1024 * 1024
            while remaining > 0:
                data = handle.read(min(chunk_size, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(content_length),
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(iter_file_chunks(), status_code=206, media_type=media_type, headers=headers)


def is_supported_youtube_url(value: str) -> bool:
    if not value:
        return False
    return bool(YOUTUBE_URL_PATTERN.match(value.strip()))


def normalize_youtube_url(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if not re.match(r"^https?://", raw, flags=re.IGNORECASE):
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in {"youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}:
        raise HTTPException(status_code=400, detail="Only YouTube links are supported.")

    if host == "youtu.be":
        video_id = parsed.path.strip("/")
    else:
        query = parse_qs(parsed.query)
        video_id = (query.get("v") or [""])[0]
    if not video_id:
        raise HTTPException(status_code=400, detail="Unable to read the YouTube video ID from the link.")
    return raw


def download_youtube_audio(youtube_url: str, output_dir: Path, debug_lines: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = output_dir / "youtube_source.%(ext)s"
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--restrict-filenames",
        "-f",
        YTDLP_AUDIO_FORMAT,
        "-o",
        str(output_template),
        youtube_url,
    ]
    debug_lines.append("youtube_url=" + youtube_url)
    debug_lines.append("yt_dlp_command=" + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    debug_lines.append(f"yt_dlp_returncode={result.returncode}")
    if result.stdout:
        debug_lines.append("yt_dlp_stdout=" + result.stdout[-4000:])
    if result.stderr:
        debug_lines.append("yt_dlp_stderr=" + result.stderr[-4000:])

    candidates = sorted(output_dir.glob("youtube_source.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.suffix.lower() in ALLOWED_EXTENSIONS or candidate.suffix.lower() == ".wav":
            return candidate

    raise RuntimeError(
        "YouTube audio download failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def run_pipeline(
    *,
    file_bytes: bytes | None,
    original_name: str,
    source_type: str,
    normalized_youtube_url: str = "",
    job_id: str | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    current_job_id = job_id or uuid.uuid4().hex[:12]
    job_upload_dir = UPLOADS_DIR / current_job_id
    job_output_dir = OUTPUTS_DIR / current_job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(original_name).suffix.lower() or ".wav"
    raw_upload_path = job_upload_dir / f"input_original{ext}"
    input_wav_path = job_upload_dir / "prepared_input.wav"
    debug_path = job_output_dir / "debug.txt"

    debug_lines: list[str] = []
    debug_lines.append(f"job_id={current_job_id}")
    debug_lines.append(f"source_type={source_type}")
    debug_lines.append(f"original_name={original_name}")

    try:
        report_progress(progress_cb, "init", "Preparing job.", 2)

        if source_type == "upload":
            file_size = len(file_bytes or b"")
            if file_size > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit")
            raw_upload_path.write_bytes(file_bytes or b"")
            debug_lines.append(f"upload_size_bytes={file_size}")
            report_progress(progress_cb, "upload", "Audio upload received.", 8)
        else:
            report_progress(progress_cb, "download", "Downloading audio from YouTube.", 8)
            raw_upload_path = download_youtube_audio(normalized_youtube_url, job_upload_dir, debug_lines)
            report_progress(progress_cb, "download", "YouTube audio downloaded.", 14)

        debug_lines.append(f"raw_upload_path={raw_upload_path}")
        debug_lines.append(f"input_wav_path={input_wav_path}")
        debug_lines.append(f"chord_window_seconds={CHORD_WINDOW_SECONDS}")
        debug_lines.append(f"min_active_notes_for_chord={MIN_ACTIVE_NOTES_FOR_CHORD}")
        debug_lines.append(f"min_segment_seconds={MIN_SEGMENT_SECONDS}")
        debug_lines.append(f"max_segment_seconds={MAX_SEGMENT_SECONDS}")
        debug_lines.append(f"stage6_chroma_enabled={STAGE6_CHROMA_ENABLED}")
        debug_lines.append(f"stage6_chroma_confidence_threshold={RUNTIME_STAGE6_CHROMA_CONFIDENCE_THRESHOLD}")
        debug_lines.append(f"stage6_chroma_override_threshold={RUNTIME_STAGE6_CHROMA_OVERRIDE_THRESHOLD}")
        debug_lines.append("adaptive_subdivision=enabled")

        report_progress(progress_cb, "prepare", "Normalizing source audio with ffmpeg.", 18)
        convert_to_wav(
            input_path=raw_upload_path,
            output_path=input_wav_path,
            sample_rate=TARGET_SAMPLE_RATE,
            channels=TARGET_CHANNELS,
            debug_lines=debug_lines,
        )

        report_progress(progress_cb, "stems", "Separating vocals and instrument stems with Demucs.", 28)
        vocals_path, instrumental_path, stem_manifest = separate_with_demucs(
            input_wav_path=input_wav_path,
            output_dir=job_output_dir,
            debug_lines=debug_lines,
        )

        vocals_preview_path = job_output_dir / "vocals_preview.mp3"
        instrumental_preview_path = job_output_dir / "instrumental_preview.mp3"

        report_progress(progress_cb, "preview", "Creating preview MP3 files.", 42)
        create_preview_mp3(vocals_path, vocals_preview_path, debug_lines)
        create_preview_mp3(instrumental_path, instrumental_preview_path, debug_lines)

        report_progress(progress_cb, "transcript", "Transcribing vocals with Whisper.", 52)
        transcription = transcribe_vocals(vocals_path, debug_lines)

        report_progress(progress_cb, "chords", "Running chord extraction and Stage 6 fusion.", 66)
        chord_result: dict[str, Any]
        try:
            chord_result = infer_chords_from_instrumental(
                instrumental_path,
                job_output_dir,
                debug_lines,
                progress_cb=progress_cb,
            )
        except Exception as chord_exc:
            debug_lines.append("stage3_error=" + str(chord_exc))
            debug_lines.append(traceback.format_exc())
            chord_result = {
                "success": False,
                "error": str(chord_exc),
                "midi_url": None,
                "chord_json_url": None,
                "beat_grid_url": None,
                "timeline": [],
                "meta": {},
            }

        report_progress(progress_cb, "lead_sheet", "Building lead sheet output.", 92)
        lead_sheet = build_lead_sheet(
            lyric_segments=transcription["segments"],
            chord_timeline=chord_result.get("timeline", []),
        )

        payload = {
            "success": True,
            "job_id": current_job_id,
            "vocals_url": f"/media/{current_job_id}/vocals_preview.mp3",
            "instrumental_url": f"/media/{current_job_id}/instrumental_preview.mp3",
            "vocals_wav_url": f"/media/{current_job_id}/vocals.wav",
            "instrumental_wav_url": f"/media/{current_job_id}/instrumental.wav",
            "stems": stem_manifest.get("stems", []),
            "stem_manifest_url": f"/api/jobs/{current_job_id}/stems",
            "debug_url": f"/media/{current_job_id}/debug.txt",
            "lyrics": transcription["text"],
            "segments": transcription["segments"],
            "chords": chord_result,
            "lead_sheet": lead_sheet,
            "meta": {
                "whisper_model": transcription["model_name"],
                "language": transcription.get("language"),
                "segment_count": len(transcription["segments"]),
                "source_type": source_type,
                "youtube_url": normalized_youtube_url or None,
            },
        }

        debug_path.write_text("\n".join(debug_lines), encoding="utf-8")
        report_progress(progress_cb, "complete", "Processing finished.", 100, status="completed")
        return payload

    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            debug_path.write_text(tb, encoding="utf-8")
        except Exception:
            pass
        report_progress(progress_cb, "error", f"Processing failed: {exc}", 100, status="failed")
        raise


def validate_process_request(file: UploadFile | None, youtube_url: str | None) -> tuple[UploadFile | None, str, str, str]:
    normalized_youtube_url = ""
    uploaded_file = file if file and (file.filename or "").strip() else None
    youtube_value = (youtube_url or "").strip()

    if uploaded_file is None and not youtube_value:
        raise HTTPException(status_code=400, detail="Please upload an audio file or provide a YouTube link.")

    if uploaded_file is not None and youtube_value:
        raise HTTPException(status_code=400, detail="Please choose only one source: uploaded audio or YouTube link.")

    if uploaded_file is not None:
        original_name = uploaded_file.filename or "input_audio"
        ext = Path(original_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        source_type = "upload"
    else:
        normalized_youtube_url = normalize_youtube_url(youtube_value)
        original_name = "youtube_audio.wav"
        source_type = "youtube"

    return uploaded_file, normalized_youtube_url, original_name, source_type


@app.post("/process")
async def process_audio(file: UploadFile | None = File(None), youtube_url: str | None = Form(None)):
    uploaded_file, normalized_youtube_url, original_name, source_type = validate_process_request(file, youtube_url)
    file_bytes = await uploaded_file.read() if uploaded_file is not None else None
    payload = run_pipeline(
        file_bytes=file_bytes,
        original_name=original_name,
        source_type=source_type,
        normalized_youtube_url=normalized_youtube_url,
    )
    return JSONResponse(payload)


@app.post("/process/start")
async def start_process_audio(file: UploadFile | None = File(None), youtube_url: str | None = Form(None)):
    uploaded_file, normalized_youtube_url, original_name, source_type = validate_process_request(file, youtube_url)
    file_bytes = await uploaded_file.read() if uploaded_file is not None else None

    job_id = uuid.uuid4().hex[:12]
    create_job_state(job_id)
    publish_job_event(job_id, "queued", "Job queued.", 0, status="queued")

    def runner() -> None:
        try:
            result = run_pipeline(
                file_bytes=file_bytes,
                original_name=original_name,
                source_type=source_type,
                normalized_youtube_url=normalized_youtube_url,
                job_id=job_id,
                progress_cb=lambda **event: publish_job_event(job_id, **event),
            )
            mark_job_complete(job_id, result)
        except HTTPException as exc:
            error_payload = {"message": exc.detail, "status_code": exc.status_code}
            publish_job_event(job_id, "error", str(exc.detail), 100, status="failed")
            mark_job_failed(job_id, error_payload)
        except Exception as exc:
            error_payload = {
                "message": "Processing failed.",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            publish_job_event(job_id, "error", f"Processing failed: {exc}", 100, status="failed")
            mark_job_failed(job_id, error_payload)

    threading.Thread(target=runner, daemon=True).start()
    return {
        "ok": True,
        "job_id": job_id,
        "progress_url": f"/progress/{job_id}",
        "result_url": f"/result/{job_id}",
    }


@app.get("/progress/{job_id}")
def stream_progress(job_id: str):
    state = get_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        history = list(state["history"])
        last_seq = 0
        for item in history:
            last_seq = max(last_seq, int(item.get("seq", 0)))
            yield f"data: {json.dumps(item)}\n\n"

        while True:
            try:
                item = state["queue"].get(timeout=10)
            except queue.Empty:
                yield ": keep-alive\n\n"
                if state.get("done"):
                    break
                continue

            if item is None:
                if state.get("done"):
                    break
                continue

            seq = int(item.get("seq", 0))
            if seq <= last_seq:
                continue
            last_seq = seq
            yield f"data: {json.dumps(item)}\n\n"
            if state.get("done") and state["queue"].empty():
                break

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.get("/result/{job_id}")
def get_job_result(job_id: str):
    state = get_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.get("result") is not None:
        return JSONResponse(state["result"])
    if state.get("error") is not None:
        return JSONResponse(state["error"], status_code=500)
    return JSONResponse({"ok": False, "status": "processing"}, status_code=202)


def normalize_time_signature(value: str | None) -> str:
    raw = str(value or "4/4").strip() or "4/4"
    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", raw)
    if not match:
        return "4/4"
    beats = max(1, int(match.group(1)))
    beat_type = int(match.group(2))
    if beat_type not in {1, 2, 4, 8, 16, 32}:
        beat_type = 4
    return f"{beats}/{beat_type}"


def load_job_beat_grid(job_id: str) -> dict[str, Any]:
    beat_grid_path = OUTPUTS_DIR / Path(job_id).name / "beat_grid.json"
    if not beat_grid_path.is_file():
        return {}
    try:
        return json.loads(beat_grid_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_default_musicxml_title(result_payload: dict[str, Any], job_id: str) -> str:
    meta = result_payload.get("meta") or {}
    if meta.get("youtube_url"):
        return "YouTube Lead Sheet"
    return f"Lead Sheet {job_id}"


@app.post("/export/musicxml/{job_id}")
def export_musicxml(job_id: str, payload: MusicXMLExportRequest):
    state = get_job_state(job_id)
    if not state or state.get("result") is None:
        raise HTTPException(status_code=404, detail="Completed job not found")

    result_payload = state.get("result") or {}
    chords = ((result_payload.get("chords") or {}).get("timeline")) or []
    segments = result_payload.get("segments") or []
    beat_grid = load_job_beat_grid(job_id)

    time_signature = normalize_time_signature(payload.time_signature)
    tempo_bpm = float((((result_payload.get("chords") or {}).get("meta") or {}).get("tempo_bpm")) or beat_grid.get("tempo_bpm") or 90.0)
    title = (payload.title or "").strip() or build_default_musicxml_title(result_payload, job_id)

    measures = build_measure_plan(
        transcription_segments=segments,
        chord_timeline=chords,
        beat_grid=beat_grid,
        time_signature=time_signature,
        tempo_bpm=tempo_bpm,
    )
    if not measures:
        raise HTTPException(status_code=400, detail="No data available to export MusicXML")

    output_path = OUTPUTS_DIR / Path(job_id).name / "lead_sheet.musicxml"
    write_musicxml(
        output_path=output_path,
        title=title,
        measures=measures,
        time_signature=time_signature,
        tempo_bpm=tempo_bpm,
    )

    return {
        "ok": True,
        "job_id": job_id,
        "time_signature": time_signature,
        "tempo_bpm": tempo_bpm,
        "download_url": f"/media/{Path(job_id).name}/lead_sheet.musicxml",
    }


def convert_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    channels: int,
    debug_lines: list[str],
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-vn",
        str(output_path),
    ]
    debug_lines.append("ffmpeg_command=" + " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    debug_lines.append(f"ffmpeg_returncode={result.returncode}")
    if result.stdout:
        debug_lines.append("ffmpeg_stdout=" + result.stdout[-4000:])
    if result.stderr:
        debug_lines.append("ffmpeg_stderr=" + result.stderr[-4000:])

    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            "ffmpeg conversion failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

def create_preview_mp3(input_path: Path, output_path: Path, debug_lines: list[str]) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(output_path),
    ]
    debug_lines.append("preview_mp3_command=" + " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    debug_lines.append(f"preview_mp3_returncode={result.returncode}")
    if result.stdout:
        debug_lines.append("preview_mp3_stdout=" + result.stdout[-4000:])
    if result.stderr:
        debug_lines.append("preview_mp3_stderr=" + result.stderr[-4000:])

    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            "Preview MP3 conversion failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

def load_audio_for_demucs(wav_path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = audio.T
    return audio, sr


def write_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
        audio = audio.T

    sf.write(str(path), audio, sr)


def separate_with_demucs(input_wav_path: Path, output_dir: Path, debug_lines: list[str]) -> tuple[Path, Path, dict[str, Any]]:
    model = get_demucs_model()
    waveform, sr = load_audio_for_demucs(input_wav_path)

    debug_lines.append(f"demucs_input_shape={waveform.shape}")
    debug_lines.append(f"demucs_input_sr={sr}")

    wav_tensor = torch.from_numpy(waveform).float()
    sources = apply_model(model, wav_tensor[None], device="cpu")[0]

    source_names = list(model.sources)
    debug_lines.append("demucs_sources=" + ",".join(source_names))

    if "vocals" not in source_names:
        raise RuntimeError("Demucs model sources do not include vocals")

    stems_output_dir = output_dir / "demucs_raw"
    stems_output_dir.mkdir(parents=True, exist_ok=True)

    source_map = {name: sources[idx].detach().cpu().numpy() for idx, name in enumerate(source_names)}
    for name, stem_audio in source_map.items():
        stem_path = stems_output_dir / f"{name}.wav"
        write_audio(stem_path, np.asarray(stem_audio, dtype=np.float32), sr)
        debug_lines.append(f"demucs_stem_{name}={stem_path}")

    stem_manifest = prepare_multitrack_stems(
        job_dir=output_dir,
        demucs_output_root=stems_output_dir,
        job_id=output_dir.name,
    )

    vocals_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "instrumental.wav"

    if not vocals_path.exists() or not instrumental_path.exists():
        raise RuntimeError("Expected vocals.wav and instrumental.wav were not generated from stems")

    debug_lines.append(f"vocals_path={vocals_path}")
    debug_lines.append(f"instrumental_path={instrumental_path}")
    debug_lines.append(f"stem_manifest_path={output_dir / 'stems' / 'stem_manifest.json'}")
    debug_lines.append(f"stem_count={len(stem_manifest.get('stems', []))}")
    return vocals_path, instrumental_path, stem_manifest


def transcribe_vocals(vocals_path: Path, debug_lines: list[str]) -> dict[str, Any]:
    model = get_whisper_model()
    debug_lines.append("whisper_model=base")

    result = model.transcribe(str(vocals_path), fp16=False, verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append(
            {
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
                "text": (seg.get("text") or "").strip(),
            }
        )

    debug_lines.append(f"transcript_length={len((result.get('text') or '').strip())}")
    debug_lines.append(f"segment_count={len(segments)}")

    return {
        "text": (result.get("text") or "").strip(),
        "segments": segments,
        "language": result.get("language"),
        "model_name": "base",
    }


def build_lead_sheet(
    lyric_segments: list[dict[str, Any]],
    chord_timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    if not lyric_segments:
        return {"text": "", "blocks": [], "chordpro": ""}

    usable_chords = [
        item for item in (chord_timeline or [])
        if float(item.get("end", 0.0)) > float(item.get("start", 0.0))
    ]

    blocks: list[dict[str, Any]] = []

    for seg in lyric_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_text = " ".join((seg.get("text") or "").split()).strip()

        if not seg_text:
            continue

        overlapping = []
        for chord in usable_chords:
            chord_start = float(chord.get("start", 0.0))
            chord_end = float(chord.get("end", chord_start))
            if chord_end <= seg_start or chord_start >= seg_end:
                continue

            overlap_start = max(seg_start, chord_start)
            overlap_end = min(seg_end, chord_end)
            if overlap_end > overlap_start:
                overlapping.append({
                    "start": overlap_start,
                    "end": overlap_end,
                    "label": (chord.get("label") or "N.C.").strip() or "N.C.",
                })

        if not overlapping:
            overlapping = [{
                "start": seg_start,
                "end": seg_end,
                "label": "N.C.",
            }]

        merged: list[dict[str, Any]] = []
        for item in overlapping:
            if merged and merged[-1]["label"] == item["label"]:
                merged[-1]["end"] = item["end"]
            else:
                merged.append(dict(item))

        words = seg_text.split()
        if not words:
            continue

        total_words = len(words)
        total_duration = max(0.001, seg_end - seg_start)

        counts: list[int] = []
        running = 0
        for idx, item in enumerate(merged):
            duration = max(0.001, float(item["end"]) - float(item["start"]))
            if idx == len(merged) - 1:
                count = total_words - running
            else:
                ratio = duration / total_duration
                count = max(1, int(round(total_words * ratio)))
                remaining_spans = len(merged) - idx - 1
                max_allowed = total_words - running - remaining_spans
                count = min(count, max_allowed)
            counts.append(count)
            running += count

        diff = total_words - sum(counts)
        if counts:
            counts[-1] += diff

        pairs: list[dict[str, str]] = []
        cursor = 0
        for item, count in zip(merged, counts):
            chunk_words = words[cursor:cursor + max(0, count)]
            cursor += max(0, count)
            chunk_text = " ".join(chunk_words).strip()
            if not chunk_text:
                continue
            pairs.append({
                "chord": item["label"],
                "lyrics": chunk_text,
            })

        if not pairs:
            pairs = [{"chord": merged[0]["label"], "lyrics": seg_text}]

        chord_line, lyric_line = render_chord_lyric_lines(pairs)

        blocks.append({
            "start": round(seg_start, 2),
            "end": round(seg_end, 2),
            "chord_line": chord_line,
            "lyric_line": lyric_line,
            "pairs": pairs,
        })

    lead_sheet_text = "\n\n".join(
        f"{block['chord_line']}\n{block['lyric_line']}" for block in blocks
    )

    return {
        "text": lead_sheet_text,
        "blocks": blocks,
        "chordpro": build_chordpro(blocks),
    }


def render_chord_lyric_lines(pairs: list[dict[str, str]]) -> tuple[str, str]:
    chord_parts: list[str] = []
    lyric_parts: list[str] = []

    for pair in pairs:
        chord = (pair.get("chord") or "N.C.").strip() or "N.C."
        lyric = (pair.get("lyrics") or "").strip()

        width = max(len(chord), len(lyric), 2)
        chord_parts.append(chord.ljust(width))
        lyric_parts.append(lyric.ljust(width))

    chord_line = "  ".join(chord_parts).rstrip()
    lyric_line = "  ".join(lyric_parts).rstrip()
    return chord_line, lyric_line


def build_chordpro(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        pairs = block.get("pairs", []) or []
        if not pairs:
            continue
        line = "".join(
            f"[{(pair.get('chord') or 'N.C.').strip() or 'N.C.'}]{(pair.get('lyrics') or '').strip()}"
            for pair in pairs
        ).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def infer_chords_from_instrumental(
    instrumental_path: Path,
    output_dir: Path,
    debug_lines: list[str],
    progress_cb=None,
) -> dict[str, Any]:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    midi_output_path = output_dir / "instrumental_basic_pitch.mid"
    chord_json_path = output_dir / "chord_timeline.json"
    beat_grid_path = output_dir / "beat_grid.json"

    debug_lines.append("basic_pitch_model=" + str(ICASSP_2022_MODEL_PATH))
    debug_lines.append(f"basic_pitch_input={instrumental_path}")
    report_progress(progress_cb, "chords", "Extracting MIDI note events with Basic Pitch.", 70)

    model_output, midi_data, note_events = predict(
        str(instrumental_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        minimum_note_length=MIN_NOTE_DURATION_SECONDS * 1000.0,
    )

    _ = model_output
    midi_data.write(str(midi_output_path))
    debug_lines.append(f"basic_pitch_note_event_count={len(note_events)}")
    debug_lines.append(f"basic_pitch_midi_path={midi_output_path}")

    note_data = normalize_note_events(note_events)
    audio_duration = get_audio_duration_seconds(instrumental_path, note_data)
    report_progress(progress_cb, "beats", "Tracking beats and downbeats with madmom.", 77)
    beat_grid = run_madmom_beat_tracking(instrumental_path, audio_duration, debug_lines)
    beat_grid_path.write_text(json.dumps(beat_grid, indent=2), encoding="utf-8")
    debug_lines.append(f"beat_grid_path={beat_grid_path}")

    report_progress(progress_cb, "timeline", "Building adaptive chord timeline.", 84)
    timeline = build_chord_timeline_from_segments(
        note_data=note_data,
        beat_grid=beat_grid,
        window_seconds=CHORD_WINDOW_SECONDS,
        audio_duration=audio_duration,
    )

    if STAGE6_CHROMA_ENABLED and timeline:
        try:
            report_progress(progress_cb, "stage6", "Applying Stage 6 chroma fusion.", 88)
            timeline = apply_stage6_chroma_fusion(
                instrumental_path=instrumental_path,
                timeline=timeline,
                debug_lines=debug_lines,
            )
        except Exception as chroma_exc:
            debug_lines.append("stage6_chroma_error=" + str(chroma_exc))
            debug_lines.append(traceback.format_exc())

    chord_json_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    report_progress(progress_cb, "timeline", f"Chord timeline ready with {len(timeline)} segments.", 90)
    debug_lines.append(f"chord_timeline_entries={len(timeline)}")
    debug_lines.append(f"chord_json_path={chord_json_path}")

    return {
        "success": True,
        "error": None,
        "midi_url": f"/media/{output_dir.name}/instrumental_basic_pitch.mid",
        "chord_json_url": f"/media/{output_dir.name}/chord_timeline.json",
        "beat_grid_url": f"/media/{output_dir.name}/beat_grid.json",
        "timeline": timeline,
        "meta": {
            "basic_pitch_model": "ICASSP_2022_MODEL_PATH",
            "note_event_count": len(note_data),
            "chord_window_seconds": CHORD_WINDOW_SECONDS,
            "tempo_bpm": beat_grid.get("tempo_bpm"),
            "meter": beat_grid.get("meter"),
            "segmentation_mode": beat_grid.get("segmentation_mode"),
            "beat_count": len(beat_grid.get("beats", [])),
            "downbeat_count": len(beat_grid.get("downbeats", [])),
            "segment_count": len(beat_grid.get("segments", [])),
            "adaptive_split_count": sum(1 for item in timeline if item.get("subsegment_index", 1) > 1),
            "stage6_chroma_enabled": STAGE6_CHROMA_ENABLED,
            "stage6_fused_count": sum(1 for item in timeline if item.get("source") == "fused"),
            "stage6_chroma_count": sum(1 for item in timeline if item.get("source") == "chroma"),
            "stage6_midi_count": sum(1 for item in timeline if item.get("source") == "midi"),
        },
    }


def normalize_note_events(note_events: list[Any]) -> list[tuple[float, float, int, float]]:
    normalized_notes: list[tuple[float, float, int, float]] = []

    for event in note_events:
        if len(event) < 4:
            continue
        start_time = max(0.0, float(event[0]))
        end_time = max(start_time, float(event[1]))
        midi_note = int(event[2])
        amplitude = float(event[3])
        if end_time - start_time < MIN_NOTE_DURATION_SECONDS:
            continue
        normalized_notes.append((start_time, end_time, midi_note, amplitude))

    return normalized_notes


def get_audio_duration_seconds(audio_path: Path, note_data: list[tuple[float, float, int, float]]) -> float:
    try:
        info = sf.info(str(audio_path))
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    if note_data:
        return max(note[1] for note in note_data)
    return 0.0


def run_madmom_beat_tracking(audio_path: Path, audio_duration: float, debug_lines: list[str]) -> dict[str, Any]:
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    beat_activations = RNNBeatProcessor()(str(audio_path))
    beats = DBNBeatTrackingProcessor(fps=100)(beat_activations)
    beat_times = sorted({round(float(item), 4) for item in np.asarray(beats).reshape(-1) if float(item) >= 0.0})

    downbeat_activations = RNNDownBeatProcessor()(str(audio_path))
    downbeat_data = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(downbeat_activations)

    downbeats: list[dict[str, Any]] = []
    beat_positions: list[int] = []
    for row in np.asarray(downbeat_data):
        row = np.asarray(row).reshape(-1)
        if row.size < 2:
            continue
        time_value = round(float(row[0]), 4)
        beat_number = int(round(float(row[1])))
        if time_value < 0.0:
            continue
        beat_positions.append(beat_number)
        if beat_number == 1:
            downbeats.append({"time": time_value, "beat_in_bar": beat_number})

    meter = infer_meter_from_beat_positions(beat_positions)
    tempo_bpm = estimate_tempo_bpm(beat_times)
    segments, mode = build_segments_from_beats_and_downbeats(beat_times, downbeats, audio_duration)

    debug_lines.append(f"madmom_beats={len(beat_times)}")
    debug_lines.append(f"madmom_downbeats={len(downbeats)}")
    debug_lines.append(f"madmom_meter={meter}")
    debug_lines.append(f"madmom_tempo_bpm={tempo_bpm}")
    debug_lines.append(f"segmentation_mode={mode}")

    return {
        "tempo_bpm": tempo_bpm,
        "meter": meter,
        "beats": beat_times,
        "downbeats": downbeats,
        "segments": segments,
        "segmentation_mode": mode,
    }


def infer_meter_from_beat_positions(beat_positions: list[int]) -> str | None:
    filtered = [pos for pos in beat_positions if pos >= 1]
    if not filtered:
        return None
    try:
        meter = statistics.mode(filtered)
    except statistics.StatisticsError:
        meter = max(set(filtered), key=filtered.count)
    return f"{meter}/4" if meter >= 2 else None


def estimate_tempo_bpm(beat_times: list[float]) -> float | None:
    if len(beat_times) < 2:
        return None
    intervals = [b - a for a, b in zip(beat_times, beat_times[1:]) if b - a > 1e-4]
    if not intervals:
        return None
    median_interval = statistics.median(intervals)
    if median_interval <= 0:
        return None
    return round(60.0 / median_interval, 2)


def build_segments_from_beats_and_downbeats(
    beat_times: list[float],
    downbeats: list[dict[str, Any]],
    audio_duration: float,
) -> tuple[list[dict[str, Any]], str]:
    bar_boundaries = [float(item["time"]) for item in downbeats if item.get("beat_in_bar") == 1]
    if len(bar_boundaries) >= 2:
        segments = boundaries_to_segments(bar_boundaries, audio_duration, kind="bar")
        if segments:
            return segments, "bar"

    if len(beat_times) >= 2:
        segments = boundaries_to_segments(beat_times, audio_duration, kind="beat")
        if segments:
            return segments, "beat"

    fallback_end = max(audio_duration, CHORD_WINDOW_SECONDS)
    return [{"start": 0.0, "end": round(fallback_end, 4), "kind": "window", "index": 1}], "window"


def boundaries_to_segments(boundaries: list[float], audio_duration: float, kind: str) -> list[dict[str, Any]]:
    cleaned = sorted({round(float(value), 4) for value in boundaries if float(value) >= 0.0})
    if len(cleaned) < 2:
        return []

    segments: list[dict[str, Any]] = []
    for index, (start_time, end_time) in enumerate(zip(cleaned, cleaned[1:]), start=1):
        if end_time - start_time < MIN_SEGMENT_SECONDS:
            continue
        span = end_time - start_time
        if span > MAX_SEGMENT_SECONDS and kind == "beat":
            continue
        segments.append({
            "start": round(start_time, 4),
            "end": round(end_time, 4),
            "kind": kind,
            "index": index,
        })

    if cleaned:
        last_boundary = cleaned[-1]
        if audio_duration > last_boundary + MIN_SEGMENT_SECONDS and audio_duration - last_boundary <= MAX_SEGMENT_SECONDS * 2:
            segments.append({
                "start": round(last_boundary, 4),
                "end": round(audio_duration, 4),
                "kind": kind,
                "index": len(segments) + 1,
            })

    return [seg for seg in segments if seg["end"] > seg["start"]]


def build_chord_timeline_from_segments(
    note_data: list[tuple[float, float, int, float]],
    beat_grid: dict[str, Any],
    window_seconds: float,
    audio_duration: float,
) -> list[dict[str, Any]]:
    from music21 import chord as m21chord
    from music21 import harmony

    if not note_data:
        return []

    segments = beat_grid.get("segments") or []
    beats = [float(value) for value in (beat_grid.get("beats") or [])]
    if not segments:
        return build_chord_timeline_from_note_events(note_data, window_seconds)

    timeline: list[dict[str, Any]] = []
    for segment in segments:
        start_time = float(segment["start"])
        end_time = float(segment["end"])
        subsegments = split_segment_on_harmonic_change(note_data, start_time, end_time, beats)
        for sub_index, subsegment in enumerate(subsegments, start=1):
            detected = detect_chord_for_span(note_data, subsegment["start"], subsegment["end"], m21chord, harmony)
            timeline.append({
                "start": round(subsegment["start"], 2),
                "end": round(min(subsegment["end"], audio_duration), 2),
                "segment_kind": segment.get("kind"),
                "segment_index": segment.get("index"),
                "subsegment_index": sub_index,
                "split_level": subsegment.get("split_level", "full"),
                "split_reason": subsegment.get("split_reason", "base"),
                **detected,
            })

    return merge_adjacent_timeline_entries(timeline)




def build_pitch_class_profile(
    note_data: list[tuple[float, float, int, float]],
    current_start: float,
    current_end: float,
) -> dict[int, float]:
    profile: dict[int, float] = defaultdict(float)
    for start_time, end_time, midi_note, amplitude in note_data:
        overlap = max(0.0, min(current_end, end_time) - max(current_start, start_time))
        if overlap <= 0:
            continue
        profile[midi_note % 12] += overlap * max(0.15, amplitude)
    return profile


def cosine_distance_from_profiles(left_profile: dict[int, float], right_profile: dict[int, float]) -> float:
    keys = sorted(set(left_profile) | set(right_profile))
    if not keys:
        return 0.0
    left_vec = np.asarray([left_profile.get(key, 0.0) for key in keys], dtype=float)
    right_vec = np.asarray([right_profile.get(key, 0.0) for key in keys], dtype=float)
    left_norm = float(np.linalg.norm(left_vec))
    right_norm = float(np.linalg.norm(right_vec))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    similarity = float(np.dot(left_vec, right_vec) / (left_norm * right_norm))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def profile_support(profile: dict[int, float]) -> tuple[int, float]:
    active = sum(1 for value in profile.values() if value > 1e-6)
    total = float(sum(profile.values()))
    return active, total


def should_split_profiles(
    left_profile: dict[int, float],
    right_profile: dict[int, float],
    span: float,
    threshold: float,
) -> bool:
    left_active, left_total = profile_support(left_profile)
    right_active, right_total = profile_support(right_profile)
    if left_active < MIN_ACTIVE_NOTES_FOR_CHORD or right_active < MIN_ACTIVE_NOTES_FOR_CHORD:
        return False
    if min(left_total, right_total) < max(0.12, span * 0.08):
        return False
    return cosine_distance_from_profiles(left_profile, right_profile) >= threshold


def split_segment_on_harmonic_change(
    note_data: list[tuple[float, float, int, float]],
    start_time: float,
    end_time: float,
    beat_times: list[float],
) -> list[dict[str, Any]]:
    span = end_time - start_time
    if span < MIN_SEGMENT_SECONDS * 1.5:
        return [{"start": start_time, "end": end_time, "split_level": "full", "split_reason": "short"}]

    internal_beats = [beat for beat in beat_times if start_time + 0.08 < beat < end_time - 0.08]
    if not internal_beats:
        return [{"start": start_time, "end": end_time, "split_level": "full", "split_reason": "no_internal_beats"}]

    ordered_candidates: list[tuple[str, float]] = []
    midpoint = start_time + span / 2.0
    midpoint_candidate = min(internal_beats, key=lambda beat: abs(beat - midpoint))
    ordered_candidates.append(("half", midpoint_candidate))
    for beat in internal_beats:
        if abs(beat - midpoint_candidate) > 1e-4:
            ordered_candidates.append(("beat", beat))

    def recursive_split(local_start: float, local_end: float, local_candidates: list[tuple[str, float]], depth: int = 0) -> list[dict[str, Any]]:
        local_span = local_end - local_start
        if local_span < MIN_SEGMENT_SECONDS * 1.5 or depth >= 2:
            return [{"start": local_start, "end": local_end, "split_level": "full", "split_reason": "stable_or_limit"}]

        filtered = [(level, point) for level, point in local_candidates if local_start + 0.08 < point < local_end - 0.08]
        for level, point in filtered:
            left_profile = build_pitch_class_profile(note_data, local_start, point)
            right_profile = build_pitch_class_profile(note_data, point, local_end)
            threshold = 0.34 if level == "half" else 0.42
            if should_split_profiles(left_profile, right_profile, local_span, threshold):
                left_candidates = [(lvl, cand) for lvl, cand in filtered if cand < point]
                right_candidates = [(lvl, cand) for lvl, cand in filtered if cand > point]
                left_parts = recursive_split(local_start, point, left_candidates, depth + 1)
                right_parts = recursive_split(point, local_end, right_candidates, depth + 1)
                if left_parts:
                    left_parts[-1]["split_level"] = level
                    left_parts[-1]["split_reason"] = f"harmonic_change_before_{level}"
                if right_parts:
                    right_parts[0]["split_level"] = level
                    right_parts[0]["split_reason"] = f"harmonic_change_after_{level}"
                return left_parts + right_parts

        return [{"start": local_start, "end": local_end, "split_level": "full", "split_reason": "stable"}]

    parts = recursive_split(start_time, end_time, ordered_candidates)
    return [part for part in parts if part["end"] - part["start"] >= MIN_SEGMENT_SECONDS * 0.75]


def simplify_base_chord_symbol(symbol: str) -> str:
    cleaned = (symbol or "").strip()
    if not cleaned:
        return "N.C."

    cleaned = cleaned.split(",", 1)[0].strip()
    cleaned = re.split(r"(?i)add", cleaned, maxsplit=1)[0].strip()
    cleaned = cleaned.rstrip("-+/ ") if cleaned.endswith(("+", "/")) else cleaned.rstrip()

    match = re.match(
        r"^([A-G](?:#|b|-)?(?:maj13|maj11|maj9|maj7|maj6|min13|min11|min9|min7|min6|dim7|hdim7|dim|aug7|aug|sus4|sus2|sus|m13|m11|m9|m7|m6|m|7|9|11|13|6)?)",
        cleaned,
    )
    if match:
        return match.group(1)

    fallback = re.match(r"^([A-G](?:#|b|-)?[A-Za-z0-9+#-]*)", cleaned)
    if fallback:
        return fallback.group(1)

    return cleaned or "N.C."


def simplify_chord_label(label: str) -> str:
    cleaned = (label or "").strip()
    if not cleaned or cleaned.upper() == "N.C.":
        return "N.C."

    primary = cleaned.split(",", 1)[0].strip()
    if "/" in primary:
        base, slash_part = primary.split("/", 1)
        base = simplify_base_chord_symbol(base)
        bass_match = re.match(r"^([A-G](?:#|b|-)?)", slash_part.strip())
        if bass_match:
            return f"{base}/{bass_match.group(1)}"
        return base

    return simplify_base_chord_symbol(primary)


def normalize_display_chord_label(label: str) -> str:
    normalized = (label or "").strip()
    if not normalized:
        return "N.C."
    normalized = re.sub(r"([A-G])-", r"\1b", normalized)
    return normalized



def choose_fused_chord_label(segment: dict[str, Any]) -> tuple[str, str, float]:
    midi_label = (segment.get("label") or "N.C.").strip() or "N.C."
    chroma_label = normalize_display_chord_label(simplify_chord_label(segment.get("chroma_label") or "N.C."))

    midi_conf = float(segment.get("midi_confidence") or 0.0)
    chroma_conf = float(segment.get("chroma_confidence") or 0.0)

    if chroma_label == "N.C." and midi_label != "N.C.":
        return midi_label, "midi", midi_conf

    if midi_label == chroma_label:
        return midi_label, "fused", max(midi_conf, chroma_conf)

    if midi_label == "N.C." and chroma_conf >= RUNTIME_STAGE6_CHROMA_CONFIDENCE_THRESHOLD:
        return chroma_label, "chroma", chroma_conf

    if chroma_conf >= RUNTIME_STAGE6_CHROMA_OVERRIDE_THRESHOLD:
        return chroma_label, "chroma", chroma_conf

    return midi_label, "midi", max(midi_conf, chroma_conf)


def apply_stage6_chroma_fusion(
    instrumental_path: Path,
    timeline: list[dict[str, Any]],
    debug_lines: list[str],
) -> list[dict[str, Any]]:
    chroma_results = infer_chroma_for_timeline(instrumental_path, timeline)
    debug_lines.append(f"stage6_chroma_segments={len(chroma_results)}")

    fused_timeline: list[dict[str, Any]] = []
    for segment, chroma_info in zip(timeline, chroma_results):
        merged = dict(segment)
        merged["midi_label"] = merged.get("label", "N.C.")
        merged["midi_raw_label"] = merged.get("raw_label", merged.get("label", "N.C."))
        merged["midi_confidence"] = float(merged.get("midi_confidence") or merged.get("confidence") or 0.0)
        merged["chroma_label"] = chroma_info.get("label", "N.C.")
        merged["chroma_raw_label"] = chroma_info.get("raw_label", chroma_info.get("label", "N.C."))
        merged["chroma_confidence"] = float(chroma_info.get("confidence") or 0.0)
        merged["chroma_scores"] = chroma_info.get("scores", {})

        final_label, source, confidence = choose_fused_chord_label(merged)
        merged["label"] = final_label
        merged["source"] = source
        merged["confidence"] = round(float(confidence), 4)
        if source == "chroma":
            merged["raw_label"] = merged.get("chroma_raw_label", merged.get("raw_label", final_label))
        else:
            merged["raw_label"] = merged.get("midi_raw_label", merged.get("raw_label", final_label))
        fused_timeline.append(merged)

    return merge_adjacent_timeline_entries(fused_timeline)


def merge_adjacent_timeline_entries(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in timeline:
        if merged and merged[-1]["label"] == item["label"] and abs(merged[-1]["end"] - item["start"]) <= 0.06:
            merged[-1]["end"] = item["end"]
            merged[-1]["split_reason"] = "merged_adjacent"
            continue
        merged.append(dict(item))
    return [entry for entry in merged if entry["end"] > entry["start"]]

def detect_chord_for_span(
    note_data: list[tuple[float, float, int, float]],
    current_start: float,
    current_end: float,
    m21chord: Any,
    harmony: Any,
) -> dict[str, Any]:
    pitch_weights: dict[int, float] = defaultdict(float)

    for start_time, end_time, midi_note, amplitude in note_data:
        overlap = max(0.0, min(current_end, end_time) - max(current_start, start_time))
        if overlap <= 0:
            continue
        weighted_overlap = overlap * max(0.15, amplitude)
        pitch_weights[midi_note] += weighted_overlap

    if len(pitch_weights) < MIN_ACTIVE_NOTES_FOR_CHORD:
        return {
            "label": "N.C.",
            "name": "No chord",
            "pitch_classes": [],
            "pitches": [],
            "midi_confidence": 0.0,
            "source": "midi",
            "confidence": 0.0,
        }

    ranked_pitches = sorted(pitch_weights.items(), key=lambda item: (-item[1], item[0]))
    kept_pitches = [pitch for pitch, _weight in ranked_pitches[:6]]
    return describe_chord(kept_pitches, m21chord, harmony)


def build_chord_timeline_from_note_events(note_events: list[Any], window_seconds: float) -> list[dict[str, Any]]:
    from music21 import chord as m21chord
    from music21 import harmony

    if not note_events:
        return []

    normalized_notes: list[tuple[float, float, int, float]] = []
    last_end = 0.0

    for event in note_events:
        if len(event) < 4:
            continue
        start_time = max(0.0, float(event[0]))
        end_time = max(start_time, float(event[1]))
        midi_note = int(event[2])
        amplitude = float(event[3])
        if end_time - start_time < MIN_NOTE_DURATION_SECONDS:
            continue
        normalized_notes.append((start_time, end_time, midi_note, amplitude))
        last_end = max(last_end, end_time)

    if not normalized_notes:
        return []

    windows: list[dict[str, Any]] = []
    current_start = 0.0
    while current_start < last_end + 1e-9:
        current_end = min(last_end, current_start + window_seconds)
        pitch_weights: dict[int, float] = defaultdict(float)
        pitch_velocities: dict[int, float] = defaultdict(float)

        for start_time, end_time, midi_note, amplitude in normalized_notes:
            overlap = max(0.0, min(current_end, end_time) - max(current_start, start_time))
            if overlap <= 0:
                continue
            pitch_weights[midi_note] += overlap
            pitch_velocities[midi_note] = max(pitch_velocities[midi_note], amplitude)

        if len(pitch_weights) >= MIN_ACTIVE_NOTES_FOR_CHORD:
            ranked_pitches = sorted(
                pitch_weights.items(),
                key=lambda item: (-item[1], item[0]),
            )
            kept_pitches = [pitch for pitch, _weight in ranked_pitches[:6]]
            detected = describe_chord(kept_pitches, m21chord, harmony)
        else:
            detected = {
                "label": "N.C.",
                "name": "No chord",
                "pitch_classes": [],
                "pitches": [],
                "midi_confidence": 0.0,
                "source": "midi",
                "confidence": 0.0,
            }

        windows.append(
            {
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                **detected,
            }
        )
        current_start += window_seconds

    merged: list[dict[str, Any]] = []
    for item in windows:
        if merged and merged[-1]["label"] == item["label"]:
            merged[-1]["end"] = item["end"]
            continue
        merged.append(item)

    return [entry for entry in merged if entry["end"] > entry["start"]]


def describe_chord(pitches: list[int], m21chord: Any, harmony: Any) -> dict[str, Any]:
    chord_obj = m21chord.Chord(pitches)
    unique_pitch_classes = sorted({pitch % 12 for pitch in pitches})

    label = None
    try:
        chord_symbol = harmony.chordSymbolFromChord(chord_obj)
        label = getattr(chord_symbol, "figure", None) or None
    except Exception:
        label = None

    if not label:
        try:
            label = harmony.chordSymbolFigureFromChord(chord_obj)
        except Exception:
            label = None

    if not label:
        root = chord_obj.root()
        quality = getattr(chord_obj, "quality", None) or "other"
        quality_map = {
            "major": "",
            "minor": "m",
            "augmented": "aug",
            "diminished": "dim",
            "other": "",
        }
        if root is not None:
            label = f"{root.name}{quality_map.get(quality, '')}".strip()
        else:
            label = chord_obj.commonName or "Unknown chord"

    long_name = getattr(chord_obj, "pitchedCommonName", None) or getattr(chord_obj, "commonName", None) or label

    simplified_label = normalize_display_chord_label(simplify_chord_label(label))

    midi_confidence = round(min(1.0, 0.42 + 0.08 * len(unique_pitch_classes) + 0.04 * min(len(pitches), 6)), 4)

    return {
        "label": simplified_label,
        "raw_label": label,
        "name": long_name,
        "pitch_classes": unique_pitch_classes,
        "pitches": sorted(pitches),
        "midi_confidence": midi_confidence,
        "source": "midi",
        "confidence": midi_confidence,
    }

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DISPLAY_NAMES = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "other": "Other",
    "piano": "Piano",
    "guitar": "Guitar",
}


def _safe_stem_id(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_") or "stem"


def _stem_sort_key(stem_id: str) -> tuple[int, str]:
    order = {
        "vocals": 0,
        "drums": 1,
        "bass": 2,
        "piano": 3,
        "guitar": 4,
        "other": 5,
    }
    return (order.get(stem_id, 999), stem_id)


def find_demucs_stem_files(search_root: Path) -> list[Path]:
    candidates: list[Path] = []
    if not search_root.exists():
        return candidates

    for file_path in search_root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
            candidates.append(file_path)

    return sorted(candidates)


def collect_stems_into_job_dir(job_dir: Path, demucs_output_root: Path) -> list[dict[str, Any]]:
    """
    Copy every Demucs-produced audio stem into `<job_dir>/stems/` and build a manifest.

    This function is intentionally tolerant of different Demucs folder layouts.
    """
    job_dir = Path(job_dir)
    demucs_output_root = Path(demucs_output_root)
    stems_dir = job_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    detected_files = find_demucs_stem_files(demucs_output_root)
    stems: list[dict[str, Any]] = []
    used_targets: set[str] = set()

    for source_path in detected_files:
        stem_id = _safe_stem_id(source_path.stem)
        target_name = f"{stem_id}.wav"
        if target_name in used_targets:
            continue
        used_targets.add(target_name)

        target_path = stems_dir / target_name
        shutil.copy2(source_path, target_path)

        stems.append(
            {
                "id": stem_id,
                "label": DISPLAY_NAMES.get(stem_id, source_path.stem.replace("_", " ").title()),
                "filename": target_name,
                "path": str(target_path),
            }
        )

    stems.sort(key=lambda item: _stem_sort_key(item["id"]))
    return stems


def derive_instrumental_mix(job_dir: Path, stems: list[dict[str, Any]]) -> Path | None:
    """
    Build `<job_dir>/instrumental.wav` by mixing every non-vocal stem with ffmpeg amix.
    """
    job_dir = Path(job_dir)
    non_vocal_paths = [Path(item["path"]) for item in stems if item["id"] != "vocals"]
    output_path = job_dir / "instrumental.wav"

    if not non_vocal_paths:
        return None

    if len(non_vocal_paths) == 1:
        shutil.copy2(non_vocal_paths[0], output_path)
        return output_path

    cmd = ["ffmpeg", "-y"]
    for file_path in non_vocal_paths:
        cmd.extend(["-i", str(file_path)])

    amix = f"amix=inputs={len(non_vocal_paths)}:normalize=0:dropout_transition=0"
    cmd.extend(["-filter_complex", amix, "-c:a", "pcm_s16le", str(output_path)])

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def ensure_vocals_shortcut(job_dir: Path, stems: list[dict[str, Any]]) -> Path | None:
    """
    Keep compatibility with the rest of the pipeline by copying `stems/vocals.wav` to `<job_dir>/vocals.wav`.
    """
    job_dir = Path(job_dir)
    for item in stems:
        if item["id"] == "vocals":
            source = Path(item["path"])
            target = job_dir / "vocals.wav"
            shutil.copy2(source, target)
            return target
    return None


def write_stem_manifest(job_dir: Path, stems: list[dict[str, Any]], job_id: str) -> dict[str, Any]:
    job_dir = Path(job_dir)
    manifest_path = job_dir / "stems" / "stem_manifest.json"

    manifest_stems = [
        {
            "id": item["id"],
            "label": item["label"],
            "url": f"/media/{job_id}/stems/{item['filename']}",
        }
        for item in stems
    ]

    manifest: dict[str, Any] = {
        "job_id": job_id,
        "stems": manifest_stems,
        "instrumental_url": f"/media/{job_id}/instrumental.wav",
        "vocals_url": f"/media/{job_id}/vocals.wav",
        "manifest_url": f"/media/{job_id}/stems/stem_manifest.json",
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_multitrack_stems(job_dir: Path, demucs_output_root: Path, job_id: str) -> dict[str, Any]:
    """
    Full helper for Stage 1:
    1. collect all stems from Demucs output
    2. create `vocals.wav`
    3. create `instrumental.wav`
    4. write `stem_manifest.json`
    """
    stems = collect_stems_into_job_dir(job_dir=job_dir, demucs_output_root=demucs_output_root)
    ensure_vocals_shortcut(job_dir=job_dir, stems=stems)
    derive_instrumental_mix(job_dir=job_dir, stems=stems)
    return write_stem_manifest(job_dir=job_dir, stems=stems, job_id=job_id)


def load_stem_manifest(job_dir: Path) -> dict[str, Any] | None:
    manifest_path = Path(job_dir) / "stems" / "stem_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))

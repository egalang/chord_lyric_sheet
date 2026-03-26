
from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_TEMPLATES: dict[str, np.ndarray] = {
    "": np.asarray([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float),
    "m": np.asarray([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float),
    "7": np.asarray([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=float),
    "maj7": np.asarray([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=float),
    "m7": np.asarray([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=float),
    "dim": np.asarray([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float),
    "sus2": np.asarray([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float),
    "sus4": np.asarray([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=float),
}


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return vec
    return vec / norm


def _score_vector(chroma_vec: np.ndarray) -> tuple[str, float, float, dict[str, float]]:
    if chroma_vec.shape[0] != 12:
        raise ValueError("Expected 12-bin chroma vector")

    chroma_vec = _normalize(np.asarray(chroma_vec, dtype=float).reshape(12))
    if float(np.sum(np.abs(chroma_vec))) <= 1e-6:
        return "N.C.", 0.0, 0.0, {}

    scores: list[tuple[str, float]] = []
    for root in range(12):
        for suffix, template in CHORD_TEMPLATES.items():
            rotated = _normalize(np.roll(template, root))
            score = float(np.dot(chroma_vec, rotated))
            label = NOTE_NAMES[root] + suffix
            scores.append((label, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    best_label, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0
    confidence = max(0.0, min(1.0, 0.65 * best_score + 0.35 * max(0.0, best_score - second_score)))
    top_scores = {label: round(score, 4) for label, score in scores[:5]}
    return best_label, round(confidence, 4), round(best_score, 4), top_scores


def infer_chroma_for_timeline(audio_path: Path, timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not timeline:
        return []

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    results: list[dict[str, Any]] = []
    for segment in timeline:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end <= start:
            results.append({"label": "N.C.", "raw_label": "N.C.", "confidence": 0.0, "scores": {}})
            continue

        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            nearest_idx = int(np.argmin(np.abs(frame_times - start))) if frame_times.size else 0
            vec = chroma[:, nearest_idx] if chroma.shape[1] else np.zeros(12, dtype=float)
        else:
            vec = np.mean(chroma[:, mask], axis=1)

        label, confidence, best_score, top_scores = _score_vector(vec)
        results.append({
            "label": label,
            "raw_label": label,
            "confidence": confidence,
            "best_score": best_score,
            "scores": top_scores,
        })

    return results

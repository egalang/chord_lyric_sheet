from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

NOTE_TO_STEP_ALTER = {
    "C": ("C", 0),
    "C#": ("C", 1),
    "Db": ("D", -1),
    "D": ("D", 0),
    "D#": ("D", 1),
    "Eb": ("E", -1),
    "E": ("E", 0),
    "Fb": ("F", -1),
    "E#": ("F", 1),
    "F": ("F", 0),
    "F#": ("F", 1),
    "Gb": ("G", -1),
    "G": ("G", 0),
    "G#": ("G", 1),
    "Ab": ("A", -1),
    "A": ("A", 0),
    "A#": ("A", 1),
    "Bb": ("B", -1),
    "B": ("B", 0),
    "Cb": ("C", -1),
    "B#": ("C", 1),
}

KINDS = [
    ("maj7", "major-seventh"),
    ("m7", "minor-seventh"),
    ("min7", "minor-seventh"),
    ("dim", "diminished"),
    ("aug", "augmented"),
    ("sus2", "suspended-second"),
    ("sus4", "suspended-fourth"),
    ("sus", "suspended-fourth"),
    ("m", "minor"),
    ("7", "dominant"),
]


def normalize_chord_label(label: str | None) -> str:
    value = str(label or "N.C.").strip()
    if not value:
        return "N.C."
    return value.replace("-", "b")


def parse_time_signature(value: str | None) -> tuple[int, int, str]:
    raw = str(value or "4/4").strip() or "4/4"
    try:
        beats_str, beat_type_str = raw.split("/", 1)
        beats = max(1, int(beats_str))
        beat_type = int(beat_type_str)
        if beat_type not in {1, 2, 4, 8, 16, 32}:
            beat_type = 4
        return beats, beat_type, f"{beats}/{beat_type}"
    except Exception:
        return 4, 4, "4/4"


def parse_chord_symbol(label: str | None) -> dict[str, Any] | None:
    value = normalize_chord_label(label)
    if value in {"N.C.", "NC", "None"}:
        return None

    root = value[0].upper()
    remainder = value[1:]
    accidental = ""
    if remainder[:1] in {"#", "b"}:
        accidental = remainder[:1]
        remainder = remainder[1:]
    root_name = f"{root}{accidental}"
    step, alter = NOTE_TO_STEP_ALTER.get(root_name, (root, 0))

    kind_value = "major"
    for token, mapped in KINDS:
        if remainder.startswith(token):
            kind_value = mapped
            break

    bass = None
    if "/" in remainder:
        _, bass_token = remainder.split("/", 1)
        bass_token = bass_token.strip()
        if bass_token:
            bass_step, bass_alter = NOTE_TO_STEP_ALTER.get(bass_token, (bass_token[:1].upper(), 0))
            bass = {"step": bass_step, "alter": bass_alter}

    return {
        "display": value,
        "step": step,
        "alter": alter,
        "kind": kind_value,
        "bass": bass,
    }


def estimate_end_time(segments: list[dict[str, Any]], chords: list[dict[str, Any]], beat_grid: dict[str, Any]) -> float:
    candidates: list[float] = []
    for seg in segments or []:
        candidates.append(float(seg.get("end", 0.0)))
    for item in chords or []:
        candidates.append(float(item.get("end", 0.0)))
    for beat in beat_grid.get("beats", []) or []:
        candidates.append(float(beat))
    return max(candidates) if candidates else 0.0


def build_anchor_times(beat_grid: dict[str, Any], tempo_bpm: float, beats_per_measure: int, max_time: float) -> list[float]:
    beat_times = [float(x) for x in (beat_grid.get("beats") or []) if float(x) >= 0.0]
    if len(beat_times) >= 2:
        return sorted(beat_times)

    beat_duration = 60.0 / max(tempo_bpm, 1.0)
    total_beats = max(beats_per_measure, int(math.ceil(max_time / beat_duration)) + 1)
    return [round(i * beat_duration, 4) for i in range(total_beats)]


def choose_chord_at_time(chords: list[dict[str, Any]], t: float) -> str | None:
    for item in chords:
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if start <= t < end:
            return normalize_chord_label(item.get("label") or item.get("name"))
    previous = [item for item in chords if float(item.get("start", 0.0)) <= t]
    if previous:
        return normalize_chord_label(previous[-1].get("label") or previous[-1].get("name"))
    return None


def attach_lyrics_to_anchors(segments: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> None:
    if not anchors:
        return
    for seg in segments or []:
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue
        words = text.split()
        if not words:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        candidate_indexes = [i for i, a in enumerate(anchors) if start <= float(a["time"]) <= max(end, start)]
        if not candidate_indexes:
            nearest_index = min(range(len(anchors)), key=lambda i: abs(float(anchors[i]["time"]) - start))
            candidate_indexes = [nearest_index]
        if len(candidate_indexes) == 1:
            idx = candidate_indexes[0]
            anchors[idx]["lyric"] = ((anchors[idx].get("lyric") or "") + (" " if anchors[idx].get("lyric") else "") + " ".join(words)).strip()
            continue
        for word, idx in zip(words, candidate_indexes + [candidate_indexes[-1]] * max(0, len(words) - len(candidate_indexes))):
            current = anchors[idx].get("lyric") or ""
            anchors[idx]["lyric"] = (current + (" " if current else "") + word).strip()


def build_measure_plan(
    transcription_segments: list[dict[str, Any]],
    chord_timeline: list[dict[str, Any]],
    beat_grid: dict[str, Any],
    time_signature: str,
    tempo_bpm: float,
) -> list[dict[str, Any]]:
    beats_per_measure, beat_type, normalized_signature = parse_time_signature(time_signature)
    max_time = estimate_end_time(transcription_segments, chord_timeline, beat_grid)
    beat_times = build_anchor_times(beat_grid, tempo_bpm, beats_per_measure, max_time)

    anchors: list[dict[str, Any]] = []
    for beat_index, t in enumerate(beat_times):
        chord_label = choose_chord_at_time(chord_timeline, t)
        anchors.append({
            "time": t,
            "global_beat_index": beat_index,
            "chord": chord_label,
            "lyric": "",
        })

    attach_lyrics_to_anchors(transcription_segments, anchors)

    measures: list[dict[str, Any]] = []
    for offset in range(0, len(anchors), beats_per_measure):
        chunk = anchors[offset: offset + beats_per_measure]
        if not chunk:
            continue
        measures.append({
            "number": len(measures) + 1,
            "beats": chunk,
        })

    return measures


def add_harmony(measure_el: ET.Element, chord_label: str | None) -> None:
    parsed = parse_chord_symbol(chord_label)
    if not parsed:
        return
    harmony = ET.SubElement(measure_el, "harmony")
    root = ET.SubElement(harmony, "root")
    ET.SubElement(root, "root-step").text = parsed["step"]
    if parsed["alter"]:
        ET.SubElement(root, "root-alter").text = str(parsed["alter"])
    kind = ET.SubElement(harmony, "kind")
    kind.text = parsed["kind"]
    kind.set("text", parsed["display"])
    if parsed.get("bass"):
        bass = ET.SubElement(harmony, "bass")
        ET.SubElement(bass, "bass-step").text = parsed["bass"]["step"]
        if parsed["bass"]["alter"]:
            ET.SubElement(bass, "bass-alter").text = str(parsed["bass"]["alter"])


def add_placeholder_note(measure_el: ET.Element, beat_type: int, lyric_text: str | None) -> None:
    note = ET.SubElement(measure_el, "note")
    ET.SubElement(note, "rest")
    ET.SubElement(note, "duration").text = "1"
    ET.SubElement(note, "voice").text = "1"
    note_type = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th"}.get(beat_type, "quarter")
    ET.SubElement(note, "type").text = note_type
    lyric_text = str(lyric_text or "").strip()
    if lyric_text:
        lyric = ET.SubElement(note, "lyric")
        ET.SubElement(lyric, "text").text = lyric_text


def write_musicxml(
    output_path: Path,
    title: str,
    measures: list[dict[str, Any]],
    time_signature: str,
    tempo_bpm: float,
) -> Path:
    beats_per_measure, beat_type, normalized_signature = parse_time_signature(time_signature)

    root = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title or "Lead Sheet"

    identification = ET.SubElement(root, "identification")
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "Chord Lyric Sheet"

    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Lead Sheet"

    part = ET.SubElement(root, "part", id="P1")

    for measure_data in measures:
        measure_el = ET.SubElement(part, "measure", number=str(measure_data["number"]))
        if measure_data["number"] == 1:
            attributes = ET.SubElement(measure_el, "attributes")
            ET.SubElement(attributes, "divisions").text = "1"
            key = ET.SubElement(attributes, "key")
            ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = str(beats_per_measure)
            ET.SubElement(time, "beat-type").text = str(beat_type)
            clef = ET.SubElement(attributes, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"

            direction = ET.SubElement(measure_el, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = {1: "whole", 2: "half", 4: "quarter", 8: "eighth"}.get(beat_type, "quarter")
            ET.SubElement(metronome, "per-minute").text = str(int(round(tempo_bpm or 90)))
            sound = ET.SubElement(direction, "sound")
            sound.set("tempo", str(float(tempo_bpm or 90)))

        previous_chord = None
        for beat in measure_data["beats"]:
            chord_label = normalize_chord_label(beat.get("chord")) if beat.get("chord") else None
            if chord_label and chord_label != previous_chord:
                add_harmony(measure_el, chord_label)
                previous_chord = chord_label
            add_placeholder_note(measure_el, beat_type, beat.get("lyric"))

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


# Demucs + Whisper Stage 6 Hybrid Chord MVP

This package builds on the working Stage 5 MVP and adds a new Stage 6 harmonic engine focused on improving chord accuracy while preserving the current UI and workflow.

## Included pipeline

- Stage 1: Demucs stem separation
- Stage 2: Whisper lyric transcription with timestamps
- Stage 3: Basic Pitch MIDI extraction
- Stage 4: madmom beat and downbeat tracking
- Stage 5: adaptive intra-bar chord segmentation with music21 chord inference
- Stage 6: chroma-based chord detection with confidence scoring and hybrid fusion
- Lead Sheet MVP: chord-over-lyric output shown in the browser

## What is new in Stage 6

- adds `app/chord/chroma_engine.py`
- extracts chroma features directly from `instrumental.wav` with librosa
- scores chord templates over each Stage 5 timeline segment
- keeps the existing MIDI/music21 result as fallback
- fuses MIDI and chroma outputs conservatively
- includes `confidence`, `source`, `midi_label`, and `chroma_label` in chord timeline entries

## Fusion behavior

- if MIDI and chroma agree, the result is marked as `source: fused`
- if MIDI is `N.C.` and chroma confidence is high, chroma can take over
- if chroma confidence is very high, it can override the MIDI label
- otherwise the Stage 5 MIDI result is preserved

## Run

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
```

## Notes

- UI remains the same so you can compare Stage 5 and Stage 6 without changing workflow.
- The package keeps MP3 previews and range-aware media streaming for browser seeking.
- This is still a lead sheet MVP, not engraved staff notation yet.

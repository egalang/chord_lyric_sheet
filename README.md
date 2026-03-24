# Demucs + Whisper + Adaptive Chord Segmentation + Lead Sheet MVP

This Docker Compose package keeps the working Stage 1 and Stage 2 flow intact and adds a Stage 5 chord pass:

- Stage 1: Demucs stem separation
- Stage 2: Whisper lyric transcription with timestamped segments
- Stage 3/4 note + harmony pipeline:
  - Basic Pitch converts `instrumental.wav` to MIDI note events
  - madmom estimates beat and downbeat positions from the same instrumental audio
  - the app builds a beat-aware or bar-aware segment grid
  - music21 applies the same chord labeling logic on those beat-synchronous or bar-aware spans

## Outputs

For each upload the app returns:

- `vocals.wav`
- `instrumental.wav`
- full transcript
- timestamped lyric segments
- `instrumental_basic_pitch.mid`
- `beat_grid.json`
- `chord_timeline.json`
- `debug.txt`
- lead sheet text view that combines chords and lyrics

## Stage 5 behavior

Stage 5 keeps the Stage 4 beat/bar grid and adds adaptive intra-bar splitting:

1. build bar-aware segments from madmom downbeats
2. test likely split points inside each bar, especially half-bar and beat positions
3. split only when left/right harmonic profiles differ strongly enough
4. keep stable bars intact and merge adjacent identical chord labels afterward

This keeps the same Basic Pitch -> music21 chord logic, but avoids forcing one chord label across a full bar when there is evidence of an intra-bar change.

## Run

```bash
docker compose up --build
```

Open: `http://localhost:8000`

## Notes

- Stage 1 and Stage 2 remain isolated so they can still succeed even if Stage 5 chord inference fails.
- `madmom` is installed from the CPJKU GitHub repository in this package so it is more compatible with modern Python versions than the very old PyPI release.


## Lead sheet MVP

This update adds a browser-side lead sheet panel that displays chord-over-lyric text using the Stage 5 chord timeline plus Whisper lyric segments. It is intended as a quick MVP display, not engraved staff notation yet.

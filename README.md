# Demucs + Whisper Stage 6 Hybrid Chord MVP

This package builds on the working Stage 5 MVP and adds a Stage 6 harmonic engine focused on improving chord accuracy while preserving the current UI and workflow.

## Included pipeline

- Stage 1: Demucs stem separation
- Stage 2: Whisper lyric transcription with timestamps
- Stage 3: Basic Pitch MIDI extraction
- Stage 4: madmom beat and downbeat tracking
- Stage 5: adaptive intra-bar chord segmentation with music21 chord inference
- Stage 6: chroma-based chord detection with confidence scoring and hybrid fusion
- Lead Sheet MVP: chord-over-lyric output shown in the browser

## What is new in this package

- adds `app/chord/chroma_engine.py`
- extracts chroma features directly from `instrumental.wav` with librosa
- scores chord templates over each Stage 5 timeline segment
- keeps the existing MIDI/music21 result as fallback
- fuses MIDI and chroma outputs conservatively
- includes `confidence`, `source`, `midi_label`, and `chroma_label` in chord timeline entries
- accepts either a local audio upload or a YouTube link as the input source
- uses `yt-dlp` and `ffmpeg` to download and prepare YouTube audio before running the same pipeline

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

- The upload screen now supports either an uploaded audio file or a YouTube link.
- For YouTube processing, the container must have outbound internet access and the video must be publicly accessible.
- The package keeps MP3 previews and range-aware media streaming for browser seeking.
- This is still a lead sheet MVP, not engraved staff notation yet.


## MusicXML export

This package now includes a downloadable MusicXML export from completed jobs. In the browser, choose a time signature (default `4/4`) and click **Download MusicXML** from the Lead sheet panel. The server generates `lead_sheet.musicxml` inside the job output folder and serves it back so it can be opened in MuseScore or other notation apps.

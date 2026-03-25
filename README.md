# Demucs + Whisper Stage 5 Lead Sheet MVP

This package keeps the working Stage 5 pipeline and upgrades the web UI so the generated output is easier to review as a musician-facing MVP.

## Included pipeline

- Stage 1: Demucs stem separation
- Stage 2: Whisper lyric transcription with timestamps
- Stage 3: Basic Pitch MIDI extraction
- Stage 4: madmom beat and downbeat tracking
- Stage 5: adaptive intra-bar chord segmentation with music21 chord inference
- Lead Sheet MVP: chord-over-lyric output shown in the browser

## UI upgrades in this package

- refreshed landing section with clearer workflow guidance
- processing summary cards for language, lyric count, chord count, and tempo
- improved stem validation layout
- cleaner transcript and chord timeline cards
- jump buttons to audition lyric and chord entries in the correct audio player
- upgraded lead sheet area with:
  - plain text view
  - structured line cards
  - copy lead sheet button
  - copy chord-lyric line format button

## Run

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
```

## Notes

- The backend API shape is unchanged from the working lead sheet MVP package.
- The UI reads the existing `lead_sheet.blocks` structure for the structured lead sheet view.
- This is still a lead sheet MVP, not engraved staff notation yet.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A containerized Python audio transcription pipeline using WhisperX for end-to-end transcription with speaker diarization. Takes audio files (`.m4a`) and produces speaker-labeled transcripts.

## Build & Run

**Build the Docker image:**
```bash
docker build -t whisperx_transcriber .
```

**Run transcription:**
```bash
docker run -it --rm \
  -v $(pwd)/io:/app/io \
  -e HF_TOKEN=<your_huggingface_token> \
  -e WHISPER_MODEL=large-v2 \
  -e NUM_SPEAKERS=2 \
  -e AUDIO_FILE=io/recording.m4a \
  -e OUTPUT_FILE=io/recording.txt \
  whisperx_transcriber
```

Or use the sample shell script: `run_script.sh`

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | required | Hugging Face token for gated diarization model |
| `WHISPER_MODEL` | `large-v2` | Model size: `tiny`, `base`, `small`, `medium`, `large-v2` |
| `NUM_SPEAKERS` | `2` | Expected number of speakers |
| `AUDIO_FILE` | `io/recording.m4a` | Input audio path (relative to container `/app`) |
| `OUTPUT_FILE` | `io/recording.txt` | Output transcript path |
| `COMPUTE_TYPE` | `int8` | PyTorch compute type (falls back to `float32` if unsupported) |

## Architecture

**Single-file pipeline** — `transcribe.py` orchestrates the full flow:

1. **Transcription** — WhisperX model transcribes audio to text with timestamps
2. **Alignment** — Aligns words to audio timestamps with a language-specific model
3. **Diarization** — `pyannote-audio` identifies speaker segments via HuggingFace token
4. **Speaker mapping** — Extra speakers beyond `NUM_SPEAKERS` are merged into the top-N canonical speakers (by total speaking duration)
5. **Output** — Writes `Person 1:`, `Person 2:`, etc. labeled transcript to file

**Key function:** `speaker_for_interval(start, end)` — maps a transcript segment to a speaker by finding maximum temporal overlap with diarization segments.

Device is hardcoded to `"cpu"` in `transcribe.py`. To enable GPU, change `DEVICE = "cpu"` to `"cuda"`.

## Audio Files

Place input `.m4a` files in `io/` before running. Transcripts are written there too. The `io/` folder is gitignored (only `io/.gitkeep` is tracked).

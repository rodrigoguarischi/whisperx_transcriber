# WhisperX Transcriber

This repository provides a simple **Dockerized pipeline** to transcribe audio using [WhisperX](https://github.com/m-bain/whisperX).
It supports **alignment** and **speaker diarization**.

---

## Files

- `Dockerfile` — Defines the container
- `requirements.txt` — Python dependencies
- `transcribe.py` — Main transcription script (WhisperX with diarization)
- `io/` — Drop input audio files here; transcripts are written here too (gitignored)

---

## Usage

### 1. Clone the repository
```bash
git clone https://github.com/rodrigoguarischi/whisperx_transcriber.git
cd whisperx_transcriber
```

### 2. Place your audio file

Put your audio file (e.g., `recording.m4a`) in the `io/` folder.

### 3. Build the Docker image
```bash
docker build -t whisperx_transcriber .
```

### 4. Run transcription
```bash
docker run -it --rm \
  -v $(pwd)/io:/app/io \
  -e HF_TOKEN=hf_xxx \
  -e WHISPER_MODEL=large-v2 \
  -e NUM_SPEAKERS=2 \
  -e AUDIO_FILE=io/recording.m4a \
  -e OUTPUT_FILE=io/recording.txt \
  whisperx_transcriber
```

| Flag | Description |
|---|---|
| `-v $(pwd)/io:/app/io` | Mounts the `io/` folder into the container |
| `HF_TOKEN` | Your Hugging Face access token (required for diarization) |
| `WHISPER_MODEL` | Model to use — `tiny`, `base`, `small`, `medium`, `large-v2` (default: `large-v2`) |
| `NUM_SPEAKERS` | Expected number of speakers |
| `AUDIO_FILE` | Input file path (relative to container) |
| `OUTPUT_FILE` | Output transcript path |

### 5. View the output
```bash
cat io/recording.txt
```

---

## Notes

- Diarization requires a valid Hugging Face token. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- The transcript is saved with speaker labels (e.g., **Person 1: ...**, **Person 2: ...**).
- CPU inference is supported; for faster transcription use a GPU-enabled container.

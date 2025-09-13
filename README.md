# WhisperX Transcriber

This repository provides a simple **Dockerized pipeline** to transcribe audio using [WhisperX](https://github.com/m-bain/whisperX).  
It supports **alignment** and **speaker diarization**, defaulting the output to two speakers.

---

## Files

- `Dockerfile` — Defines the container  
- `requirements.txt` — Python dependencies  
- `transcribe.py` — Main transcription script (WhisperX with diarization)  
- `README.md` — Instructions  

---

## Usage

### 1. Clone the repository
```bash
git clone https://github.com/rodrigoguarischi/whisperx-transcriber.git
cd whisperx-transcriber
```

### 2. Place your audio file

Put your audio file (e.g., audio.m4a) in the project root.
By default, the script looks for audio.m4a, but you can override it via environment variables.

### 3. Build the Docker image
```bash
docker build -t whisperx-transcriber .
```

### 4. Run transcription
```bash
docker run -it --rm -v $(pwd):/app \
  -e HF_TOKEN=hf_xxx \
  -e WHISPER_MODEL=large-v2 \
  -e NUM_SPEAKERS=2 \
  -e AUDIO_FILE=audio.m4a \
  -e OUTPUT_FILE=transcribed.txt \
  whisperx-transcriber
```

Explanation:

 - -v $(pwd):/app → Mounts your working directory inside the container
 - HF_TOKEN=hf_xxx → Your Hugging Face access token (required for diarization)
 - WHISPER_MODEL=large-v2 → Model to use (default: large-v2)
 - NUM_SPEAKERS=2 → Number of diarized speakers (script enforces 2 in this version)
 - AUDIO_FILE=audio.m4a → Input file (must exist in your project root)
 - OUTPUT_FILE=transcribed.txt → Output transcript file.

 ### 5. View the output
```bash
cat transcribed.txt
```

Notes:
------
 - Default model is large-v2 but you can replace it (options: `tiny`, `base`, `small`,` medium`, `large-v2`).
 - Diarization requires a valid Hugging Face token. Get one from: [Hugging Face settings → Access Tokens](https://huggingface.co/settings/tokens).
 - The transcript is saved with speaker labels (e.g., **Person 1: ...**, **Person 2: ...**).
 - CPU inference is supported, but for faster transcription use a GPU-enabled container.
#!/usr/bin/env python3
"""
Transcribe audio.m4a -> transcribed.txt with diarization constrained to NUM_SPEAKERS.
Environment variables:
  HF_TOKEN       : your HuggingFace token (required for gated pyannote model)
  WHISPER_MODEL  : whisper model name (default: large-v2)
  NUM_SPEAKERS   : integer (default: 2)
  AUDIO_FILE     : input filename (default: audio.m4a)
  OUTPUT_FILE    : output filename (default: transcribed.txt)
  COMPUTE_TYPE   : optional compute_type for whisperx.load_model (e.g. int8). Default 'int8'.
"""
import os
from collections import defaultdict
import sys
import math
import whisperx
import pandas as pd

# robust defaults, override with env vars
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v2")
NUM_SPEAKERS = int(os.getenv("NUM_SPEAKERS", "2"))
AUDIO_FILE = os.getenv("AUDIO_FILE", "audio.m4a")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "transcribed.txt")
DEVICE = "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # try int8 for faster CPU, fallback handled below

# Helper to find speaker for a transcript segment (by overlap)
def speaker_for_interval(start, end):
    overlaps = defaultdict(float)
    for _, row in diarization.iterrows():
        label = row["speaker"]
        seg_start, seg_end = row["start"], row["end"]
        overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
        if overlap > 0:
            overlaps[label] += overlap
    if overlaps:
        chosen = max(overlaps.items(), key=lambda x: x[1])[0]
    else:
        # fallback: nearest center
        if centers:
            chosen = min(centers.keys(), key=lambda k: abs(((start + end) / 2.0) - centers[k]))
        else:
            chosen = None
    if chosen in label_map:
        chosen = label_map[chosen]
    return person_names.get(chosen, f"Person ? ({chosen})")

# Code starts here
if HF_TOKEN is None:
    print("Warning: HF_TOKEN not set. If the diarization model is gated you must provide a valid HF token.", file=sys.stderr)

print(f"Model: {MODEL_NAME}  |  Num speakers requested: {NUM_SPEAKERS}  |  File: {AUDIO_FILE}")
print("Loading Whisper model (this can take a while for large models on CPU)...")
try:
    model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
except Exception as e:
    print("Warning: load_model with compute_type failed:", e, file=sys.stderr)
    print("Retrying without compute_type...")
    model = whisperx.load_model(MODEL_NAME, DEVICE)

# Transcribe (get segments)
print("Loading audio and transcribing...")
audio = whisperx.load_audio(AUDIO_FILE)
result = model.transcribe(audio, verbose=False)

# Alignment
print("Loading alignment model and aligning words...")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE)

# Diarization: use the whisperx-provided wrapper (pyannote under the hood)
from whisperx.diarize import DiarizationPipeline

print("Running diarization (may download models the first time)...")
diarize_pipeline = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)

# Try calling with num_speakers argument (preferred)
diarization = None
try:
    diarization = diarize_pipeline(AUDIO_FILE, num_speakers=NUM_SPEAKERS)
except TypeError:
    # API signature might not accept file path + num_speakers together; try other forms:
    try:
        diarization = diarize_pipeline(audio, num_speakers=NUM_SPEAKERS)
    except TypeError:
        # last resort: call without num_speakers and we'll post-process to collapse speakers
        diarization = diarize_pipeline(AUDIO_FILE)

if isinstance(diarization, pd.DataFrame):
    # Build simple stats per speaker
    durations = defaultdict(float)
    center_sums = defaultdict(float)
    counts = defaultdict(int)

    for _, row in diarization.iterrows():
        label = row["speaker"]
        start, end = row["start"], row["end"]
        dur = end - start
        durations[label] += dur
        center_sums[label] += (start + end) / 2.0
        counts[label] += 1

    centers = {
        l: center_sums[l] / counts[l] if counts[l] else 0.0
        for l in center_sums
    }

    all_labels = list(durations.keys())
else:
    raise RuntimeError("Unexpected diarization output type:", type(diarization))

print("Diarization labels found (with total duration in s):")
for l in sorted(durations.keys(), key=lambda x: -durations[x]):
    print(f"  {l} : {durations[l]:.2f}s  (center {centers.get(l,0):.2f}s)")

# Keep top-N by duration as the canonical speakers; map the rest to nearest canonical speaker by center time
sorted_by_duration = sorted(durations.items(), key=lambda kv: -kv[1])
canonical = [t[0] for t in sorted_by_duration[:NUM_SPEAKERS]]
extra = [l for l in all_labels if l not in canonical]

label_map = {}  # map extra->canonical
for l in extra:
    # nearest by center distance
    if canonical:
        nearest = min(canonical, key=lambda c: abs(centers.get(l, 0.0) - centers.get(c, 0.0)))
        label_map[l] = nearest
    else:
        # fallback: map to itself
        label_map[l] = l

print("Canonical speakers chosen:", canonical)
if label_map:
    print("Mapping of extra labels -> canonical:", label_map)

# Friendly names like Person 1, Person 2 (ordered by canonical durations)
person_names = {}
for idx, lab in enumerate(canonical):
    person_names[lab] = f"Person {idx+1}"

# If canonical list is empty (weird edge case), assign discovered labels to Person 1..N
if not person_names:
    for i, lab in enumerate(all_labels[:NUM_SPEAKERS]):
        person_names[lab] = f"Person {i+1}"

# Write final transcript, with speaker labels mapped to Person 1/2...
print(f"Writing diarized transcript to {OUTPUT_FILE} ...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for seg in result_aligned["segments"]:
        s = seg["start"]
        e = seg["end"]
        text = seg["text"].strip()
        speaker = speaker_for_interval(s, e)
        out.write(f"{speaker}: {text}\n")

print("✅ Done — saved to", OUTPUT_FILE)
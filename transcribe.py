import whisperx
import torch
import os
import pandas as pd
from collections import defaultdict
from whisperx.diarize import DiarizationPipeline

AUDIO_FILE = "audio.m4a"
OUTPUT_FILE = "transcribed.txt"
HF_TOKEN = os.getenv("HF_TOKEN")  # set via: docker run -e HF_TOKEN=xxx ...

device = "cpu"
compute_type = "int8"  # faster on CPU

# ----------------------
# Load Whisper model
# ----------------------
print("Loading Whisper model (large-v2)... this may take a while on CPU.")
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# ----------------------
# Transcribe audio
# ----------------------
print("Transcribing audio...")
audio = whisperx.load_audio(AUDIO_FILE)
result = model.transcribe(audio)

# ----------------------
# Alignment
# ----------------------
print("Running alignment...")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

# ----------------------
# Diarization
# ----------------------
print("Running diarization...")
diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
diarization = diarize_model(audio)

if not isinstance(diarization, pd.DataFrame):
    raise RuntimeError(f"Unexpected diarization output type: {type(diarization)}")

# ----------------------
# Force only 2 speakers
# ----------------------
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

# Pick 2 main speakers (longest durations)
main_speakers = sorted(durations.items(), key=lambda x: x[1], reverse=True)[:2]
label_map = {spk: f"SPEAKER_{i}" for i, (spk, _) in enumerate(main_speakers)}

person_names = {"SPEAKER_0": "Person 1", "SPEAKER_1": "Person 2"}

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
    return person_names.get(chosen, f"Unknown ({chosen})")

# ----------------------
# Save transcript
# ----------------------
print("Saving transcript...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for seg in result_aligned["segments"]:
        speaker = speaker_for_interval(seg["start"], seg["end"])
        text = seg["text"].strip()
        f.write(f"{speaker}: {text}\n")

print(f"✅ Transcription complete. Saved to {OUTPUT_FILE}")

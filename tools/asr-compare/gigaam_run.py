"""Transcribe a 16 kHz mono WAV with GigaAM and write one timestamped line per segment.

GigaAM decodes at most 25 s per call. Its own `transcribe_longform` splits with
pyannote, which needs a gated Hugging Face model and a token; Silero VAD needs
neither, so speech regions come from Silero and are packed into windows of at
most MAX_WINDOW_S before decoding.

Usage: gigaam_run.py <in_16k.wav> <out.txt>
Env:   GIGAAM_MODEL (default v3_e2e_rnnt: the variant with punctuation)
       GIGAAM_DOWNLOAD_ROOT (where checkpoints are cached; set by compare.sh)
"""

import os
import sys

import gigaam
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SAMPLE_RATE = 16_000
MAX_WINDOW_S = 22.0  # below GigaAM's 25 s limit, with headroom for padding
MAX_GAP_S = 1.0  # regions further apart than this start a new window


def speech_windows(wav: torch.Tensor) -> list[tuple[float, float]]:
    regions = get_speech_timestamps(
        wav,
        load_silero_vad(),
        sampling_rate=SAMPLE_RATE,
        max_speech_duration_s=MAX_WINDOW_S,
        return_seconds=True,
    )
    windows: list[tuple[float, float]] = []
    for region in regions:
        start, end = region["start"], region["end"]
        if windows:
            w_start, w_end = windows[-1]
            if start - w_end <= MAX_GAP_S and end - w_start <= MAX_WINDOW_S:
                windows[-1] = (w_start, end)
                continue
        windows.append((start, end))
    return windows


def timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    model_name = os.environ.get("GIGAAM_MODEL", "v3_e2e_rnnt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = gigaam.load_model(
        model_name,
        device=device,
        download_root=os.environ.get("GIGAAM_DOWNLOAD_ROOT"),
    )

    samples, rate = sf.read(in_path, dtype="float32")
    if rate != SAMPLE_RATE:
        sys.exit(f"expected {SAMPLE_RATE} Hz input, got {rate}")
    wav = torch.from_numpy(samples)

    lines = []
    with torch.inference_mode():
        for start, end in speech_windows(wav):
            piece = wav[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
            batch = piece.to(model._device).to(model._dtype).unsqueeze(0)
            length = torch.full([1], batch.shape[-1], device=model._device)
            encoded, encoded_len = model.forward(batch, length)
            text, _ = model._decode(encoded, encoded_len, length)[0]
            if text.strip():
                lines.append(f"[{timestamp(start)}] {text.strip()}")

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
    print(f"gigaam: {model_name} on {device}, {len(lines)} segments")


if __name__ == "__main__":
    main()

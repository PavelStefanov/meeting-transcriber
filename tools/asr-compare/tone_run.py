"""Transcribe an 8 kHz mono WAV with T-one and write one timestamped line per phrase.

T-one is a telephony model: it takes 8 kHz int32 samples and segments phrases
itself, so the whole file goes through `forward_offline` in one call. Its output
has no punctuation or capitals; that is the model, not this script.

Usage: tone_run.py <in_8k.wav> <out.txt>
Env:   HF_HOME (where the acoustic model and LM are cached; set by compare.sh)
       TONE_DECODER=greedy skips the KenLM beam search. The LM is a 5.5 GB
       download; greedy needs only the 144 MB acoustic model, at some cost in
       accuracy. Default is beam, the pipeline's own default.
"""

import os
import sys

import numpy as np
import soundfile as sf
from tone import DecoderType, StreamingCTCPipeline

SAMPLE_RATE = 8_000


def timestamp(seconds: float) -> str:
    total = int(max(seconds, 0))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    samples, rate = sf.read(in_path, dtype="int16")
    if rate != SAMPLE_RATE:
        sys.exit(f"expected {SAMPLE_RATE} Hz input, got {rate}")

    greedy = os.environ.get("TONE_DECODER") == "greedy"
    decoder = DecoderType.GREEDY if greedy else DecoderType.BEAM_SEARCH
    pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=decoder)
    phrases = pipeline.forward_offline(samples.astype(np.int32))

    lines = [f"[{timestamp(p.start_time)}] {p.text.strip()}" for p in phrases if p.text.strip()]
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
    print(f"t-one: {'greedy' if greedy else 'beam+kenlm'}, {len(lines)} phrases")


if __name__ == "__main__":
    main()

# Russian ASR engine comparison (October 2026)

Status: **decided — keep the whisper.cpp engine (full f16 large-v3).**
Tooling: `tools/asr-compare/compare.sh`, branch `test/asr-compare`.

## Question

Is there a better engine than the fork's whisper.cpp large-v3 for Russian work
meetings, where a large share of the vocabulary is English: acronyms, product
and service names, and English words used with Russian inflection? Candidates
were the two open Russian-specific models and the WhisperKit model the app's
picker already offers.

| # | Engine | Model | Runtime |
|---|--------|-------|---------|
| 1 | GigaAM (Sber) | `v3_e2e_rnnt` (the variant with punctuation), MIT | PyTorch, MPS |
| 2 | T-one (T-Bank) | streaming CTC + 5.5 GB KenLM beam search, telephony 8 kHz | ONNX Runtime, CPU |
| 3 | whisper.cpp (the fork's engine) | `ggml-large-v3.bin`, f16, 32 decoder layers | Metal, VAD chunking as in the pipeline |
| 4 | WhisperKit, picker "Large V3" | `openai_whisper-large-v3-v20240930` | CoreML |

**The picker's "Large V3" is not large-v3.** Argmax's
`openai_whisper-large-v3-v20240930` and `..._turbo` both carry the
large-v3-turbo checkpoint (`decoder_layers: 4` in their `config.json`); the
`_turbo` suffix is Argmax's own optimisation, not the model. The real
large-v3 (`decoder_layers: 32`) is published as `openai_whisper-large-v3` and
is not in the picker. Any "whisper.cpp beat WhisperKit Large V3" result
therefore compares two different models, not two runtimes.

## Method

- Two real dual-source recordings of two-person calls, chosen from the local
  archive by statistics over their existing transcripts: exactly two speakers,
  highest speaker-turn rate (a fast back-and-forth), and highest density of
  Latin-script and anglicism tokens. Input was the `_mix.wav` track, so this is
  close to, not identical with, the pipeline's per-track decode.
- No reference transcript exists, so there is no WER. Evidence is (a) automated
  counts over each output and (b) side-by-side reading of the same time window
  in all four outputs, roughly ten windows per recording, chosen at the moments
  where English terms occur.
- Each recording's audio and transcripts stay outside git. The numbers below
  are the whole of what is recorded here.

## Results

Wall clock on an M-series Mac, models already downloaded:

| Recording | Length | GigaAM | T-one | whisper.cpp | WhisperKit |
|-----------|--------|--------|-------|-------------|------------|
| A (fast turn-taking) | 13.9 min | 29 s | 35 s | 321 s ¹ | 95 s |
| B (anglicism-dense) | 31.9 min | 60 s | 72 s | 203 s | 166 s |

¹ includes loading the 3 GB model cold; B ran with it warm.

Word and token counts:

| | GigaAM | T-one | whisper.cpp | WhisperKit |
|---|---|---|---|---|
| Words, A | 1398 | 1437 | 1306 | 1131 |
| Words, B | 3612 | 3676 | 3412 | 3370 |
| Hesitation fillers kept, B | 58 | 30 | 15 | 3 |
| Latin-script tokens, B | 35 | 0 | 54 | 36 |

English-only terms in B (acronyms, hyphenated English compounds, an English
noun with no established Russian spelling — five distinct terms):

| | GigaAM | T-one | whisper.cpp | WhisperKit |
|---|---|---|---|---|
| Written recognisably | 0 / 5 | 0 / 5 | 5 / 5 | 2 / 5 |

### Per engine

- **whisper.cpp large-v3** — the only engine that writes English terms
  correctly and consistently. Weaknesses: it normalises speech (drops fillers
  and discourse particles, occasionally whole short phrases), and in a handful
  of sampled windows it substituted a Russian word that is phonetically close
  but impossible in context, or inserted a word nobody said.
- **GigaAM v3** — the most faithful Russian: verbatim, fillers kept, and the
  best at Russian words that the Whisper models misheard. English words that
  have a settled Russified jargon form come out in that Cyrillic form, which is
  fine for a protocol. English-only terms come out as phonetic Cyrillic or as
  broken Latin (a word truncated mid-way, a nonsense acronym); the model cannot
  produce them correctly at all. Also the fastest by far.
- **T-one** — lowercase, no punctuation (by design, telephony model), and the
  most mishearings of the four. Not suitable for meeting transcripts.
- **WhisperKit "Large V3" (= turbo)** — fragments segments and loses content
  (13 % fewer words than whisper.cpp on A); English terms hit or miss.

## Decision

Keep whisper.cpp large-v3. English terms matter in these meetings, and it is the
only engine that gets them right. Its weaknesses mostly do not survive protocol
generation (the LLM drops fillers anyway), while GigaAM's English failures
would. GigaAM would also cost a new native runtime (PyTorch or ONNX), its own
chunking (25 s input limit) and Russian-only gating.

## Open threads (how to continue)

1. **Fair WhisperKit comparison, for any upstream proposal.** Run
   `openai_whisper-large-v3` (the real one) through the same script:
   `compare.sh <file> --only whisperkit --whisperkit-variant openai_whisper-large-v3`.
   The model has to be in
   `~/Library/Application Support/MeetingTranscriber/models/whisperkit/models/argmaxinc/whisperkit-coreml/`.
   Without this, the case for a third engine upstream rests on a model
   difference, not a runtime one.
2. **Terminology rules** (Settings, `TerminologyNormalizer`,
   `Canonical => variant | variant`) can fix whisper.cpp's repeat
   substitutions and any engine's stable mis-spellings, post-ASR. A
   `compare.sh --rules <file>` that applies the same normaliser to all four
   outputs would measure that without re-running the models.
3. **Vocabulary prompt for whisper.cpp.** The fork's whisper.cpp engine ignores
   the custom vocabulary file today (WhisperKit uses it as a 32-token prompt,
   Parakeet for CTC boosting). Wiring it to `initial_prompt` is an app change.
4. **GigaAM as an engine** stays an option only if Russian fidelity starts to
   matter more than English terms; MIT licence, ~450 MB checkpoint, ONNX export
   and sherpa-onnx builds exist.

# whisper.cpp: why the decode is chunked, and what it measured

Reference record for the `whisper.cpp` engine's decode framing. Committed
because the numbers below are the only reason the code is shaped the way it is,
and because they were expensive to obtain: each row is a full decode of a real
30-minute recording.

**Status:** measured and settled. The framing decision is not open; the
remaining open questions are listed at the end.

## The problem this solved

The engine was added to reproduce a measured result — full f16 large-v3 through
whisper.cpp beating the WhisperKit path on Russian meeting audio, as the
reference pipeline (Meetily) achieves it. As first written it did not
reproduce that result. It was *worse* than the reference on the same audio.

Measured on a 29.8-minute Russian meeting, against the reference pipeline's own
transcript of the same file:

| variant | words | repeated 8-grams | punctuation /100 words | capitals | wall clock |
|---|---|---|---|---|---|
| whole file (as first written) | 4053 | **344 (8.5%)** | **1.2** | 22 | 6:12 |
| VAD, concatenated into one stream | 3433 | 0 | 4.5 | 61 | 26:00 |
| **chunked** | 3633 | 0 | 26.5 | 515 | 6:10 |
| **chunked + short-chunk merging** | **3645** | **0** | **26.8** | 505 | **5:26** |
| reference pipeline | 3430 | 0 | 24.6 | 431 | — |

Repeated 8-grams measure the decoder's repetition loop: one phrase appeared six
times in a row. Punctuation density measures whether the output is sentences or
an unbroken lowercase stream.

## The cause was framing, not the model

Same weights, same decode parameters, same `WhisperCppDecodingConfig`. What
differed was the shape of the input.

The reference cuts a recording into speech-bounded chunks and decodes each
separately with `no_timestamps`, taking timing from its own chunk offsets. This
engine handed over the whole file and let whisper.cpp cut its own 30 s windows.
Those land mid-sentence; `no_context` leaves no prompt to recover from it; and
the decoder advances by timestamp token on a single decoding path (the beam
width is inert — see `WhisperCppDecodingConfig`), so a mispredicted timestamp
re-decodes audio it has already seen.

Both defects follow: no sentence boundary to punctuate, and a window that can
fail to advance.

Direct confirmation, from the same engine and parameters: a short
single-utterance file came back fully punctuated and capitalised, while the
30-minute file did not. Only the length differed.

## What "VAD" alone does not fix

Enabling the existing VAD preprocessing removes repetition (344 → 0) and brings
the word count to within three words of the reference. It does **not** restore
punctuation (4.5 vs 24.6) and it costs 4x the wall clock.

The reason is that VAD trimming concatenates all speech into one long stream and
still hands over a single file. It removes the silence but not the stream, and
the stream is the problem. This is a trap worth remembering: the metric that
looks most alarming (repetition) is fixed, so the change looks successful.


## Measured against the reference pipeline

Two Russian recordings, both decoded by the reference pipeline (Meetily)
through the same engine and model, so the comparison isolates framing.

A 29.8-minute meeting:

| | this app | reference |
|---|---|---|
| words | 3667 | 3430 |
| repeated 8-grams | 0 | 0 |
| punctuation /100 words | 27.1 | 24.6 |
| capitals | 509 | 431 |
| speaker attribution | yes, 43 turns | none — its output carries no speaker field |

Word-level alignment of the two transcripts, which is the part that says
something about recognition rather than formatting:

| | words |
|---|---|
| agreed | 2832 |
| **only this app** — speech the reference missed | **243** (15 runs of 4+ words) |
| only the reference — speech this app missed | 74 (5 runs) |
| substituted | 449 / 513 |

So 3.3x more speech recovered than lost. Whole utterances the reference
dropped entirely include greetings and several complete exchanges.

Among the substitutions, the cases where meaning decides went to this app in
every instance checked: the reference returned words that were impossible in
context — a homophone of the right word, carrying an unrelated sense — where
this app returned the one the sentence required. Four such pairs were verified
by hand. The reference wins a smaller number of spots, and this app still
returns the occasional word split in half by a space.

A 2:17 interview clip, where an independent third transcript (YouTube's
auto-captions) is also available:

| | divergence from reference | divergence from third-party |
|---|---|---|
| this app | 0.101 | **0.210** |
| reference | — | 0.224 |

Being closer to an unrelated system than the reference is weak evidence of
being closer to the truth. It is not proof, and neither of those transcripts
is ground truth.

**Read these numbers as divergence, not error.** No hand-verified Russian
reference exists, so nothing here measures accuracy directly. Word counts,
punctuation density and repetition are objective; who is right in a given
substitution was judged by semantic plausibility, sampled rather than
exhaustive.

## Chunk sizing: the mirror-image fault

Chunking introduced the opposite failure — a chunk can be too *short*, leaving
the decoder nothing to disambiguate against. Only visible once WER was measured
per fixture, whole-file against chunked:

| fixture | duration | whisper windows | whole file | chunked |
|---|---|---|---|---|
| `two_speakers_de` | 17.0 s | one | **0.179** | 0.500 |
| `three_speakers_de` | 29.8 s | one | **0.283** | 0.302 |
| `four_speakers_en_ami` | 93.8 s | four | 0.311 | **0.260** |

The sign flips exactly at the 30 s window boundary. That is the mechanism, not a
coincidence: a recording inside one window is already decoded in a single pass
with the whole file as context — no window advance, so no timestamp to
mispredict, so none of the repetition chunking exists to prevent. Splitting it
only removes context.

Two guards follow, both the same principle — never hand the decoder less context
than one decode needs:

1. **Short neighbours are merged**, across pauses up to a second, only until the
   chunk reaches a workable length (`targetChunkSeconds`). Alone this took
   `two_speakers_de` from 0.500 to 0.357. It merges across a breath, never
   across a turn boundary, so a chunk still begins and ends where speech does
   and one segment never spans two speakers.
2. **Recordings inside one window are decoded whole** (`shouldChunk`).

## Rejected alternatives

- **Padding each chunk with surrounding audio.** Gives context, but the decoder
  transcribes the padding too, duplicating words at every seam.
- **Carrying the previous chunk's text as an initial prompt.** A known cause of
  the very repetition loops this path removed — which is why the reference
  configuration sets `no_context` at all.
- **Loosening the WER threshold** to absorb the one failing measurement. That
  turns the measurement into decoration; the failure was a real regression and
  was fixed instead.
- **Changing `WhisperCppDecodingConfig.meetilyParity`.** Untouched throughout.
  The chunked path flips `no_timestamps` and `single_segment` on a local copy.

## The `no_timestamps` divergence is retired, not worked around

That flag was left off only because whole-file decoding had no other source of
per-utterance timing, which diarization, the dual-track merge and the echo dedup
all need. A chunk supplies its own timing, and a VAD region is better bounded
than a predicted timestamp token because it marks where speech actually stopped.

Verified on the fixture with engineered silence in [8 s, 13 s]: the chunked path
places a segment at 00:14, matching the whole-file baseline's 00:13.

## Costs accepted

- **Coarser timestamps.** Merging produced 43 transcript lines where the
  unmerged split produced 53. Timing is now per merged region (up to
  `targetChunkSeconds`) rather than per whisper segment. Speaker attribution
  stayed sensible on the measured recording, but this is the knob to turn if
  speakers start bleeding together.
- **The chunked path requires VAD.** It engages only when `vadEnabled` is on,
  which is **off by default**. With VAD off the engine still takes the
  whole-file path and reproduces every defect above.

## Open questions

- **No Russian WER.** Every Russian figure here is a proxy — repetition,
  punctuation, volume. Recognition accuracy on Russian is unmeasured because the
  project's quality fixtures are German and English. A Russian fixture needs a
  hand-verified reference; generating one would compare against another engine's
  hypothesis rather than against truth.
- **`targetChunkSeconds` is fitted, not derived.** 8 s came from the region
  average of the one recording where chunking helped, which is reasoning about
  a particular file. The principled statement is "as much context as fits in a
  window without crossing a turn boundary", and it collides with timestamp
  granularity for diarization. Settle it by sweeping the value across the
  fixtures and reading WER, not by picking from one observation.
- **Word splits from the decoder itself** are untouched by chunk placement,
  since they occur where no cut was made. They would need post-processing over
  the decoded text.
- **In-app model download** was interrupted twice by memory contention during
  this work and completed out of band; the in-app path is only verified as far
  as ~1.1 GB of the 2.9 GB transfer.

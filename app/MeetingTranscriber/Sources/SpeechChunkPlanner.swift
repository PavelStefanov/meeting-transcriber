import Foundation

/// Turns VAD speech regions into the chunks a chunk-batch engine decodes one
/// at a time.
///
/// Pure and value-typed so the planning is testable without a model or an
/// audio file — the same split `WhisperCppEngine+Chunked` feeds to the decoder.
///
/// ## Why chunks and not the whole recording
///
/// Whisper's output quality depends on how the audio is framed, not only on
/// the weights. Handed a long stream it cuts its own 30 s windows at arbitrary
/// offsets, so a window routinely starts mid-sentence; with `no_context` there
/// is no prompt to recover from that, and the result is unpunctuated run-on
/// text. Handed one speech-bounded utterance it emits a finished sentence,
/// capital letter and full stop included. Measured on real Russian meeting
/// audio: the whole-file path produced 1.2 punctuation marks per 100 words
/// against the reference pipeline's 24.6, while a short single-utterance file
/// through the identical engine and parameters came back fully punctuated.
///
/// The same framing is what keeps the decoder out of its repetition loop. A
/// long stream advances its window by timestamp tokens and decodes with a
/// single path (the beam width is inert — see `WhisperCppDecodingConfig`), and
/// that combination re-decodes the same audio when a timestamp comes back
/// wrong: 8.5% of all 8-grams on that recording were verbatim repeats. Speech
/// regions decoded separately cannot loop into each other.
enum SpeechChunkPlanner {
    /// Longest chunk handed to one decode.
    ///
    /// 30 s is whisper's own window (`WHISPER_CHUNK_SIZE`), so a chunk at or
    /// below it is exactly one decode: no internal window advance, and
    /// therefore no timestamp token to get wrong. A longer chunk would
    /// reintroduce the multi-window behaviour this planner exists to avoid.
    static let maxChunkSeconds: TimeInterval = 30

    /// Length a chunk is grown towards by merging neighbours before it is
    /// decoded.
    ///
    /// A chunk that is too SHORT costs accuracy, which is the mirror image of
    /// the problem chunking solves and was only visible once WER was measured
    /// per fixture: on a synthetic fixture whose turns are ~4 s apart, the
    /// split produced 4 s chunks and WER went from 0.179 (whole file) to 0.500,
    /// with substitutions tripling — the decoder had no context left to
    /// disambiguate against. On the real overlapping meeting, where regions
    /// average around 8 s, chunking instead improved WER to 0.266 from 0.311.
    /// 8 s is that working range, and merging only ever moves a chunk towards
    /// it.
    static let targetChunkSeconds: TimeInterval = 8

    /// Longest pause merged across while growing a chunk.
    ///
    /// A short pause inside a chunk is harmless — the decoder handles internal
    /// silence, and the chunk still begins and ends where speech does. A long
    /// one is not: bridging it would pack dead air into the decode and hand
    /// back one segment covering two conversational turns, which is worse for
    /// speaker attribution than a short chunk. So merging follows the
    /// conversation's own rhythm rather than a target length at any cost.
    static let maxBridgedGapSeconds: TimeInterval = 1.0

    /// Shortest chunk worth decoding on its own.
    ///
    /// Below this a region is almost always a breath or a click that VAD let
    /// through, and a decode of it returns either nothing or a hallucinated
    /// filler — Whisper is known to invent text for near-silent input. Dropped
    /// rather than merged into a neighbour: merging would bridge a real pause
    /// and hand the decoder the boundary it mishandles.
    static let minChunkSeconds: TimeInterval = 0.2

    /// Whether a recording of `duration` is long enough for chunking to be
    /// worth anything.
    ///
    /// Below one decode window there is nothing to win and something to lose.
    /// A recording that fits in a single window IS already decoded in one pass
    /// with the whole file as context: no window advance, so no timestamp token
    /// to mispredict, so none of the repetition this planner exists to prevent.
    /// Splitting it only takes context away, and WER says so — measured per
    /// fixture, whole-file against chunked:
    ///
    ///   17.0 s (one window)     0.179  vs  0.357
    ///   29.8 s (one window)     0.283  vs  0.321
    ///   93.8 s (four windows)   0.311  vs  0.260
    ///
    /// The sign flips exactly at the window boundary, which is the mechanism
    /// rather than a coincidence, so that is where the gate sits.
    static func shouldChunk(duration: TimeInterval) -> Bool {
        duration > maxChunkSeconds
    }

    /// Split `regions` so no chunk exceeds `maxChunkSeconds`.
    ///
    /// Region boundaries are always kept — they are where the speech actually
    /// stops, which is the whole value of the input. Only a region longer than
    /// the limit is divided, and then into equal parts rather than
    /// greedy 30 s slices plus a short tail: a 31 s region becomes two chunks
    /// of 15.5 s instead of 30 s and 1 s, and the 1 s tail is exactly the
    /// fragment that decodes badly.
    static func plan(
        regions: [SpeechRegion],
        maxChunkSeconds: TimeInterval = maxChunkSeconds,
    ) -> [SpeechRegion] {
        guard maxChunkSeconds > 0 else { return [] }
        var chunks: [SpeechRegion] = []
        for region in merged(regions, maxChunkSeconds: maxChunkSeconds) {
            let duration = region.duration
            guard duration >= minChunkSeconds else { continue }
            if duration <= maxChunkSeconds {
                chunks.append(region)
                continue
            }
            // `ceil` so the part count is the smallest that fits under the
            // limit; dividing by it then puts every part equally under it.
            let parts = Int(ceil(duration / maxChunkSeconds))
            let step = duration / Double(parts)
            for index in 0 ..< parts {
                let start = region.start + step * Double(index)
                // Last part ends exactly on the region end rather than on an
                // accumulated multiple of `step`, so floating-point drift
                // cannot leave a sliver of audio undecoded.
                let end = index == parts - 1 ? region.end : start + step
                chunks.append(SpeechRegion(start: start, end: end))
            }
        }
        return chunks
    }

    /// How far a split may move to find a quiet moment.
    ///
    /// Wide enough to reach the next gap between words at conversational pace,
    /// narrow enough that the split stays roughly where the even division put
    /// it and no chunk grows appreciably.
    static let splitSearchSeconds: TimeInterval = 0.75

    /// Window the loudness of a candidate split point is judged over. About one
    /// phoneme, so a single glottal stop inside a word does not read as a gap.
    private static let splitProbeSeconds: TimeInterval = 0.03

    /// Move interior splits onto the quietest nearby moment.
    ///
    /// A region boundary is silence by construction, so it never lands inside a
    /// word. An interior split does: `plan` divides an over-long region evenly,
    /// which is arithmetic and knows nothing about the audio, so the cut falls
    /// wherever it falls. Observed twice in real output as a word returned in
    /// two halves — where each half decoded separately and joining them
    /// inserted a space inside a word. On the 137 s sample this fires for real:
    /// 5 speech regions became 7 chunks, so two cuts were made blind.
    ///
    /// Only splits between chunks that actually touch are moved, so a real
    /// pause between two regions is never dragged; and the movement is clamped
    /// so neither neighbour can cross a decode window, which is the invariant
    /// the whole split exists to maintain.
    ///
    /// Pure, taking the samples as an argument rather than reading a file, so
    /// the choice is testable against a synthetic buffer.
    static func snapSplitsToQuietMoments(
        _ chunks: [SpeechRegion],
        samples: [Float],
        sampleRate: Int,
        maxChunkSeconds: TimeInterval = maxChunkSeconds,
    ) -> [SpeechRegion] {
        guard chunks.count > 1, sampleRate > 0, !samples.isEmpty else { return chunks }
        var out = chunks
        for index in 1 ..< out.count {
            let previous = out[index - 1]
            let next = out[index]
            // Touching means `plan` cut here; a gap means speech stopped and
            // the boundary is already in silence.
            guard abs(next.start - previous.end) < 0.001 else { continue }

            // Never let either side cross a window, and never collapse a side
            // to nothing.
            let earliest = max(
                next.start - splitSearchSeconds,
                max(previous.start + minChunkSeconds, next.end - maxChunkSeconds),
            )
            let latest = min(
                next.start + splitSearchSeconds,
                min(next.end - minChunkSeconds, previous.start + maxChunkSeconds),
            )
            guard earliest < latest,
                  let moved = quietestPoint(
                      between: earliest, and: latest, samples: samples, sampleRate: sampleRate,
                  )
            else { continue }

            out[index - 1] = SpeechRegion(start: previous.start, end: moved)
            out[index] = SpeechRegion(start: moved, end: next.end)
        }
        return out
    }

    /// Time in `earliest ... latest` whose surrounding audio is quietest.
    private static func quietestPoint(
        between earliest: TimeInterval,
        and latest: TimeInterval,
        samples: [Float],
        sampleRate: Int,
    ) -> TimeInterval? {
        let probe = max(1, Int(splitProbeSeconds * Double(sampleRate)))
        // Step by half a probe: fine enough to find a between-word gap, coarse
        // enough that a two-second search is a few dozen sums rather than
        // tens of thousands.
        let step = max(1, probe / 2)
        let first = Int(earliest * Double(sampleRate))
        let last = Int(latest * Double(sampleRate))
        guard first < last else { return nil }

        var bestPosition: Int?
        var bestEnergy = Float.greatestFiniteMagnitude
        var position = first
        while position <= last {
            let from = max(0, position - probe / 2)
            let to = min(samples.count, from + probe)
            guard from < to else { break }
            var energy: Float = 0
            for sample in samples[from ..< to] {
                energy += sample * sample
            }
            if energy < bestEnergy {
                bestEnergy = energy
                bestPosition = position
            }
            position += step
        }
        guard let bestPosition else { return nil }
        return Double(bestPosition) / Double(sampleRate)
    }

    /// Grow short regions into neighbours so the decoder gets enough audio to
    /// disambiguate against.
    ///
    /// Runs before the split, not after: merging first and splitting second
    /// means a run of short turns becomes one well-sized chunk, while the split
    /// still guarantees nothing exceeds a decode window. Doing it the other way
    /// round could re-join parts the split had just separated.
    ///
    /// Only pauses up to `maxBridgedGapSeconds` are crossed, and growth stops
    /// as soon as the chunk reaches `targetChunkSeconds` — so a chunk gets
    /// context without swallowing a turn boundary it has no reason to.
    private static func merged(
        _ regions: [SpeechRegion],
        maxChunkSeconds: TimeInterval,
    ) -> [SpeechRegion] {
        var out: [SpeechRegion] = []
        for region in regions {
            guard let open = out.last else {
                out.append(region)
                continue
            }
            let gap = region.start - open.end
            let grown = region.end - open.start
            let joinable = open.duration < targetChunkSeconds
                && gap >= 0
                && gap <= maxBridgedGapSeconds
                && grown <= maxChunkSeconds
            if joinable {
                out[out.count - 1] = SpeechRegion(start: open.start, end: region.end)
            } else {
                out.append(region)
            }
        }
        return out
    }

    /// Sample range of `chunk` in a buffer at `sampleRate`, clamped to
    /// `sampleCount`.
    ///
    /// Returns nil when the range is empty or starts past the buffer, which is
    /// what a region derived from a slightly longer decode of the same file
    /// looks like — the caller skips it instead of slicing out of bounds.
    static func sampleRange(
        for chunk: SpeechRegion,
        sampleRate: Int,
        sampleCount: Int,
    ) -> Range<Int>? {
        let start = max(0, Int(chunk.start * Double(sampleRate)))
        let end = min(sampleCount, Int(chunk.end * Double(sampleRate)))
        guard start < end, start < sampleCount else { return nil }
        return start ..< end
    }
}

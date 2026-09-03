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

    /// Shortest chunk worth decoding on its own.
    ///
    /// Below this a region is almost always a breath or a click that VAD let
    /// through, and a decode of it returns either nothing or a hallucinated
    /// filler — Whisper is known to invent text for near-silent input. Dropped
    /// rather than merged into a neighbour: merging would bridge a real pause
    /// and hand the decoder the boundary it mishandles.
    static let minChunkSeconds: TimeInterval = 0.2

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
        for region in regions {
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

import Foundation

/// Turns whisper.cpp's raw segment output into the `[TimestampedSegment]` the
/// rest of the pipeline consumes.
///
/// Pure and value-typed so the mapping is testable without a model: the C
/// bridge reads `whisper_full_get_segment_text/_t0/_t1` into `RawSegment`s and
/// hands them here.
///
/// The filtering matches `WhisperKitEngine.transcribeSegments` deliberately —
/// same token stripping, same consecutive-duplicate drop — so switching engines
/// changes the recognition and not the shape of what lands in the transcript.
enum WhisperCppSegmentBuilder {
    /// One segment exactly as whisper.cpp reports it.
    ///
    /// `Sendable` explicitly rather than by inference: these cross out of
    /// `WhisperCppRunner`'s serial queue into the `@MainActor` engine, and an
    /// inferred conformance is one field away from being lost silently.
    struct RawSegment: Equatable, Sendable {
        let text: String
        /// `whisper_full_get_segment_t0`, in centiseconds.
        let startCentiseconds: Int64
        /// `whisper_full_get_segment_t1`, in centiseconds.
        let endCentiseconds: Int64
    }

    /// whisper.cpp reports segment bounds in hundredths of a second.
    static let centisecondsPerSecond: TimeInterval = 100

    /// Map, clean and sanity-check.
    ///
    /// Sanitisation is deliberately minimal — negatives clamped to zero and an
    /// end never before its start — and specifically does NOT force the
    /// sequence to be globally monotonic. Segment times drive diarization
    /// matching, and dragging a segment forward to satisfy an ordering
    /// invariant would move real speech to a time it did not happen at, which
    /// is worse for speaker attribution than a slightly overlapping pair.
    static func segments(from raw: [RawSegment]) -> [TimestampedSegment] {
        var result: [TimestampedSegment] = []
        result.reserveCapacity(raw.count)
        var previousText = ""
        for item in raw {
            let text = WhisperKitEngine.stripWhisperTokens(item.text)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            // Empty after stripping means the segment was nothing but special
            // tokens; an exact repeat of the previous line is whisper's classic
            // repetition loop and is dropped for the same reason WhisperKit
            // drops it.
            if text.isEmpty || text == previousText { continue }
            previousText = text

            let start = max(TimeInterval(item.startCentiseconds) / centisecondsPerSecond, 0)
            let end = max(TimeInterval(item.endCentiseconds) / centisecondsPerSecond, start)
            result.append(TimestampedSegment(start: start, end: end, text: text))
        }
        return result
    }
}

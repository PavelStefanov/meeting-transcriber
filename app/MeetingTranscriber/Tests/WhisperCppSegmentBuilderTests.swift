@testable import MeetingTranscriber
import XCTest

/// The boundary between whisper.cpp's output and everything downstream that
/// reads `TimestampedSegment` — diarization, the dual-track merge, the echo
/// dedup and the rendered transcript. A units mistake here would not fail
/// loudly; it would put every speaker label in the wrong place.
final class WhisperCppSegmentBuilderTests: XCTestCase {
    // MARK: - Units

    /// whisper.cpp reports centiseconds. Getting this wrong by a factor of 100
    /// is the single most damaging silent bug available at this boundary: an
    /// hour-long meeting would map onto 36 seconds and diarization would assign
    /// every segment to whoever spoke first.
    func test_segments_convertCentisecondsToSeconds() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "Привет", startCentiseconds: 150, endCentiseconds: 275),
        ])
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].start, 1.5, accuracy: 0.0001)
        XCTAssertEqual(result[0].end, 2.75, accuracy: 0.0001)
    }

    // MARK: - Cleaning

    func test_segments_trimTheLeadingSpaceWhisperEmits() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: " добрый день ", startCentiseconds: 0, endCentiseconds: 100),
        ])
        XCTAssertEqual(result.map(\.text), ["добрый день"])
    }

    func test_segments_stripSpecialTokens() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "<|ru|>текст<|endoftext|>", startCentiseconds: 0, endCentiseconds: 100),
        ])
        XCTAssertEqual(result.map(\.text), ["текст"])
    }

    func test_segments_dropASegmentThatWasNothingButTokens() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "<|nospeech|>", startCentiseconds: 0, endCentiseconds: 100),
            .init(text: "real", startCentiseconds: 100, endCentiseconds: 200),
        ])
        XCTAssertEqual(result.map(\.text), ["real"])
    }

    func test_segments_dropEmptyAndWhitespaceOnlyText() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "", startCentiseconds: 0, endCentiseconds: 50),
            .init(text: "   \n ", startCentiseconds: 50, endCentiseconds: 100),
        ])
        XCTAssertTrue(result.isEmpty)
    }

    /// Same filter `WhisperKitEngine.transcribeSegments` applies, so switching
    /// engines does not change what a repetition loop looks like in the saved
    /// transcript.
    func test_segments_dropConsecutiveDuplicates() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "да", startCentiseconds: 0, endCentiseconds: 100),
            .init(text: "да", startCentiseconds: 100, endCentiseconds: 200),
            .init(text: "да", startCentiseconds: 200, endCentiseconds: 300),
        ])
        XCTAssertEqual(result.map(\.text), ["да"])
    }

    /// Only *consecutive* ones. A meeting genuinely contains the same short
    /// utterance more than once, and dropping every repeat would delete real
    /// speech.
    func test_segments_keepANonConsecutiveRepeat() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "да", startCentiseconds: 0, endCentiseconds: 100),
            .init(text: "нет", startCentiseconds: 100, endCentiseconds: 200),
            .init(text: "да", startCentiseconds: 200, endCentiseconds: 300),
        ])
        XCTAssertEqual(result.map(\.text), ["да", "нет", "да"])
    }

    // MARK: - Sanity limits

    func test_segments_clampANegativeStartToZero() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "x", startCentiseconds: -50, endCentiseconds: 100),
        ])
        XCTAssertEqual(result[0].start, 0)
    }

    /// `token_timestamps` is experimental and `whisper_wrap_segment` derives
    /// split points from it, so an end before its start is reachable. A
    /// negative-length segment would give the diarization overlap search an
    /// empty window.
    func test_segments_neverEndBeforeTheyStart() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "x", startCentiseconds: 500, endCentiseconds: 200),
        ])
        XCTAssertEqual(result[0].start, 5.0, accuracy: 0.0001)
        XCTAssertEqual(result[0].end, 5.0, accuracy: 0.0001)
    }

    /// Deliberately NOT forcing global monotonicity: dragging a segment forward
    /// to satisfy an ordering invariant would claim speech happened at a time
    /// it did not, which is worse for speaker attribution than an overlap.
    func test_segments_leaveAnOverlappingPairWhereWhisperPutIt() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "a", startCentiseconds: 0, endCentiseconds: 500),
            .init(text: "b", startCentiseconds: 300, endCentiseconds: 800),
        ])
        XCTAssertEqual(result[1].start, 3.0, accuracy: 0.0001)
    }

    // MARK: - Downstream shape

    func test_segments_carryNoSpeakerAndAreNotSuppressed() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "x", startCentiseconds: 0, endCentiseconds: 100),
        ])
        XCTAssertEqual(result[0].speaker, "", "the diarization stage owns speaker labels")
        XCTAssertFalse(result[0].suppressed, "the echo classifier owns suppression")
    }

    func test_segments_renderThroughTheSharedTranscriptFormatter() {
        let result = WhisperCppSegmentBuilder.segments(from: [
            .init(text: "первое", startCentiseconds: 0, endCentiseconds: 100),
            .init(text: "второе", startCentiseconds: 6500, endCentiseconds: 7000),
        ])
        XCTAssertEqual(result.transcriptText, "[00:00] первое\n[01:05] второе")
    }
}

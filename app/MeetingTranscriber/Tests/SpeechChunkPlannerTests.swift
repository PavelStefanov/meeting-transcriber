@testable import MeetingTranscriber
import XCTest

/// Pins the chunk split a chunk-batch decode depends on.
///
/// The planner is the whole reason the chunked path can set `no_timestamps`:
/// every timestamp in the transcript comes from a chunk boundary, so a
/// mis-planned chunk is a mis-timed utterance and nothing downstream can
/// notice.
final class SpeechChunkPlannerTests: XCTestCase {
    func test_plan_keepsARegionThatAlreadyFits() {
        let regions = [SpeechRegion(start: 4, end: 12)]
        let chunks = SpeechChunkPlanner.plan(regions: regions)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].start, 4)
        XCTAssertEqual(chunks[0].end, 12)
    }

    func test_plan_preservesRegionBoundaries() {
        // Two regions with a pause between them must stay two chunks: the pause
        // is where speech stopped, and bridging it hands the decoder exactly
        // the mid-sentence boundary this path exists to avoid.
        let regions = [SpeechRegion(start: 0, end: 5), SpeechRegion(start: 20, end: 24)]
        let chunks = SpeechChunkPlanner.plan(regions: regions)
        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[1].start, 20)
    }

    // MARK: - Merging short neighbours

    func test_plan_growsShortTurnsSeparatedByAShortPause() {
        // Four 2 s turns 0.3 s apart: decoded alone each is too short to
        // disambiguate against, which measured as WER 0.500 vs 0.179 for the
        // whole file. Merged they become one chunk with real context.
        let regions = (0 ..< 4).map { index in
            SpeechRegion(start: Double(index) * 2.3, end: Double(index) * 2.3 + 2)
        }
        let chunks = SpeechChunkPlanner.plan(regions: regions)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].start, 0)
        XCTAssertEqual(chunks[0].end, 8.9, accuracy: 0.001)
    }

    func test_plan_stopsGrowingOnceTheChunkHasEnoughContext() {
        // The first region already exceeds the target, so the second must not
        // be absorbed: growth exists to rescue short chunks, not to build long
        // ones.
        let regions = [SpeechRegion(start: 0, end: 9), SpeechRegion(start: 9.2, end: 11)]
        let chunks = SpeechChunkPlanner.plan(regions: regions)
        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].end, 9)
    }

    func test_plan_refusesToBridgeALongPause() {
        // A 5 s pause is a turn boundary, not a breath. Crossing it would pack
        // dead air into the decode and return one segment spanning two turns,
        // which attributes speech to the wrong speaker.
        let regions = [SpeechRegion(start: 0, end: 2), SpeechRegion(start: 7, end: 9)]
        let chunks = SpeechChunkPlanner.plan(regions: regions)
        XCTAssertEqual(chunks.count, 2, "a 5 s gap must stay a chunk boundary")
    }

    func test_plan_mergingNeverExceedsADecodeWindow() {
        // Twenty 2 s turns 0.3 s apart: merging must stop at the window even
        // though every gap is bridgeable.
        let regions = (0 ..< 20).map { index in
            SpeechRegion(start: Double(index) * 2.3, end: Double(index) * 2.3 + 2)
        }
        for chunk in SpeechChunkPlanner.plan(regions: regions) {
            XCTAssertLessThanOrEqual(chunk.duration, SpeechChunkPlanner.maxChunkSeconds)
        }
    }

    func test_plan_neverExceedsTheWhisperWindow() {
        // 70 s of continuous speech: whisper's window is 30 s, so a single
        // decode is impossible and the split must keep every part under it.
        let chunks = SpeechChunkPlanner.plan(regions: [SpeechRegion(start: 0, end: 70)])
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(
                chunk.duration, SpeechChunkPlanner.maxChunkSeconds,
                "chunk \(chunk.start)-\(chunk.end) exceeds one decode window",
            )
        }
    }

    func test_plan_splitsEvenlyRatherThanLeavingAShortTail() {
        // A greedy 30 s slice of a 31 s region leaves a 1 s tail, and a 1 s
        // fragment is where Whisper hallucinates filler. Equal parts instead.
        let chunks = SpeechChunkPlanner.plan(regions: [SpeechRegion(start: 0, end: 31)])
        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].duration, 15.5, accuracy: 0.001)
        XCTAssertEqual(chunks[1].duration, 15.5, accuracy: 0.001)
    }

    func test_plan_coversTheRegionWithoutGapsOrOverlap() {
        let region = SpeechRegion(start: 10, end: 100)
        let chunks = SpeechChunkPlanner.plan(regions: [region])
        XCTAssertEqual(chunks.first?.start, 10)
        XCTAssertEqual(chunks.last?.end, 100, "a sliver of audio would go undecoded")
        for (previous, next) in zip(chunks, chunks.dropFirst()) {
            XCTAssertEqual(
                previous.end, next.start, accuracy: 0.001,
                "chunks must tile the region — a gap drops audio, an overlap decodes it twice",
            )
        }
    }

    func test_plan_dropsARegionTooShortToDecode() {
        // Near-silent input is what Whisper invents text for, so a click that
        // VAD let through must not reach the decoder.
        let chunks = SpeechChunkPlanner.plan(regions: [SpeechRegion(start: 1, end: 1.05)])
        XCTAssertTrue(chunks.isEmpty)
    }

    func test_plan_onNoRegionsYieldsNoChunks() {
        XCTAssertTrue(SpeechChunkPlanner.plan(regions: []).isEmpty)
    }

    // MARK: - The length gate

    func test_shouldChunk_isFalseInsideOneDecodeWindow() {
        // A file this short is decoded in a single pass with the whole thing as
        // context, so there is no window advance to mispredict and nothing for
        // chunking to fix — measured WER 0.179 whole-file against 0.357 chunked.
        XCTAssertFalse(SpeechChunkPlanner.shouldChunk(duration: 17))
        XCTAssertFalse(SpeechChunkPlanner.shouldChunk(duration: 29.8))
    }

    func test_shouldChunk_isFalseExactlyAtTheWindow() {
        XCTAssertFalse(
            SpeechChunkPlanner.shouldChunk(duration: SpeechChunkPlanner.maxChunkSeconds),
            "a recording that exactly fills one window still needs only one decode",
        )
    }

    func test_shouldChunk_isTrueBeyondOneWindow() {
        // Past the window the decoder starts advancing by timestamp token on a
        // single decoding path, which is where the repetition comes from.
        XCTAssertTrue(SpeechChunkPlanner.shouldChunk(duration: 93.8))
        XCTAssertTrue(SpeechChunkPlanner.shouldChunk(duration: 1785))
    }

    // MARK: - sampleRange

    func test_sampleRange_mapsSecondsOntoTheBuffer() {
        let range = SpeechChunkPlanner.sampleRange(
            for: SpeechRegion(start: 1, end: 2), sampleRate: 16000, sampleCount: 48000,
        )
        XCTAssertEqual(range, 16000 ..< 32000)
    }

    func test_sampleRange_clampsToTheBufferEnd() {
        // VAD runs on its own decode of the file, so a region can end a few
        // samples past what the engine's decode produced. Clamped, not fatal.
        let range = SpeechChunkPlanner.sampleRange(
            for: SpeechRegion(start: 1, end: 5), sampleRate: 16000, sampleCount: 40000,
        )
        XCTAssertEqual(range, 16000 ..< 40000)
    }

    func test_sampleRange_isNilWhenItStartsPastTheBuffer() {
        XCTAssertNil(SpeechChunkPlanner.sampleRange(
            for: SpeechRegion(start: 10, end: 11), sampleRate: 16000, sampleCount: 8000,
        ))
    }

    func test_sampleRange_isNilForAnEmptyRange() {
        XCTAssertNil(SpeechChunkPlanner.sampleRange(
            for: SpeechRegion(start: 2, end: 2), sampleRate: 16000, sampleCount: 48000,
        ))
    }
}

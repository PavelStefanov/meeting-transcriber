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

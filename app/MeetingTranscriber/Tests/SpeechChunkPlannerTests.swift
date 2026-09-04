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

    // MARK: - Moving a split off a word

    /// 16 kHz buffer of steady tone with a short near-silent gap centred on
    /// `gapAt`, i.e. the shape of two words with a breath between them.
    private func samples(seconds: Double, gapAt: Double, gapWidth: Double = 0.06) -> [Float] {
        let rate = 16000
        var out = [Float](repeating: 0, count: Int(seconds * Double(rate)))
        for index in out.indices {
            let t = Double(index) / Double(rate)
            out[index] = abs(t - gapAt) < gapWidth / 2 ? 0.0001 : 0.5
        }
        return out
    }

    func test_snapSplits_movesACutOntoTheNearbyGap() {
        // Even division would cut at 10 s, mid-word; the gap sits at 10.4 s.
        let chunks = [SpeechRegion(start: 0, end: 10), SpeechRegion(start: 10, end: 20)]
        let moved = SpeechChunkPlanner.snapSplitsToQuietMoments(
            chunks, samples: samples(seconds: 20, gapAt: 10.4), sampleRate: 16000,
        )
        XCTAssertEqual(moved.count, 2)
        XCTAssertEqual(moved[0].end, 10.4, accuracy: 0.05, "the cut should land in the gap")
        XCTAssertEqual(moved[1].start, moved[0].end, "the two chunks must stay contiguous")
        XCTAssertEqual(moved[1].end, 20, "the outer bounds are region boundaries and must not move")
    }

    func test_snapSplits_leavesACutAloneWhenNoGapIsInReach() {
        // Gap is 5 s away, far outside the search window.
        let chunks = [SpeechRegion(start: 0, end: 10), SpeechRegion(start: 10, end: 20)]
        let moved = SpeechChunkPlanner.snapSplitsToQuietMoments(
            chunks, samples: samples(seconds: 20, gapAt: 15), sampleRate: 16000,
        )
        XCTAssertEqual(moved[0].end, 10, accuracy: SpeechChunkPlanner.splitSearchSeconds)
    }

    func test_snapSplits_neverMovesARealPauseBetweenRegions() {
        // These chunks do not touch: the space between them is silence VAD
        // already found, and dragging that boundary would pull speech across it.
        let chunks = [SpeechRegion(start: 0, end: 8), SpeechRegion(start: 12, end: 20)]
        let moved = SpeechChunkPlanner.snapSplitsToQuietMoments(
            chunks, samples: samples(seconds: 20, gapAt: 8.3), sampleRate: 16000,
        )
        XCTAssertEqual(moved[0].end, 8)
        XCTAssertEqual(moved[1].start, 12)
    }

    func test_snapSplits_keepsBothSidesInsideADecodeWindow() {
        // 58 s region split evenly gives 29 s halves; a search that moved the
        // cut freely could push one side past the 30 s window.
        let chunks = SpeechChunkPlanner.plan(regions: [SpeechRegion(start: 0, end: 58)])
        let moved = SpeechChunkPlanner.snapSplitsToQuietMoments(
            chunks, samples: samples(seconds: 58, gapAt: 29.7), sampleRate: 16000,
        )
        for chunk in moved {
            XCTAssertLessThanOrEqual(chunk.duration, SpeechChunkPlanner.maxChunkSeconds)
        }
    }

    func test_snapSplits_onASingleChunkChangesNothing() {
        let chunks = [SpeechRegion(start: 0, end: 10)]
        let moved = SpeechChunkPlanner.snapSplitsToQuietMoments(
            chunks, samples: samples(seconds: 10, gapAt: 5), sampleRate: 16000,
        )
        XCTAssertEqual(moved.count, 1)
        XCTAssertEqual(moved[0].start, 0)
        XCTAssertEqual(moved[0].end, 10)
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

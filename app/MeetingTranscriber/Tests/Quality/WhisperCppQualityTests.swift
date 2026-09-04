@testable import MeetingTranscriber
import XCTest

/// Production-model whisper.cpp quality tests. Skipped by default — gated by
/// `RUN_QUALITY_TESTS=1` like the sibling engine suites, and more so here: this
/// engine pulls a 2.9 GB model.
///
/// Measures BOTH decode paths on purpose, because they are different
/// experiments and only one of them is what the app now runs:
///
///   * whole file — `transcribeSegments`, the path a caller gets with VAD off;
///   * chunked — `transcribeChunks` over VAD regions, the path `PipelineQueue`
///     takes when VAD is on.
///
/// Keeping only the first would baseline behaviour the pipeline no longer uses;
/// keeping only the second would lose the comparison that justifies chunking.
/// The two rows carry distinct engine labels so `quality-baseline.json` can
/// hold a bound for each.
@MainActor
final class WhisperCppQualityTests: XCTestCase {
    /// The engine runs one pinned model, so there is no variant to read from
    /// the environment the way `WHISPERKIT_MODEL` is read. Recorded anyway, so
    /// a results row says which weights produced it.
    private var modelVariant: String {
        WhisperCppModel.filename
    }

    // MARK: - Whole-file decode

    func test_whisperCpp_twoSpeakers_de_wer() async throws {
        try skipUnlessQualityRun()
        let engine = await loadedEngine(language: "de")
        try await runWERAgainstFixture(
            named: "two_speakers_de",
            engine: engine,
            engineLabel: "whisperCpp",
            modelVariant: modelVariant,
            threshold: 0.5,
        )
    }

    func test_whisperCpp_threeSpeakers_de_wer() async throws {
        try skipUnlessQualityRun()
        let engine = await loadedEngine(language: "de")
        try await runWERAgainstFixture(
            named: "three_speakers_de",
            engine: engine,
            engineLabel: "whisperCpp",
            modelVariant: modelVariant,
            threshold: 0.5,
        )
    }

    /// Real recorded meeting with heavy overlap — the fixture that separates a
    /// synthetic result from a usable one.
    func test_whisperCpp_fourSpeakers_en_real_wer() async throws {
        try skipUnlessQualityRun()
        let engine = await loadedEngine(language: "en")
        try await runWERAgainstFixture(
            named: "four_speakers_en_ami",
            engine: engine,
            engineLabel: "whisperCpp",
            modelVariant: modelVariant,
            threshold: 0.5,
        )
    }

    // MARK: - Chunked decode (the path the pipeline uses with VAD on)

    func test_whisperCppChunked_twoSpeakers_de_wer() async throws {
        try skipUnlessQualityRun()
        try await runChunkedFixture(named: "two_speakers_de", language: "de")
    }

    func test_whisperCppChunked_threeSpeakers_de_wer() async throws {
        try skipUnlessQualityRun()
        try await runChunkedFixture(named: "three_speakers_de", language: "de")
    }

    func test_whisperCppChunked_fourSpeakers_en_real_wer() async throws {
        try skipUnlessQualityRun()
        try await runChunkedFixture(named: "four_speakers_en_ami", language: "en")
    }

    // MARK: - Helpers

    private func loadedEngine(language: String) async -> WhisperCppEngine {
        let engine = WhisperCppEngine()
        engine.language = language
        await engine.loadModel()
        XCTAssertEqual(
            engine.modelState, .loaded,
            "whisper.cpp model failed to load — the 2.9 GB download may be missing or truncated",
        )
        return engine
    }

    /// The chunked equivalent of `runWERAgainstFixture`.
    ///
    /// Not folded into that shared helper because it needs a second input the
    /// others have no notion of — the VAD regions — and because the helper's
    /// contract is deliberately "an already-loaded `any TranscribingEngine`",
    /// which is exactly the abstraction that cannot express a chunked decode.
    /// Same measurement, same writer, same soft threshold.
    private func runChunkedFixture(named fixture: String, language: String) async throws {
        let truth = try GroundTruth.load(named: fixture)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: truth.audioURL.path),
            "Audio fixture missing: \(truth.audioURL.path)",
        )
        // Only measure what the pipeline would actually do. Below one decode
        // window it takes the whole-file path, and baselining a chunked number
        // for a fixture that never gets chunked would gate on behaviour nothing
        // ships — the short fixtures measured worse chunked precisely because
        // they fit in one window (`SpeechChunkPlanner.shouldChunk`).
        try XCTSkipUnless(
            SpeechChunkPlanner.shouldChunk(duration: truth.duration),
            "\(fixture) fits in one decode window — the pipeline decodes it whole",
        )
        let engine = await loadedEngine(language: language)

        // Regions come from the same VAD the pipeline uses, at its default
        // threshold, so the measurement is of the shipped chain rather than of
        // a hand-picked split.
        let vad = FluidVAD(threshold: 0.5)
        let (samples, _) = try await AudioMixer.loadAudioAsFloat32(url: truth.audioURL)
        let map = try await vad.detectSpeech(samples: samples)
        let chunks = SpeechChunkPlanner.plan(regions: map.segments)
        try XCTSkipIf(chunks.isEmpty, "VAD found no speech in \(fixture) — nothing to decode")

        let started = Date()
        let segments = try await engine.transcribeChunks(audioPath: truth.audioURL, chunks: chunks)
        let hypothesis = segments.map(\.text).joined(separator: " ")
        let breakdown = WERCalculator.werBreakdown(reference: truth.text, hypothesis: hypothesis)
        let elapsed = Date().timeIntervalSince(started)

        QualityResultsWriter.shared.append(
            QualityResult(
                engine: "whisperCpp.chunked",
                fixture: fixture,
                modelVariant: modelVariant,
                wer: breakdown.wer,
                der: nil,
                werBreakdown: .init(breakdown),
                derBreakdown: nil,
                appVersion: qualityAppVersion,
                timestamp: ISO8601DateFormatter().string(from: started),
                durationSeconds: elapsed,
            ),
        )
        _ = try? QualityResultsWriter.shared.flush()

        // Same soft bound as the other engines: it catches catastrophic
        // breakage (model not loaded, audio not decoded, chunks mis-planned)
        // without pinning a number that a model bump would move.
        XCTAssertLessThan(
            breakdown.wer,
            0.5,
            "whisperCpp.chunked WER too high on \(fixture): \(breakdown.wer) — "
                + "\(chunks.count) chunks, hypothesis was: \(hypothesis)",
        )
    }
}

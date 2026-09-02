@testable import MeetingTranscriber
import XCTest

/// Pins the decode parameters and the language mapping.
///
/// This is a parity test, not a taste test. Each value below is what Meetily's
/// `transcribe_audio_with_confidence` passes to `whisper_full` on an Apple
/// Silicon machine, and the engine exists to reproduce the transcription
/// quality that configuration produced on real Russian meeting audio. Changing
/// one is allowed — but it makes the comparison a different experiment, and
/// this test is where that has to be acknowledged.
final class WhisperCppDecodingConfigTests: XCTestCase {
    // MARK: - Meetily parity

    /// Beam 3 / temperature 0.2 is Meetily's `PerformanceTier::High` branch,
    /// which is the tier an Apple Silicon Mac actually resolves to: the tier
    /// test is `memory_gb >= 16 && cpu_cores >= 8`, and Meetily's
    /// `detect_memory_gb()` returns a hard-coded 8 unless `MEMORY_GB` is set in
    /// the environment, which nothing sets. So a 16 GB M2 Pro is High, not
    /// Ultra (beam 5 / temperature 0.1).
    func test_meetilyParity_samplingMatchesTheHighTierBranch() {
        let config = WhisperCppDecodingConfig.meetilyParity
        XCTAssertEqual(config.beamSize, 3)
        XCTAssertEqual(config.patience, 1.0)
        XCTAssertEqual(config.temperature, 0.2)
    }

    func test_meetilyParity_fallbackThresholdsMatch() {
        let config = WhisperCppDecodingConfig.meetilyParity
        XCTAssertEqual(config.temperatureIncrement, 0.2)
        XCTAssertEqual(config.entropyThreshold, 2.4)
        XCTAssertEqual(config.logProbabilityThreshold, -1.0)
        XCTAssertEqual(
            config.noSpeechThreshold, 0.55,
            "Meetily lowered whisper.cpp's 0.6 to keep quiet speech",
        )
        XCTAssertEqual(config.maxInitialTimestamp, 1.0)
    }

    /// `suppressNonSpeechTokens` is the one value here that is NOT a
    /// whisper.cpp default — it ships false — and it is the most audible
    /// difference from a stock run on meeting audio, so it gets its own
    /// assertion.
    func test_meetilyParity_suppressesNonSpeechTokens() {
        XCTAssertTrue(WhisperCppDecodingConfig.meetilyParity.suppressNonSpeechTokens)
        XCTAssertTrue(WhisperCppDecodingConfig.meetilyParity.suppressBlank)
    }

    func test_meetilyParity_segmentationKeepsTokenTimestampsAndTheLengthCap() {
        let config = WhisperCppDecodingConfig.meetilyParity
        XCTAssertTrue(
            config.tokenTimestamps,
            "whisper_wrap_segment only applies max_len when token timestamps are on",
        )
        XCTAssertEqual(config.maxSegmentLength, 200)
        XCTAssertFalse(config.singleSegment)
    }

    /// The one deliberate divergence from Meetily, asserted so it cannot be
    /// "fixed" into parity by someone reading the parity comment alone.
    ///
    /// Meetily sets `no_timestamps = true` and takes its timing from its own
    /// VAD chunk offsets. This pipeline hands over whole recordings and needs
    /// per-utterance timestamps for diarization; with the flag on, whisper.cpp
    /// suppresses every timestamp token and advances a full 30 s window per
    /// decode, so a recording would come back as one segment per half minute.
    func test_meetilyParity_keepsTimestampsDespiteMeetilyDisablingThem() {
        XCTAssertFalse(
            WhisperCppDecodingConfig.meetilyParity.noTimestamps,
            "diarization, the dual-track merge and the echo dedup all read segment times",
        )
    }

    /// whisper.cpp's own default, which Meetily does not change — and the
    /// opposite of the OpenAI reference implementation's
    /// `condition_on_previous_text=True`.
    func test_meetilyParity_doesNotConditionOnPreviousText() {
        XCTAssertTrue(WhisperCppDecodingConfig.meetilyParity.noContext)
    }

    /// whisper.cpp's `min(4, cores)` default. Meetily computes a thread count
    /// from its hardware profile and then never passes it to whisper, so four
    /// is what it effectively runs with. Reproduced rather than raised because
    /// ggml's reductions are order-dependent, so a different thread count is a
    /// different numerical result.
    func test_meetilyParity_keepsWhisperCppsDefaultThreadCount() {
        XCTAssertEqual(WhisperCppDecodingConfig.meetilyParity.threadCount, 4)
    }

    // MARK: - Language mapping

    func test_languageCode_passesRussianThroughUnchanged() {
        XCTAssertEqual(WhisperCppLanguage.code(for: "ru"), "ru")
    }

    func test_languageCode_treatsTheAppsSentinelsAsAutoDetect() {
        XCTAssertNil(WhisperCppLanguage.code(for: ""))
        XCTAssertNil(WhisperCppLanguage.code(for: nil))
        XCTAssertNil(WhisperCppLanguage.code(for: "auto"))
        XCTAssertNil(WhisperCppLanguage.code(for: "   "))
    }

    func test_languageCode_normalisesCaseAndSurroundingSpace() {
        XCTAssertEqual(WhisperCppLanguage.code(for: " DE "), "de")
    }

    /// `fil` is in the app's Whisper language picker because WhisperKit accepts
    /// it. Whisper's own vocabulary has Tagalog only as `tl`, and an unknown
    /// code does not fail in whisper.cpp — it builds the prompt with the
    /// start-of-transcript token in the language slot and decodes anyway.
    func test_languageCode_rewritesTheOnePickerCodeWhisperSpellsDifferently() {
        XCTAssertEqual(WhisperCppLanguage.code(for: "fil"), "tl")
    }

    /// Every entry the shared picker offers must map to something, and only
    /// `fil` may be rewritten — otherwise the two engines would disagree about
    /// what the user selected.
    func test_languageCode_coversEveryEntryInTheSharedWhisperPicker() {
        for entry in PickerLanguages.whisperKit {
            let mapped = WhisperCppLanguage.code(for: entry.code)
            if entry.code.isEmpty {
                XCTAssertNil(mapped, "the auto-detect sentinel must stay auto-detect")
                continue
            }
            let expected = WhisperCppLanguage.aliases[entry.code] ?? entry.code
            XCTAssertEqual(mapped, expected, "unexpected mapping for \(entry.code)")
        }
    }
}

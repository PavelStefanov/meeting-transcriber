@testable import MeetingTranscriber
import XCTest

/// The seams that make the new engine reachable and correctly configured,
/// without loading a 2.9 GB model: the settings enum, the language it inherits
/// from the shared Whisper picker, and the controller's active-engine
/// selection. What the automation API reports is asserted in
/// `RPCEngineStateTests`, which is already inside the `#if !APPSTORE` the RPC
/// snapshot lives behind.
///
/// Everything asserted here is a place where adding a third engine could have
/// gone silently wrong — an engine present in the picker but never selected by
/// the controller, or selected but left on the wrong language.
@MainActor
final class WhisperCppEngineWiringTests: XCTestCase {
    // swiftlint:disable:next implicitly_unwrapped_optional
    private var defaults: UserDefaults!
    // swiftlint:disable:next implicitly_unwrapped_optional
    private var settings: AppSettings!
    // swiftlint:disable:next implicitly_unwrapped_optional
    private var suiteName: String!

    override func setUp() async throws {
        try await super.setUp()
        suiteName = "WhisperCppEngineWiringTests-\(getpid())-\(UUID().uuidString)"
        guard let suite = UserDefaults(suiteName: suiteName) else {
            XCTFail("could not create test UserDefaults suite")
            return
        }
        defaults = suite
        settings = AppSettings(defaults: defaults)
    }

    override func tearDown() async throws {
        settings = nil
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
        suiteName = nil
        try await super.tearDown()
    }

    // MARK: - The picker entry

    func test_engineSetting_appearsInThePickerAsTheLastEntry() {
        XCTAssertEqual(
            TranscriptionEngineSetting.availableCases.last, .whisperCpp,
            "appended so the existing picker order is unchanged",
        )
    }

    func test_engineSetting_labelNamesTheModelAndItsSize() {
        let label = TranscriptionEngineSetting.whisperCpp.label
        XCTAssertTrue(label.contains("Large V3"), label)
        XCTAssertTrue(label.contains("whisper.cpp"), label)
        XCTAssertTrue(
            label.contains("2.9 GB"),
            "the size in the label is what makes the download an informed choice instead of a dialog: \(label)",
        )
    }

    /// The raw value is persisted in `UserDefaults`, so renaming the case
    /// without a migration would reset every user who selected this engine
    /// back to WhisperKit.
    func test_engineSetting_rawValueIsTheStoredContract() {
        XCTAssertEqual(TranscriptionEngineSetting.whisperCpp.rawValue, "whisperCpp")
        XCTAssertEqual(TranscriptionEngineSetting(rawValue: "whisperCpp"), .whisperCpp)
    }

    func test_engineSetting_persistsAcrossAFreshAppSettings() {
        settings.transcriptionEngine = .whisperCpp
        XCTAssertEqual(AppSettings(defaults: defaults).transcriptionEngine, .whisperCpp)
    }

    // MARK: - Live captions

    /// Batch-only: no `transcribeSamples` hook. This is not a regression — with
    /// a language selected, `LiveCaptionsGate` routes captions to a streaming
    /// backend that drives its own model. Only auto-detect loses them.
    func test_engineSetting_declaresItselfUnsuitableForTheReTranscribePath() {
        XCTAssertFalse(TranscriptionEngineSetting.whisperCpp.supportsLiveTranscription)
    }

    func test_captionsGate_stillRoutesAnExplicitLanguageToAStreamingBackend() {
        XCTAssertEqual(
            LiveCaptionsGate.strategy(
                liveEnabled: true, engineLanguage: "ru",
                engineSupportsLive: TranscriptionEngineSetting.whisperCpp.supportsLiveTranscription,
            ),
            .nemotronStreaming,
        )
    }

    func test_captionsGate_turnsCaptionsOffOnAutoDetect() {
        XCTAssertEqual(
            LiveCaptionsGate.strategy(
                liveEnabled: true, engineLanguage: nil,
                engineSupportsLive: TranscriptionEngineSetting.whisperCpp.supportsLiveTranscription,
            ),
            .none,
        )
    }

    // MARK: - Language, shared with WhisperKit

    func test_engineSetting_readsTheWhisperLanguageSetting() {
        XCTAssertTrue(TranscriptionEngineSetting.whisperCpp.usesWhisperLanguage)
        XCTAssertTrue(TranscriptionEngineSetting.whisperKit.usesWhisperLanguage)
        XCTAssertFalse(TranscriptionEngineSetting.parakeet.usesWhisperLanguage)
    }

    func test_activeEngineLanguage_followsTheWhisperPickerWhenWhisperCppIsActive() {
        settings.transcriptionEngine = .whisperCpp
        settings.whisperLanguage = "ru"
        XCTAssertEqual(settings.activeEngineLanguageOrNil, "ru")
        settings.whisperLanguage = ""
        XCTAssertNil(settings.activeEngineLanguageOrNil)
    }

    // MARK: - EngineController

    func test_controller_selectsTheWhisperCppEngineWhenItIsActive() {
        settings.transcriptionEngine = .whisperCpp
        let engines = EngineController(settings: settings)
        XCTAssertIdentical(
            engines.activeTranscriptionEngine, engines.whisperCpp,
            "the pipeline transcribes through activeTranscriptionEngine and nothing else",
        )
    }

    /// The Russian case from the task this engine was added for: the existing
    /// picker's selection has to reach whisper.cpp as `ru`.
    func test_controller_pushesRussianOntoTheEngineAtInit() {
        settings.transcriptionEngine = .whisperCpp
        settings.whisperLanguage = "ru"
        let engines = EngineController(settings: settings)
        XCTAssertEqual(engines.whisperCpp.language, "ru")
        XCTAssertEqual(WhisperCppLanguage.code(for: engines.whisperCpp.language), "ru")
    }

    func test_controller_leavesAutoDetectAsNil() {
        settings.transcriptionEngine = .whisperCpp
        settings.whisperLanguage = ""
        let engines = EngineController(settings: settings)
        XCTAssertNil(engines.whisperCpp.language)
    }

    func test_controller_doesNotTouchTheOtherEnginesWhenWhisperCppIsActive() {
        settings.transcriptionEngine = .whisperCpp
        settings.whisperLanguage = "ru"
        settings.setCustomVocabularyPath("/tmp/wiring-vocab.txt")
        let engines = EngineController(settings: settings)
        XCTAssertNil(engines.whisperKit.language, "WhisperKit is inactive — no sync expected")
        XCTAssertEqual(engines.parakeetEngine.customVocabularyPath, "")
    }

    // MARK: - Engine defaults

    func test_engine_startsUnloadedWithTheParityDecodeConfig() {
        let engine = WhisperCppEngine()
        XCTAssertEqual(engine.modelState, .unloaded)
        XCTAssertEqual(engine.downloadProgress, 0)
        XCTAssertEqual(engine.decodingConfig, .meetilyParity)
    }

    /// The engine has to report timestamps, or `PipelineQueue` skips
    /// diarization with a warning and collapses the meeting onto one speaker.
    func test_engine_declaresThatItProvidesTimestamps() {
        XCTAssertTrue(WhisperCppEngine().providesTimestamps)
    }
}

import Foundation

// SwiftFormat strips redundant raw values matching their case name, which then
// trips SwiftLint's `raw_value_for_camel_cased_codable_enum`; the rawValues
// already match (the rule is a stability hint, not a behavioral requirement),
// so disable the lint here rather than fight the formatter.
enum TranscriptionEngineSetting: String, CaseIterable, Codable {
    // swiftlint:disable:next raw_value_for_camel_cased_codable_enum
    case whisperKit
    case parakeet
    /// Full-precision Whisper large-v3 (f16) through whisper.cpp on Metal.
    /// Appended last so the existing picker order is untouched and this is
    /// simply one more entry at the bottom of it.
    // swiftlint:disable:next raw_value_for_camel_cased_codable_enum
    case whisperCpp

    var label: String {
        switch self {
        case .whisperKit: "WhisperKit (Whisper)"
        case .parakeet: "Parakeet TDT v3 (NVIDIA)"
        // The size is in the label on purpose: choosing this entry starts a
        // 2.9 GB download, and an informed pick is what lets the model reuse
        // the existing progress display instead of needing a consent dialog of
        // its own.
        case .whisperCpp: "Whisper Large V3 Full (whisper.cpp, 2.9 GB)"
        }
    }

    /// Whether this engine is available on the current platform. All three
    /// current engines run everywhere the app does; kept as a capability hook
    /// for engines with stricter OS floors.
    var isAvailable: Bool {
        switch self {
        case .whisperKit, .parakeet, .whisperCpp: true
        }
    }

    /// Cases available on the current platform. Used in UI pickers instead of allCases.
    static var availableCases: [Self] {
        allCases.filter(\.isAvailable)
    }

    /// Whether this engine reads `AppSettings.whisperLanguage` instead of a
    /// language setting of its own. Both Whisper backends do: same model
    /// family, same ISO 639-1 codes, one picker in Settings and one stored
    /// value. Expressed as a property rather than as two `==` comparisons at
    /// the call site so a fourth engine cannot be added without answering the
    /// question.
    var usesWhisperLanguage: Bool {
        switch self {
        case .whisperKit, .whisperCpp: true
        case .parakeet: false
        }
    }

    /// Whether the engine implements `transcribeSamples([Float])` so the
    /// live-transcription pipeline can feed it VAD-bounded windows.
    var supportsLiveTranscription: Bool {
        switch self {
        case .whisperKit, .parakeet: true
        // `WhisperCppEngine` is batch-only and does not conform to
        // `StreamingTranscribingEngine`. This only costs captions on
        // auto-detect: with a language explicitly set, `LiveCaptionsGate`
        // routes them to a streaming backend that drives its own model and
        // never touches the active engine.
        case .whisperCpp: false
        }
    }
}

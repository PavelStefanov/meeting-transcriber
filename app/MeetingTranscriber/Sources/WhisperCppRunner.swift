import Foundation
import os.log
import whisper

private let logger = Logger(subsystem: AppPaths.logSubsystem, category: "WhisperCppRunner")

/// Everything in this app that touches the whisper.cpp C API.
///
/// ## Why a serial queue and not an actor
///
/// A `whisper_context` is explicitly not thread safe ("Not thread safe for same
/// context" — whisper.h), and both calls into it block for a long time: loading
/// the full large-v3 weights takes seconds, decoding a meeting takes minutes.
/// An actor would suspend at every `await` inside those methods and let a
/// second call reenter the same context, so the serialization has to be the
/// thing that actually runs the C code. A dedicated serial `DispatchQueue` is
/// that thing: no work item can start before the previous one returns,
/// reentrancy is impossible by construction, and no cooperative-pool thread is
/// blocked for the duration.
///
/// `@unchecked Sendable` for the same reason it is usually wrong and here is
/// not: the two mutable fields are only ever read or written inside a work item
/// on `queue`, which is serial, so there is exactly one accessor at a time. The
/// invariant is that no method touches them outside an `onQueue` body.
final class WhisperCppRunner: @unchecked Sendable {
    /// `.userInitiated` and not `.utility`, unlike the echo-detector pass that
    /// uses `Task.detached(priority: .utility)`. ggml's worker threads inherit
    /// the QoS of the thread that starts them, and on Apple Silicon `.utility`
    /// parks them on the efficiency cores — which for a 2.9 GB model is not a
    /// politeness, it is a several-fold slowdown of a job the user is waiting
    /// for the result of.
    private let queue = DispatchQueue(
        label: "com.meetingtranscriber.whispercpp", qos: .userInitiated,
    )

    /// Only ever touched inside an `onQueue` body. See the type's discussion.
    private var context: OpaquePointer?
    private var loadedModelPath: String?

    deinit {
        // Safe off the queue: `deinit` runs when the last reference is gone, so
        // no work item can be in flight or be started.
        if let context { whisper_free(context) }
    }

    /// Load `modelPath`, or return immediately if that exact path is already
    /// loaded. Replaces a context loaded from a different path.
    func load(modelPath: String) async throws {
        try await onQueue { try self.loadOnQueue(modelPath) }
    }

    /// Free the context and its ~3 GB of weights. Idempotent.
    func unload() async {
        try? await onQueue { self.unloadOnQueue() }
    }

    /// Decode 16 kHz mono samples.
    ///
    /// - Parameters:
    ///   - samples: 16 kHz mono Float32 PCM, the rate whisper.cpp requires
    ///     (`WHISPER_SAMPLE_RATE`). The caller resamples; this does not.
    ///   - language: canonical whisper.cpp language code, or nil to let
    ///     whisper.cpp detect it.
    ///   - config: the decode parameters, normally
    ///     `WhisperCppDecodingConfig.meetilyParity`.
    func transcribe(
        samples: [Float],
        language: String?,
        config: WhisperCppDecodingConfig,
    ) async throws -> [WhisperCppSegmentBuilder.RawSegment] {
        try await onQueue { try self.transcribeOnQueue(samples: samples, language: language, config: config) }
    }

    /// Canonical language code if whisper.cpp knows it, else nil.
    ///
    /// An unknown code must never reach `whisper_full`. It does not fail there:
    /// `whisper_lang_id` answers -1 and the prompt is then built with
    /// `whisper_token_lang(ctx, -1)`, which is the start-of-transcript token
    /// rather than a language token, and the decode proceeds with a corrupt
    /// prompt and no diagnostic. Falling back to auto-detect is a worse
    /// transcript than the right code and a much better one than that.
    ///
    /// Static and synchronous: `whisper_lang_id` is a lookup in a read-only
    /// table and needs neither a context nor the queue.
    static func supportedLanguage(_ code: String?) -> String? {
        guard let code else { return nil }
        let known = code.withCString { whisper_lang_id($0) >= 0 }
        guard known else {
            logger.warning("whisper.cpp does not know language '\(code, privacy: .public)' — falling back to auto-detect")
            return nil
        }
        return code
    }

    // MARK: - Queue bridge

    private func onQueue<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(with: Result { try work() })
            }
        }
    }

    // MARK: - On-queue bodies

    private func loadOnQueue(_ modelPath: String) throws {
        if context != nil, loadedModelPath == modelPath { return }
        unloadOnQueue()
        Self.redirectWhisperLogging()

        var params = whisper_context_default_params()
        // Metal, with flash attention, matching what Meetily resolves to on an
        // Apple Silicon machine (`whisper_context_acceleration_for` with a
        // Metal backend on its High/Ultra tiers). `gpu_device` is a CUDA index
        // and is ignored here; set to Meetily's value anyway so the struct is
        // identical.
        params.use_gpu = true
        params.flash_attn = true
        params.gpu_device = 0

        guard let created = whisper_init_from_file_with_params(modelPath, params) else {
            throw WhisperCppEngineError.contextInitFailed
        }
        context = created
        loadedModelPath = modelPath
        logger.info("whisper.cpp context loaded from \(modelPath, privacy: .public)")
    }

    private func unloadOnQueue() {
        guard let context else { return }
        whisper_free(context)
        self.context = nil
        loadedModelPath = nil
        logger.info("whisper.cpp context unloaded")
    }

    private func transcribeOnQueue(
        samples: [Float],
        language: String?,
        config: WhisperCppDecodingConfig,
    ) throws -> [WhisperCppSegmentBuilder.RawSegment] {
        guard let context else { throw WhisperCppEngineError.modelNotLoaded }
        guard !samples.isEmpty else { return [] }

        var params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH)
        Self.apply(config, to: &params)

        let status: Int32 = if let language {
            // The C string only has to outlive the call: whisper.cpp reads
            // `params.language` during `whisper_full` and keeps no reference to
            // it afterwards.
            language.withCString { code in
                params.language = code
                return whisper_full(context, params, samples, Int32(samples.count))
            }
        } else {
            // A nil language with `detect_language == false` is what makes
            // whisper.cpp detect the language AND then transcribe. Setting
            // `detect_language = true` makes `whisper_full` detect and
            // `return 0` immediately with zero segments, which is not what
            // "auto" means anywhere in this app.
            whisper_full(context, params, samples, Int32(samples.count))
        }

        guard status == 0 else { throw WhisperCppEngineError.decodeFailed(status: Int(status)) }
        return Self.collectSegments(context)
    }

    // MARK: - C struct mapping

    /// Transcribes `WhisperCppDecodingConfig` onto whisper.cpp's parameter
    /// struct. Every assignment here corresponds to one documented field on
    /// that type; the reasoning for each value lives there, not here.
    ///
    /// Deliberately does NOT touch `greedy.best_of`. Leaving it at the default
    /// of -1 is what reproduces Meetily's effective single-decoder decode — see
    /// `WhisperCppDecodingConfig` for why that is a reproduction and not an
    /// oversight.
    private static func apply(_ config: WhisperCppDecodingConfig, to params: inout whisper_full_params) {
        params.beam_search.beam_size = config.beamSize
        params.beam_search.patience = config.patience
        params.temperature = config.temperature
        params.temperature_inc = config.temperatureIncrement
        params.entropy_thold = config.entropyThreshold
        params.logprob_thold = config.logProbabilityThreshold
        params.no_speech_thold = config.noSpeechThreshold
        params.max_initial_ts = config.maxInitialTimestamp
        params.suppress_blank = config.suppressBlank
        params.suppress_nst = config.suppressNonSpeechTokens
        params.max_len = config.maxSegmentLength
        params.token_timestamps = config.tokenTimestamps
        params.single_segment = config.singleSegment
        params.no_timestamps = config.noTimestamps
        params.no_context = config.noContext
        params.n_threads = config.threadCount
        params.translate = false
        // whisper.cpp writes its own progress and per-segment text to stdout
        // unless every one of these is off, and `print_progress` defaults to
        // ON. In a menu-bar app that output goes nowhere useful.
        params.print_special = false
        params.print_progress = false
        params.print_realtime = false
        params.print_timestamps = false
    }

    private static func collectSegments(_ context: OpaquePointer) -> [WhisperCppSegmentBuilder.RawSegment] {
        let count = whisper_full_n_segments(context)
        guard count > 0 else { return [] }
        var raw: [WhisperCppSegmentBuilder.RawSegment] = []
        raw.reserveCapacity(Int(count))
        for index in 0 ..< count {
            guard let text = whisper_full_get_segment_text(context, index) else { continue }
            raw.append(WhisperCppSegmentBuilder.RawSegment(
                text: String(cString: text),
                startCentiseconds: whisper_full_get_segment_t0(context, index),
                endCentiseconds: whisper_full_get_segment_t1(context, index),
            ))
        }
        return raw
    }

    /// Sends ggml's and whisper.cpp's own logging to `os.log` instead of
    /// stderr, where a bundled app has no console to write to. Installed on
    /// every load rather than once: the call replaces the callback, so it is
    /// idempotent, and a one-shot global would only add a second thing that
    /// can be in the wrong state.
    ///
    /// The closure is `@convention(c)` and captures nothing — `logger` is a
    /// file-scope `let` and `Logger` is `Sendable`.
    private static func redirectWhisperLogging() {
        whisper_log_set({ level, text, _ in
            guard let text else { return }
            let message = String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !message.isEmpty else { return }
            switch level {
            case GGML_LOG_LEVEL_ERROR:
                logger.error("whisper.cpp: \(message, privacy: .public)")

            case GGML_LOG_LEVEL_WARN:
                logger.warning("whisper.cpp: \(message, privacy: .public)")

            default:
                logger.debug("whisper.cpp: \(message, privacy: .public)")
            }
        }, nil)
    }
}

/// Failures specific to the whisper.cpp engine. Separate from
/// `TranscriptionError`, whose `modelNotLoaded` message names WhisperKit.
enum WhisperCppEngineError: LocalizedError, Equatable {
    case modelNotLoaded
    case contextInitFailed
    case decodeFailed(status: Int)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "whisper.cpp model is not loaded"

        case .contextInitFailed:
            "whisper.cpp could not load the model file — it may be truncated or not a GGML model"

        case let .decodeFailed(status):
            "whisper.cpp transcription failed (whisper_full returned \(status))"
        }
    }
}

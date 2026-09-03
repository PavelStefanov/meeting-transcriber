import Foundation
import os.log

private let logger = Logger(subsystem: AppPaths.logSubsystem, category: "WhisperCppEngine")

/// Transcription engine backed by full-precision Whisper large-v3 (f16, ~2.9 GB)
/// running through whisper.cpp on Metal.
///
/// Added as a third `TranscribingEngine` alongside `WhisperKitEngine` and
/// `ParakeetEngine`, not as a replacement for either: it exists to reproduce a
/// specific quality result — measured on real Russian meeting audio, where
/// Meetily's whisper.cpp + f16 large-v3 beat this app's WhisperKit path — while
/// leaving recording, diarization, known voices and speaker recognition
/// untouched. Everything downstream sees the same `[TimestampedSegment]` the
/// other two engines produce.
///
/// It runs one fixed model. There is no variant picker, because a picker is how
/// "full large-v3" quietly becomes turbo or q5_0, which is the exact
/// substitution the engine was added to avoid. The other variants stay
/// available through the WhisperKit engine's existing model list.
///
/// Batch only — it does not conform to `StreamingTranscribingEngine`, so
/// `TranscriptionEngineSetting.whisperCpp.supportsLiveTranscription` is false.
/// Live captions still work when a language is explicitly configured, because
/// `LiveCaptionsGate` routes those to the engine-independent streaming
/// backends; only auto-detect loses them, which is the same trade the gate
/// already makes.
@MainActor
@Observable
final class WhisperCppEngine: TranscribingEngine {
    /// Explicit language code, or nil for auto-detect. Fed from the app's
    /// existing Whisper language setting (`AppSettings.whisperLanguage`) via
    /// `EngineController` — there is no second language list.
    var language: String?

    /// Decode parameters. Not a user setting; see `WhisperCppDecodingConfig`
    /// for what the default reproduces and why it is a literal rather than a
    /// picker. Left mutable so a comparison run can change one field.
    var decodingConfig: WhisperCppDecodingConfig = .meetilyParity

    private(set) var modelState: EngineModelState = .unloaded
    private(set) var downloadProgress: Double = 0
    /// Coarse by design: 0 while decoding, 1 when done. whisper.cpp can report
    /// finer progress through a C callback, but nothing in the app reads this
    /// value (the protocol declaration carries a lint suppression saying so),
    /// and `ParakeetEngine` reports it the same way.
    private(set) var transcriptionProgress: Double = 0

    private let runner = WhisperCppRunner()
    private let modelLoad = SingleFlight()
    private var idleUnload: Task<Void, Never>?

    /// How long the ~3.1 GB context stays resident after the last decode.
    ///
    /// The other two engines never unload, and for them that is fine: Parakeet
    /// is ~50 MB and WhisperKit's CoreML model is around a gigabyte and paged
    /// by the ANE runtime. This one is a single malloc'd f16 model, and holding
    /// it for a whole session on a 16 GB machine competes with the recorder,
    /// the diarizer and the live-caption models during the next meeting — the
    /// part of the session where the app must not be the reason the system
    /// swaps. Post-meeting processing is explicitly allowed to be slow, so
    /// paying the reload is the right side of that trade.
    static let idleUnloadDelay: Duration = .seconds(300)

    func loadModel() async {
        await modelLoad.run { [self] in
            cancelIdleUnload()
            do {
                let modelURL = try await installModel()
                modelState = .loading
                downloadProgress = 1
                try await runner.load(modelPath: modelURL.path)
                modelState = .loaded
                logger.info("whisper.cpp engine ready")
                // Also armed here, not only after a decode: the launch preload
                // and a user switching straight back to another engine both
                // leave a loaded model that nothing will ever transcribe with,
                // and those are exactly the cases where 3 GB should not stay
                // resident for the session.
                scheduleIdleUnload()
            } catch {
                logger.error("whisper.cpp model load failed: \(error.localizedDescription, privacy: .public)")
                modelState = .unloaded
                downloadProgress = 0
            }
        }
    }

    /// Fetch the model if needed, reporting `.downloading` only when a transfer
    /// actually happens.
    ///
    /// The state is not set to `.downloading` unconditionally the way
    /// `WhisperKitEngine.loadModel` does it. There it is harmless because the
    /// call returns immediately from cache; here a driver script or an e2e lane
    /// polling `/state` would see a 2.9 GB download announce itself and finish
    /// in the same instant, which reads like a bug in whatever it is watching.
    private func installModel() async throws -> URL {
        if WhisperCppModel.state() == .present { return WhisperCppModel.installedURL }
        modelState = .downloading
        downloadProgress = 0
        let installer = WhisperCppModelInstaller { [weak self] fraction in
            Task { @MainActor in self?.downloadProgress = fraction }
        }
        return try await installer.ensureInstalled()
    }

    /// Transcribe a 16 kHz mono WAV (the pipeline resamples before calling) and
    /// return per-utterance segments on the file's own timeline.
    func transcribeSegments(audioPath: URL) async throws -> [TimestampedSegment] {
        try await ensureModel()
        cancelIdleUnload()
        transcriptionProgress = 0

        let samples = try await Task.detached(priority: .userInitiated) {
            try await Self.prepareSamples(at: audioPath)
        }.value
        guard !samples.isEmpty else {
            transcriptionProgress = 1
            scheduleIdleUnload()
            return []
        }

        let raw = try await runner.transcribe(
            samples: samples,
            language: WhisperCppRunner.supportedLanguage(WhisperCppLanguage.code(for: language)),
            config: decodingConfig,
        )
        transcriptionProgress = 1
        scheduleIdleUnload()
        return WhisperCppSegmentBuilder.segments(from: raw)
    }

    private func ensureModel() async throws {
        if modelState == .loaded { return }
        logger.info("whisper.cpp: model not loaded, loading…")
        await loadModel()
        guard modelState == .loaded else {
            throw WhisperCppEngineError.modelNotLoaded
        }
    }

    // MARK: - Idle unload

    private func cancelIdleUnload() {
        idleUnload?.cancel()
        idleUnload = nil
    }

    private func scheduleIdleUnload() {
        cancelIdleUnload()
        idleUnload = Task { @MainActor [weak self] in
            // `try?` rather than a throwing task: the only error is
            // cancellation, and cancellation is the normal way this ends.
            try? await Task.sleep(for: Self.idleUnloadDelay)
            guard let self, !Task.isCancelled else { return }
            await self.releaseModel()
        }
    }

    /// Drop the loaded weights.
    private func releaseModel() async {
        cancelIdleUnload()
        guard modelState != .unloaded else { return }
        await runner.unload()
        modelState = .unloaded
        downloadProgress = 0
    }

    // MARK: - Audio preparation

    /// Decode the file, guarantee 16 kHz, and apply the input gain.
    ///
    /// `nonisolated` and driven from a detached task: the class is `@MainActor`,
    /// so its static members are too by default, and `AudioMixer.resample` plus
    /// the gain pass are synchronous work over tens of millions of samples. Run
    /// on the main actor they would freeze the UI for the length of a meeting's
    /// worth of audio.
    nonisolated static func prepareSamples(at url: URL) async throws -> [Float] {
        let (decoded, rate) = try await AudioMixer.loadAudioAsFloat32(url: url)
        // whisper.cpp requires exactly WHISPER_SAMPLE_RATE (16 kHz) and does no
        // resampling of its own. The pipeline already hands over 16 kHz files,
        // so this is a guard for the import path, not the normal case.
        let mono16k = rate == AudioConstants.targetSampleRate
            ? decoded
            : AudioMixer.resample(decoded, from: rate, to: AudioConstants.targetSampleRate)
        let (leveled, gain) = WhisperCppInputGain.applied(to: mono16k)
        if gain > 1 {
            let decibels = String(format: "%.1f", 20 * log10(Double(gain)))
            logger.info("whisper.cpp input gain \(decibels, privacy: .public) dB applied to quiet audio")
        }
        return leveled
    }
}

/// Chunk-at-a-time decoding, the framing the reference pipeline uses.
///
/// Same model, same weights, same `WhisperCppDecodingConfig` as
/// `transcribeSegments` — only the shape of the input differs, and that is
/// what the measurement says the quality hangs on. `SpeechChunkPlanner` carries
/// the numbers.
///
/// In this file and not a `+Chunked` one because the members it drives
/// (`runner`, `ensureModel`, the idle-unload pair, `transcriptionProgress`'s
/// setter) are deliberately `private`, and Swift's `private` is file-scoped.
/// Widening them to satisfy a file split would trade real encapsulation for
/// layout.
extension WhisperCppEngine: ChunkedTranscribingEngine {
    /// Decode each chunk on its own and take its timing from the chunk rather
    /// than from the decoder.
    ///
    /// This is what lets the path set `no_timestamps`, the one flag this app
    /// deliberately diverged on. The reference pipeline sets it and reads
    /// timing off its own chunk offsets; the divergence existed only because
    /// whole-file decoding has no other source of per-utterance timing. Handed
    /// VAD regions it does have one, and a better-bounded one: a region marks
    /// where speech actually stopped, a timestamp token only predicts it.
    ///
    /// `meetilyParity` is untouched — the flags move on a local copy, because
    /// that literal records a measured configuration and is not a scratch pad.
    func transcribeChunks(audioPath: URL, chunks: [SpeechRegion]) async throws -> [TimestampedSegment] {
        try await ensureModel()
        cancelIdleUnload()
        transcriptionProgress = 0
        defer { scheduleIdleUnload() }

        let samples = try await Task.detached(priority: .userInitiated) {
            try await Self.prepareSamples(at: audioPath)
        }.value
        guard !samples.isEmpty, !chunks.isEmpty else {
            transcriptionProgress = 1
            return []
        }

        var config = decodingConfig
        // One window per chunk, so there is no timestamp token to read and none
        // to get wrong — which is the failure that makes the whole-file path
        // repeat itself.
        config.noTimestamps = true
        // A chunk is one utterance; asking for a single segment stops the
        // decoder splitting it on punctuation it has just invented.
        config.singleSegment = true

        let code = WhisperCppRunner.supportedLanguage(WhisperCppLanguage.code(for: language))
        let rate = AudioConstants.targetSampleRate
        var out: [TimestampedSegment] = []
        out.reserveCapacity(chunks.count)

        for (index, chunk) in chunks.enumerated() {
            guard let range = SpeechChunkPlanner.sampleRange(
                for: chunk, sampleRate: rate, sampleCount: samples.count,
            ) else { continue }

            let raw = try await runner.transcribe(
                samples: Array(samples[range]), language: code, config: config,
            )
            // The decoder's own bounds are meaningless with `no_timestamps`, so
            // only its text is used and the chunk supplies the timing. Joined
            // because `singleSegment` makes more than one segment the
            // exception rather than the plan.
            let text = WhisperCppSegmentBuilder.segments(from: raw)
                .map(\.text)
                .joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                out.append(TimestampedSegment(start: chunk.start, end: chunk.end, text: text))
            }
            // Real progress, which this path can finally report: the whole-file
            // decode is one opaque call and had to settle for 0-then-1.
            transcriptionProgress = Double(index + 1) / Double(chunks.count)
        }

        transcriptionProgress = 1
        logger.info(
            "whisper.cpp chunked decode: \(chunks.count, privacy: .public) chunks → \(out.count, privacy: .public) segments",
        )
        return out
    }
}

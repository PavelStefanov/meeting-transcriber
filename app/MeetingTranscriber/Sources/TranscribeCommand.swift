// `--transcribe` launch mode: decode one audio file with one of the app's
// Whisper engines and write the segments to a text file, without starting the
// GUI. Exists for `tools/asr-compare/compare.sh`, which puts these outputs
// beside those of models the app does not ship.
#if !APPSTORE
    import Foundation

    /// Parsed `--transcribe` request.
    ///
    ///     MeetingTranscriber --transcribe <whisperCpp|whisperKit> <in.wav> <out.txt>
    ///         [--lang <code>] [--whisperkit-variant <name>]
    ///
    /// It never downloads. Both engines would fetch a missing model into the
    /// app's store on `loadModel()`, and a comparison run should not be how
    /// 3 GB lands on disk, so a missing model fails with the path it was looked
    /// for at.
    struct TranscribeCommand {
        enum Engine: String {
            case whisperCpp
            case whisperKit
        }

        let engine: Engine
        let audio: URL
        let output: URL
        let language: String
        /// Defaults to the variant behind the picker's "Large V3".
        let whisperKitVariant: String

        static let flag = "--transcribe"
        static let usage = """
        usage: MeetingTranscriber --transcribe <whisperCpp|whisperKit> <in.wav> <out.txt> \
        [--lang <code>] [--whisperkit-variant <name>]
        """

        /// nil when the flag is absent (a normal launch); a usage error when it
        /// is present but malformed, so a typo never falls through to the GUI.
        static func parse(arguments: [String]) -> Result<Self, UsageError>? {
            guard let index = arguments.firstIndex(of: flag) else { return nil }
            var rest = Array(arguments[(index + 1)...])
            guard rest.count >= 3, let engine = Engine(rawValue: rest[0]) else {
                return .failure(UsageError())
            }
            let audio = URL(fileURLWithPath: rest[1])
            let output = URL(fileURLWithPath: rest[2])
            rest.removeFirst(3)

            var language = "ru"
            var variant = "openai_whisper-large-v3-v20240930"
            while !rest.isEmpty {
                guard rest.count >= 2 else { return .failure(UsageError()) }
                switch rest[0] {
                case "--lang": language = rest[1]
                case "--whisperkit-variant": variant = rest[1]
                default: return .failure(UsageError())
                }
                rest.removeFirst(2)
            }
            return .success(Self(
                engine: engine, audio: audio, output: output,
                language: language, whisperKitVariant: variant,
            ))
        }

        struct UsageError: Error {}

        enum RunError: Error, CustomStringConvertible {
            case modelMissing(String)
            case modelNotLoaded(String)

            var description: String {
                switch self {
                case let .modelMissing(path): "model not found at \(path) (load it once in the app first)"
                case let .modelNotLoaded(path): "model failed to load from \(path)"
                }
            }
        }

        /// Run on the main actor, where both engines live, then exit. Called
        /// before the GUI exists, so nothing else is draining the main queue:
        /// `dispatchMain()` does, and never returns.
        func runAndExit() -> Never {
            Task { @MainActor in
                do {
                    let segments = try await transcribe()
                    try Self.write(segments, to: output)
                    exit(0)
                } catch {
                    FileHandle.standardError.write(Data("transcribe failed: \(error)\n".utf8))
                    exit(1)
                }
            }
            dispatchMain()
        }

        @MainActor
        private func transcribe() async throws -> [TimestampedSegment] {
            let engine: any TranscribingEngine
            switch self.engine {
            case .whisperCpp:
                let model = WhisperCppModel.installedURL
                guard WhisperCppModel.state(at: model) == .present else {
                    throw RunError.modelMissing(model.path)
                }
                let whisperCpp = WhisperCppEngine()
                whisperCpp.language = language
                await whisperCpp.loadModel()
                guard whisperCpp.modelState == .loaded else { throw RunError.modelNotLoaded(model.path) }
                engine = whisperCpp

            case .whisperKit:
                let folder = AppPaths.whisperKitModelsDir
                    .appendingPathComponent("models/argmaxinc/whisperkit-coreml/\(whisperKitVariant)")
                guard FileManager.default.fileExists(atPath: folder.path) else {
                    throw RunError.modelMissing(folder.path)
                }
                let whisperKit = WhisperKitEngine()
                whisperKit.modelVariant = whisperKitVariant
                whisperKit.language = language
                await whisperKit.loadModel()
                guard whisperKit.modelState == .loaded else { throw RunError.modelNotLoaded(folder.path) }
                engine = whisperKit
            }

            // The threshold a job plans chunks with while the user's VAD
            // setting is off, which is its default.
            let vad = FluidVAD(threshold: PipelineQueue.chunkPlanningVadThreshold)
            return try await PipelineQueue.transcribeTrack(audio, engine: engine) {
                let (samples, _) = try await AudioMixer.loadAudioAsFloat32(url: audio)
                return try await vad.detectSpeech(samples: samples)
            }
        }

        private static func write(_ segments: [TimestampedSegment], to output: URL) throws {
            let lines = segments.compactMap { segment -> String? in
                let text = segment.text.trimmingCharacters(in: .whitespaces)
                return text.isEmpty ? nil : "[\(timestamp(segment.start))] \(text)"
            }
            try (lines.joined(separator: "\n") + "\n").write(to: output, atomically: true, encoding: .utf8)
        }

        private static func timestamp(_ seconds: TimeInterval) -> String {
            let total = Int(max(seconds, 0))
            return String(format: "%02d:%02d:%02d", total / 3600, total % 3600 / 60, total % 60)
        }
    }
#endif

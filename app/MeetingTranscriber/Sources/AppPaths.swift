import Foundation
import os.log

/// Centralized path constants and logger subsystem for the app.
enum AppPaths {
    /// Logger subsystem for all os.log loggers.
    static let logSubsystem = "com.meetingtranscriber"

    /// App data directory: `~/Library/Application Support/MeetingTranscriber/`
    /// In sandbox, this automatically resolves to the container path.
    /// Falls back to `~/.MeetingTranscriber/` if Application Support is unavailable.
    static let dataDir: URL = {
        if let appSupport = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            return appSupport.appendingPathComponent("MeetingTranscriber")
        }
        Logger(subsystem: logSubsystem, category: "AppPaths")
            .error("Application Support directory unavailable — falling back to home directory")
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".MeetingTranscriber")
    }()

    /// IPC directory: under `dataDir` for sandbox compatibility.
    static let ipcDir = dataDir.appendingPathComponent("ipc")

    /// Recordings directory.
    static let recordingsDir = dataDir.appendingPathComponent("recordings")

    /// Protocols output directory (legacy, inside Application Support).
    static let protocolsDir = dataDir.appendingPathComponent("protocols")

    /// Default protocols output in Downloads: `~/Downloads/MeetingTranscriber/`
    /// In sandbox, `FileManager.urls(for: .downloadsDirectory)` resolves to the container-granted path.
    static let downloadsProtocolsDir: URL = {
        guard let downloads = FileManager.default
            .urls(for: .downloadsDirectory, in: .userDomainMask).first
        else {
            return protocolsDir
        }
        return downloads.appendingPathComponent("MeetingTranscriber")
    }()

    /// whisper.cpp GGML model store: `<dataDir>/models/whisper.cpp/`.
    ///
    /// Under `dataDir` rather than `~/Library/Caches` on purpose: the full
    /// large-v3 weights are ~2.9 GB, and a cache directory is a place macOS is
    /// entitled to empty under disk pressure. It is also not the two existing
    /// model stores — WhisperKit downloads into its own Hugging Face directory
    /// and FluidAudio into `FluidAudio/Models` — because neither exposes a hook
    /// for a foreign artifact, and putting ours beside theirs would only make
    /// three owners of one directory.
    static let whisperCppModelsDir = dataDir
        .appendingPathComponent("models")
        .appendingPathComponent("whisper.cpp")

    /// WhisperKit model store: `<dataDir>/models/whisperkit/`.
    ///
    /// Passed to WhisperKit as its download base, because the default is
    /// `~/Documents/huggingface`: the user's own document space, frequently
    /// iCloud-synced, holding gigabytes of weights the app manages by itself —
    /// 6.4 GB across four variants on the machine this was found on. That is
    /// the whole justification, and it is enough.
    ///
    /// It is NOT justified by TCC. `~/Documents` is a protected location, and a
    /// load failure there did read like one ("Could not remove corrupted
    /// metadata file … you don't have permission to access it"), which is what
    /// first sent this change in. But the app turned out to write into
    /// `~/Documents` perfectly well — the tokenizer landed there on the very
    /// next run — so that diagnosis was wrong. The load failure was a bad
    /// on-disk state in the old cache, and what actually fixed it was moving
    /// away from that state plus the percent-encoding bug in `WhisperKitEngine`
    /// that this move exposed.
    ///
    /// Under `dataDir` for the same reason as `whisperCppModelsDir`, and beside
    /// it rather than inside it: two engines, two stores, one owner each.
    static let whisperKitModelsDir = dataDir
        .appendingPathComponent("models")
        .appendingPathComponent("whisperkit")

    /// Speaker voice profiles DB.
    static let speakersDB = dataDir.appendingPathComponent("speakers.json")

    /// Custom protocol prompt file.
    static let customPromptFile = dataDir.appendingPathComponent("protocol_prompt.md")

    /// Legacy IPC directory (`~/.meeting-transcriber/`) used before sandbox migration.
    private static let legacyIpcDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".meeting-transcriber")

    private static let logger = Logger(subsystem: logSubsystem, category: "AppPaths")

    /// Migrate IPC files from `~/.meeting-transcriber/` to `dataDir/ipc/`.
    /// Safe to call multiple times — copyItem fails gracefully if destination exists.
    static func migrateIfNeeded() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: legacyIpcDir.path) else { return }

        let filesToMigrate = [
            "processed_recordings.json",
            "pipeline_queue.json",
            "pipeline_log.jsonl",
        ]

        try? fm.createDirectory(at: ipcDir, withIntermediateDirectories: true)

        for name in filesToMigrate {
            let src = legacyIpcDir.appendingPathComponent(name)
            let dst = ipcDir.appendingPathComponent(name)
            do {
                try fm.copyItem(at: src, to: dst)
                logger.info("Migrated \(name) from legacy IPC directory")
            } catch CocoaError.fileWriteFileExists {
                // Already migrated — expected on subsequent launches
            } catch CocoaError.fileReadNoSuchFile {
                // Source doesn't exist — skip
            } catch {
                logger.error("Failed to migrate \(name): \(error.localizedDescription, privacy: .public)")
            }
        }
    }
}

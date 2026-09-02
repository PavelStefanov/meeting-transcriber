import CryptoKit
import Foundation

/// The one GGML model the whisper.cpp engine runs, and where it lives on disk.
///
/// Deliberately ONE artifact, not a catalogue. The engine exists to reproduce a
/// specific measured result — full-precision large-v3 through whisper.cpp — and
/// a picker of quantized variants would invite exactly the substitution the
/// engine was added to avoid. The turbo and quantized builds stay reachable
/// through the WhisperKit engine's existing model picker.
///
/// Pinning is by content, not by name, the same way
/// `scripts/fetch-localvqe-model.sh` pins the echo-cancellation weights: the
/// Hugging Face revision says which bytes to ask for and `sha256` says what
/// must have come back. `main` moves and a file at a path can be replaced, so
/// neither alone is a pin.
enum WhisperCppModel {
    /// Hugging Face repository that whisper.cpp itself publishes its GGML
    /// conversions from, and the one Meetily downloads from
    /// (`WhisperEngine::get_model_url`) — same repository, same filename, same
    /// bytes, so the model half of an A/B comparison against Meetily is exact.
    static let repository = "ggerganov/whisper.cpp"

    /// Commit of `ggerganov/whisper.cpp` the digest below was taken at.
    static let revision = "5359861c739e955e79d9a303bcbc70fb988958b1"

    /// f16 large-v3 — the full-precision conversion, ~2.9 GB. NOT
    /// `-turbo` (4 decoder layers instead of 32) and NOT a `-q5_0`/`-q8`
    /// quantization; the file name is the only place any of that is visible,
    /// which is why it is stated once here and derived everywhere else.
    static let filename = "ggml-large-v3.bin"

    /// Exact byte count of the pinned file. Checked before hashing, so a
    /// truncated transfer is rejected in O(1) instead of after re-reading 2.9 GB.
    static let sizeBytes: Int64 = 3_095_033_483

    /// SHA-256 of the pinned file. This is also the Hugging Face LFS object id
    /// for it, which is how it can be cross-checked without downloading:
    /// `POST /api/models/ggerganov/whisper.cpp/paths-info/<revision>`.
    static let sha256 = "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"

    /// Resolved download URL for the pinned revision.
    ///
    /// `resolve/<revision>/` and not `resolve/main/`: the digest above is a
    /// statement about one commit's content, and asking `main` for it would
    /// turn every upstream re-upload into a checksum failure that looks like a
    /// corrupt download.
    static var downloadURL: URL? {
        URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(filename)")
    }

    /// Installed location of the verified model.
    static var installedURL: URL {
        AppPaths.whisperCppModelsDir.appendingPathComponent(filename)
    }

    /// Partially downloaded file, sitting next to its destination so the move
    /// that completes an install is a rename within one filesystem and cannot
    /// half-succeed.
    static var partialURL: URL {
        AppPaths.whisperCppModelsDir.appendingPathComponent("\(filename).part")
    }

    /// What is on disk right now.
    enum State: Equatable {
        /// No file at `installedURL`.
        case absent
        /// A file is there whose size already contradicts the pin. Reported
        /// apart from `.absent` because the two want different words in a log:
        /// one is "not downloaded yet", the other is "downloaded, and wrong".
        case sizeMismatch(actual: Int64)
        /// A file of exactly the pinned size. NOT "verified" — hashing 2.9 GB
        /// takes seconds and is done once, at install time, not on every state
        /// read that a Settings redraw triggers.
        case present
    }

    /// Classify the installed file by size alone.
    static func state(fileManager: FileManager = .default) -> State {
        state(at: installedURL, fileManager: fileManager)
    }

    /// Size-only classification of an arbitrary path. Split out so tests can
    /// point it at a temporary directory instead of the user's real one.
    static func state(at url: URL, fileManager: FileManager = .default) -> State {
        guard let attributes = try? fileManager.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? Int64
        else {
            return .absent
        }
        return size == sizeBytes ? .present : .sizeMismatch(actual: size)
    }

    /// Read granularity for hashing. Streamed rather than read whole because
    /// the file is 2.9 GB: `Data(contentsOf:)` would put the entire model in
    /// the app's footprint right next to the copy whisper.cpp is about to load.
    private static let readBlockBytes = 4 * 1024 * 1024

    /// Streaming SHA-256 of the file at `url`, lowercase hex, or nil if it
    /// cannot be read.
    static func sha256Hex(of url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var hasher = SHA256()
        do {
            // `read(upToCount:)` answers nil at end of file and an empty
            // `Data` for a zero-length read, so both have to end the loop.
            while let block = try handle.read(upToCount: readBlockBytes), !block.isEmpty {
                hasher.update(data: block)
            }
        } catch {
            return nil
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Whether the file at `url` matches both halves of the pin.
    static func matchesPin(at url: URL, fileManager: FileManager = .default) -> Bool {
        guard state(at: url, fileManager: fileManager) == .present else { return false }
        return sha256Hex(of: url) == sha256
    }
}

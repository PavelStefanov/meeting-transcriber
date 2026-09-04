import Foundation
import os.log

private let logger = Logger(subsystem: AppPaths.logSubsystem, category: "WhisperCppModel")

/// Why a model install did not produce a usable file.
enum WhisperCppModelError: LocalizedError, Equatable {
    case unavailableURL
    case httpStatus(Int)
    case transport(String)
    case sizeMismatch(expected: Int64, actual: Int64)
    case checksumMismatch
    case installFailed(String)

    var errorDescription: String? {
        switch self {
        case .unavailableURL:
            "The whisper.cpp model URL could not be constructed"

        case let .httpStatus(code):
            "Model download failed with HTTP status \(code)"

        case let .transport(detail):
            "Model download failed: \(detail)"

        case let .sizeMismatch(expected, actual):
            "Model download is \(actual) bytes, expected \(expected)"

        case .checksumMismatch:
            "Model download did not match its pinned SHA-256 and was discarded"

        case let .installFailed(detail):
            "Model could not be installed: \(detail)"
        }
    }
}

/// Fetches `WhisperCppModel` into `AppPaths.whisperCppModelsDir` and verifies it
/// against the pin before it is ever visible under its final name.
///
/// A separate type from the engine so the network + filesystem work is not
/// `@MainActor`: the engine only awaits it and forwards progress.
///
/// Verification is not optional and a failure is not retried automatically. At
/// 2.9 GB a silent re-download is expensive enough to be worth a person
/// looking, and the two ways this fails say different things: a size or digest
/// mismatch on a revision-pinned URL means either the transfer is not what it
/// claims or the pin is stale, and both want attention rather than another
/// attempt.
struct WhisperCppModelInstaller: Sendable {
    /// Called with 0…1 as bytes arrive. Invoked off the main actor.
    let onProgress: @Sendable (Double) -> Void

    /// Returns the installed model path, downloading it first if what is on
    /// disk does not already match the pin's size.
    ///
    /// The already-present check is size-only, matching `WhisperCppModel.state`:
    /// a full re-hash on every engine load would read 2.9 GB off disk before
    /// each meeting. The digest is checked once, on the transfer that installs
    /// the file, which is the point where a bad copy can still be rejected.
    func ensureInstalled() async throws -> URL {
        let destination = WhisperCppModel.installedURL
        switch WhisperCppModel.state() {
        case .present:
            return destination

        case let .sizeMismatch(actual):
            logger.warning(
                "whisper.cpp model at destination is \(actual, privacy: .public) bytes, expected \(WhisperCppModel.sizeBytes, privacy: .public) — re-downloading",
            )
            try? FileManager.default.removeItem(at: destination)

        case .absent:
            break
        }

        guard let url = WhisperCppModel.downloadURL else { throw WhisperCppModelError.unavailableURL }
        try createModelsDirectory()

        let partial = WhisperCppModel.partialURL
        try? FileManager.default.removeItem(at: partial)

        logger.info("Downloading whisper.cpp model \(WhisperCppModel.filename, privacy: .public)…")
        try await download(from: url, to: partial)
        try verify(partial)

        do {
            try FileManager.default.moveItem(at: partial, to: destination)
        } catch {
            try? FileManager.default.removeItem(at: partial)
            throw WhisperCppModelError.installFailed(error.localizedDescription)
        }
        logger.info("whisper.cpp model installed at \(destination.path, privacy: .public)")
        return destination
    }

    private func createModelsDirectory() throws {
        do {
            try FileManager.default.createDirectory(
                at: AppPaths.whisperCppModelsDir, withIntermediateDirectories: true,
            )
        } catch {
            throw WhisperCppModelError.installFailed(error.localizedDescription)
        }
    }

    /// Size first, then digest. The size check is O(1) and rejects the common
    /// failure (a truncated transfer, or an HTML error page saved under a .bin
    /// name) without reading the file at all.
    private func verify(_ file: URL) throws {
        switch WhisperCppModel.state(at: file) {
        case .present:
            break

        case let .sizeMismatch(actual):
            try? FileManager.default.removeItem(at: file)
            throw WhisperCppModelError.sizeMismatch(expected: WhisperCppModel.sizeBytes, actual: actual)

        case .absent:
            throw WhisperCppModelError.sizeMismatch(expected: WhisperCppModel.sizeBytes, actual: 0)
        }
        guard WhisperCppModel.sha256Hex(of: file) == WhisperCppModel.sha256 else {
            try? FileManager.default.removeItem(at: file)
            throw WhisperCppModelError.checksumMismatch
        }
    }

    /// One download, delivered through a session-level delegate.
    ///
    /// A session of its own rather than `URLSession.shared`, and a session-level
    /// delegate rather than a per-task one, because the file has to be moved out
    /// of URLSession's temporary location inside
    /// `didFinishDownloadingTo` — that callback is the only point at which the
    /// temporary file is guaranteed to still exist. The session is invalidated
    /// afterwards so it stops retaining the delegate.
    private func download(from url: URL, to destination: URL) async throws {
        let (stream, continuation) = AsyncStream<Result<Void, WhisperCppModelError>>.makeStream()
        let delegate = WhisperCppDownloadDelegate(
            destination: destination, onProgress: onProgress, sink: continuation,
        )
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        session.downloadTask(with: url).resume()
        for await result in stream {
            switch result {
            case .success: return
            case let .failure(error): throw error
            }
        }
        throw WhisperCppModelError.transport("download ended without a result")
    }
}

/// Bridges `URLSessionDownloadDelegate`'s callbacks onto an `AsyncStream`, the
/// same shape `ClaudeCLIProtocolGenerator` uses to await process termination.
///
/// `@unchecked Sendable`: the stored values are immutable and the one mutable
/// field (`lastReportedPercent`) is only ever touched from URLSession's
/// serial delegate queue.
private final class WhisperCppDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let destination: URL
    private let onProgress: @Sendable (Double) -> Void
    private let sink: AsyncStream<Result<Void, WhisperCppModelError>>.Continuation
    private var lastReportedPercent = -1

    init(
        destination: URL,
        onProgress: @escaping @Sendable (Double) -> Void,
        sink: AsyncStream<Result<Void, WhisperCppModelError>>.Continuation,
    ) {
        self.destination = destination
        self.onProgress = onProgress
        self.sink = sink
    }

    /// Throttled to whole percentage points. Unthrottled this fires for every
    /// buffer of a 2.9 GB transfer, and each report crosses to the main actor.
    func urlSession(
        _: URLSession,
        downloadTask _: URLSessionDownloadTask,
        didWriteData _: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64,
    ) {
        // `totalBytesExpectedToWrite` is NSURLSessionTransferSizeUnknown (-1)
        // when the response carries no Content-Length. Fall back to the pinned
        // size, which is the length this transfer is required to have anyway.
        let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : WhisperCppModel.sizeBytes
        let fraction = min(max(Double(totalBytesWritten) / Double(total), 0), 1)
        let percent = Int(fraction * 100)
        guard percent != lastReportedPercent else { return }
        lastReportedPercent = percent
        onProgress(fraction)
    }

    func urlSession(
        _: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL,
    ) {
        if let response = downloadTask.response as? HTTPURLResponse, !(200 ..< 300).contains(response.statusCode) {
            sink.yield(.failure(.httpStatus(response.statusCode)))
            sink.finish()
            return
        }
        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
            onProgress(1)
            sink.yield(.success(()))
        } catch {
            sink.yield(.failure(.installFailed(error.localizedDescription)))
        }
        sink.finish()
    }

    /// Fires for transport failures and also after a successful
    /// `didFinishDownloadingTo`, where `error` is nil and the stream is already
    /// finished — a second `yield` on a finished continuation is a no-op, so no
    /// guard is needed beyond checking for an error.
    func urlSession(_: URLSession, task _: URLSessionTask, didCompleteWithError error: (any Error)?) {
        if let error {
            sink.yield(.failure(.transport(error.localizedDescription)))
        }
        sink.finish()
    }
}

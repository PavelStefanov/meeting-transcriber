@testable import MeetingTranscriber
import XCTest

/// Pins the whisper.cpp model artifact and the on-disk classification the
/// engine loads on.
///
/// The pin assertions are not busywork. The engine's whole reason to exist is
/// that it runs the FULL f16 large-v3 and not a turbo or quantized build, and
/// the only place that is visible is the filename, the size and the digest. A
/// well-meant edit to any one of them silently turns this engine into a
/// different experiment, so all three are asserted together with the reason
/// spelled out.
///
/// The digest and size were verified against the pinned Hugging Face revision
/// by downloading it and hashing the bytes; the same values are what
/// `POST /api/models/ggerganov/whisper.cpp/paths-info/<revision>` reports as
/// the LFS object id and size.
final class WhisperCppModelTests: XCTestCase {
    // MARK: - The pin

    func test_pin_namesTheFullPrecisionLargeV3() {
        XCTAssertEqual(
            WhisperCppModel.filename, "ggml-large-v3.bin",
            "must be the f16 conversion: -turbo has 4 decoder layers and -q5_0/-q8 are quantized",
        )
        XCTAssertFalse(WhisperCppModel.filename.contains("turbo"))
        XCTAssertFalse(WhisperCppModel.filename.contains("-q"))
    }

    func test_pin_sizeAndDigestMatchTheVerifiedArtifact() {
        XCTAssertEqual(WhisperCppModel.sizeBytes, 3_095_033_483)
        XCTAssertEqual(
            WhisperCppModel.sha256,
            "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
        )
    }

    func test_pin_digestIsALowercaseHex256BitValue() {
        XCTAssertEqual(WhisperCppModel.sha256.count, 64)
        XCTAssertTrue(
            WhisperCppModel.sha256.allSatisfy { $0.isHexDigit && !$0.isUppercase },
            "a mixed-case digest would never compare equal to a computed one",
        )
    }

    func test_downloadURL_resolvesThePinnedRevisionAndNotAMovingBranch() throws {
        let url = try XCTUnwrap(WhisperCppModel.downloadURL)
        XCTAssertEqual(
            url.absoluteString,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/"
                + "5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3.bin",
        )
        XCTAssertFalse(
            url.absoluteString.contains("/resolve/main/"),
            "a branch URL would turn an upstream re-upload into a checksum failure",
        )
    }

    func test_installedURL_livesUnderTheAppDataDirectoryAndNotInACache() {
        let path = WhisperCppModel.installedURL.path
        XCTAssertTrue(path.hasSuffix("/models/whisper.cpp/ggml-large-v3.bin"), path)
        XCTAssertFalse(
            path.contains("/Caches/"),
            "2.9 GB in a cache directory is 2.9 GB macOS may reclaim between meetings",
        )
    }

    func test_partialURL_sitsNextToTheDestination() {
        XCTAssertEqual(
            WhisperCppModel.partialURL.deletingLastPathComponent(),
            WhisperCppModel.installedURL.deletingLastPathComponent(),
            "the install has to be a rename inside one filesystem, not a cross-device copy",
        )
    }

    // MARK: - On-disk state

    func test_state_absentWhenNothingIsThere() {
        let missing = temporaryDirectory().appendingPathComponent("nope.bin")
        XCTAssertEqual(WhisperCppModel.state(at: missing), .absent)
    }

    func test_state_reportsSizeMismatchSeparatelyFromAbsent() throws {
        let file = temporaryDirectory().appendingPathComponent("short.bin")
        try Data(count: 17).write(to: file)
        XCTAssertEqual(WhisperCppModel.state(at: file), .sizeMismatch(actual: 17))
    }

    func test_matchesPin_rejectsAFileOfTheWrongSizeWithoutHashingIt() throws {
        let file = temporaryDirectory().appendingPathComponent("short.bin")
        try Data(count: 17).write(to: file)
        XCTAssertFalse(WhisperCppModel.matchesPin(at: file))
    }

    // MARK: - Digest

    /// The streaming hasher is the part that could silently be wrong: it reads
    /// in 4 MB blocks, so an off-by-one in the loop would produce a stable but
    /// incorrect digest that only a known-answer test catches.
    func test_sha256Hex_matchesTheKnownDigestOfAnEmptyFile() throws {
        let file = temporaryDirectory().appendingPathComponent("empty.bin")
        try Data().write(to: file)
        XCTAssertEqual(
            WhisperCppModel.sha256Hex(of: file),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
    }

    /// 5 MB spans two of the hasher's 4 MB read blocks, which is the case an
    /// off-by-one in the loop would get wrong while still passing on any file
    /// that fits in one block. Reference value from
    /// `head -c 5242880 /dev/zero | shasum -a 256`.
    func test_sha256Hex_matchesTheKnownDigestAcrossAReadBlockBoundary() throws {
        let file = temporaryDirectory().appendingPathComponent("zeros.bin")
        try Data(count: 5 * 1024 * 1024).write(to: file)
        XCTAssertEqual(
            WhisperCppModel.sha256Hex(of: file),
            "c036cbb7553a909f8b8877d4461924307f27ecb66cff928eeeafd569c3887e29",
        )
    }

    func test_sha256Hex_isNilForAMissingFile() {
        XCTAssertNil(
            WhisperCppModel.sha256Hex(of: temporaryDirectory().appendingPathComponent("gone.bin")),
        )
    }

    // MARK: - Helpers

    private func temporaryDirectory() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("WhisperCppModelTests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }
}

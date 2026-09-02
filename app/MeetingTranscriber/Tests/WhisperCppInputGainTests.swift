@testable import MeetingTranscriber
import XCTest

/// The input gain is boost-only, bounded and peak-guarded. Those three
/// properties are the whole safety argument for applying it at all — a gain
/// pass that could attenuate, run away, or clip would be able to make
/// transcription worse, which is the opposite of why it is here.
final class WhisperCppInputGainTests: XCTestCase {
    // MARK: - Cases that must be left alone

    func test_gain_isUnityForAnEmptyBuffer() {
        XCTAssertEqual(WhisperCppInputGain.gain(for: []), 1)
    }

    func test_gain_isUnityForDigitalSilence() {
        XCTAssertEqual(WhisperCppInputGain.gain(for: [Float](repeating: 0, count: 16000)), 1)
    }

    /// A recording already at or above the target must not be touched. The gain
    /// never goes below 1, so a hot track keeps its own level rather than being
    /// pulled down to -23 dBFS the way a full loudness normalizer would.
    func test_gain_isUnityForAudioAlreadyLouderThanTheTarget() {
        let loud = Self.tone(amplitude: 0.5, samples: 16000)
        XCTAssertEqual(WhisperCppInputGain.gain(for: loud), 1)
    }

    func test_gain_isUnityWhenThePeakLeavesNoHeadroom() {
        // Peak at full scale: any boost would clip, so none is applied even
        // though the RMS of a sparse signal like this sits below the target.
        var sparse = [Float](repeating: 0.0001, count: 16000)
        sparse[8000] = 1.0
        XCTAssertEqual(WhisperCppInputGain.gain(for: sparse), 1)
    }

    // MARK: - Cases that must be boosted

    func test_gain_boostsQuietSpeechTowardsTheTarget() {
        let quiet = Self.tone(amplitude: 0.01, samples: 16000)
        let gain = WhisperCppInputGain.gain(for: quiet)
        XCTAssertGreaterThan(gain, 1)

        let boostedRMS = Self.rootMeanSquare(quiet.map { $0 * gain })
        XCTAssertEqual(
            boostedRMS, WhisperCppInputGain.targetActiveRMS, accuracy: 0.005,
            "a steady tone has no silence to gate out, so it should land on the target",
        )
    }

    func test_gain_neverExceedsTheBoostCeiling() {
        // Far below the target: the desired gain is in the hundreds.
        let nearlySilent = Self.tone(amplitude: 0.00001, samples: 16000)
        XCTAssertEqual(WhisperCppInputGain.gain(for: nearlySilent), WhisperCppInputGain.maximumGain)
    }

    func test_appliedGain_neverProducesASampleAboveThePeakCeiling() {
        for amplitude in [Float(0.00001), 0.001, 0.01, 0.05, 0.2, 0.6, 0.95] {
            let (result, _) = WhisperCppInputGain.applied(to: Self.tone(amplitude: amplitude, samples: 8000))
            let peak = result.reduce(into: Float(0)) { $0 = max($0, abs($1)) }
            XCTAssertLessThanOrEqual(
                peak, WhisperCppInputGain.peakCeiling + 1e-5,
                "amplitude \(amplitude) clipped — the ceiling is what removes the need for a limiter",
            )
        }
    }

    func test_appliedGain_returnsTheInputUnchangedWhenNoBoostIsNeeded() {
        let loud = Self.tone(amplitude: 0.5, samples: 4000)
        let (result, gain) = WhisperCppInputGain.applied(to: loud)
        XCTAssertEqual(gain, 1)
        XCTAssertEqual(result, loud)
    }

    // MARK: - The activity gate

    /// The gate is the reason a normal recording is not treated as quiet. A
    /// track that is speech for a tenth of its length and silent for the rest
    /// has a very low overall RMS; measuring that would boost it hard and bring
    /// the noise floor up with it. Gating to the frames that carry the signal
    /// is what keeps the decision about the speech.
    func test_gain_ignoresTheSilenceBetweenUtterances() {
        let speech = Self.tone(amplitude: 0.3, samples: 16000)
        let silence = [Float](repeating: 0, count: 144_000)
        let mostlySilent = speech + silence

        XCTAssertEqual(
            WhisperCppInputGain.gain(for: mostlySilent), 1,
            "the speech is already above the target; the surrounding silence must not change that",
        )
    }

    // MARK: - Helpers

    /// A constant-amplitude square-ish signal: alternating ±amplitude, so RMS
    /// equals the amplitude exactly and the expected gain is arithmetic rather
    /// than approximate.
    private static func tone(amplitude: Float, samples: Int) -> [Float] {
        (0 ..< samples).map { $0.isMultiple(of: 2) ? amplitude : -amplitude }
    }

    private static func rootMeanSquare(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let total = samples.reduce(into: Float(0)) { $0 += $1 * $1 }
        return (total / Float(samples.count)).squareRoot()
    }
}

import Foundation

/// The one piece of audio preprocessing the whisper.cpp engine applies before
/// decoding: a single broadband gain that lifts quiet speech to a level Whisper
/// is comfortable with. Never attenuates, never clips.
///
/// ## Why there is anything here at all
///
/// Meetily normalizes loudness on the microphone path before transcription and
/// this pipeline does not, so "same model, same parameters" would still not be
/// the same input. Its live path runs a stateful EBU R128 normalizer to
/// -23 LUFS with a -1 dBTP true-peak limiter (`LoudnessNormalizer`), applied to
/// the microphone stream at 48 kHz during capture, after an 80 Hz high-pass and
/// optional RNNoise suppression.
///
/// ## Why it is not that
///
/// Three reasons, and they are the honest limits of this reproduction:
///
/// 1. **Meetily's own two paths disagree.** Its file-import path uses
///    `normalize_v2` instead, whose `target_rms = 0.9` is unreachable, so its
///    `min(rms_scaling, peak_scaling)` collapses to peak-normalising at 0.95 —
///    a completely different target from -23 LUFS. There is no single "the
///    Meetily normalization" to copy.
/// 2. **Streaming and whole-file are not the same operation.** Meetily's gain
///    tracks cumulative loudness and therefore changes over a meeting, with a
///    lookahead limiter to catch what the changing gain pushes over. A pass
///    over a finished file needs neither: the required gain can be computed
///    once and chosen so that nothing clips, which is strictly less
///    destructive than limiting.
/// 3. **This runs after capture, not during it.** The high-pass and RNNoise
///    stages sit in Meetily's capture pipeline. Reproducing them here would
///    change what one engine transcribes relative to the other two in this app,
///    and denoising is not what the engine was added for.
///
/// So: boost-only, so a healthy recording is passed through untouched and the
/// change can only affect the case it is aimed at; bounded, so near-silence
/// cannot be amplified into noise; and peak-guarded, so no sample is ever
/// clipped or soft-clipped.
enum WhisperCppInputGain {
    /// Target level for the active part of the signal, as linear amplitude.
    /// -23 dBFS RMS, chosen to match the LUFS target Meetily normalizes to.
    /// Not the same measurement — this is plain RMS, not K-weighted and
    /// gated — but for speech in a 16 kHz band the two land within a couple of
    /// dB of each other, and the gain is clamped well before that matters.
    static let targetActiveRMS: Float = 0.0708

    /// Hard ceiling on the boost, +20 dB. A recording that needs more than this
    /// is not quiet, it is nearly empty, and the remaining gain would be spent
    /// on the noise floor.
    static let maximumGain: Float = 10

    /// Highest sample magnitude the result may reach. Below 1.0 so the gain can
    /// never produce a clipped sample, which is what lets this skip a limiter
    /// entirely.
    static let peakCeiling: Float = 0.97

    /// Frames shorter than this are the analysis unit for the activity gate.
    /// 20 ms at 16 kHz.
    static let frameLength = 320

    /// A frame counts as active when its RMS is within this factor of the
    /// loudest frame — 20 dB down. A cheap stand-in for BS.1770's two-stage
    /// gate: it keeps the measurement off the silence between utterances,
    /// which is what would otherwise drag the average down and turn a normal
    /// recording into a boosted one.
    static let activityFloorRatio: Float = 0.1

    /// Gain to apply to `samples`, always ≥ 1.
    ///
    /// Returns exactly 1 when the signal is silent, already at level, or
    /// already close enough to the peak ceiling that no boost fits.
    static func gain(for samples: [Float]) -> Float {
        let peak = samples.reduce(into: Float(0)) { result, sample in
            result = max(result, abs(sample))
        }
        guard peak > 0 else { return 1 }

        let activeRMS = activeRootMeanSquare(of: samples)
        guard activeRMS > 0 else { return 1 }

        let desired = targetActiveRMS / activeRMS
        // Order matters: clamp to the boost range first, then let the peak
        // ceiling reduce it, then floor the result at 1. Applying the peak
        // ceiling first would let it *raise* a gain that the range had capped.
        let bounded = min(max(desired, 1), maximumGain)
        return max(1, min(bounded, peakCeiling / peak))
    }

    /// `samples` scaled by `gain(for:)`. Returns the input untouched when the
    /// gain is 1, so the common case allocates nothing.
    static func applied(to samples: [Float]) -> (samples: [Float], gain: Float) {
        let factor = gain(for: samples)
        guard factor > 1 else { return (samples, 1) }
        return (samples.map { $0 * factor }, factor)
    }

    /// RMS over the frames that carry the signal, ignoring the quiet ones.
    private static func activeRootMeanSquare(of samples: [Float]) -> Float {
        let frames = frameMeanSquares(of: samples)
        guard !frames.isEmpty else { return 0 }

        let loudest = frames.max() ?? 0
        guard loudest > 0 else { return 0 }

        // The gate is expressed as an amplitude ratio, so it squares into the
        // mean-square domain the frames are measured in.
        let threshold = loudest * activityFloorRatio * activityFloorRatio
        let active = frames.filter { $0 >= threshold }
        guard !active.isEmpty else { return 0 }

        return (active.reduce(0, +) / Float(active.count)).squareRoot()
    }

    /// Mean square of each whole `frameLength` block. A trailing partial frame
    /// is dropped: it would be measured over fewer samples than the gate's
    /// reference frame and could pass or fail the gate for that reason alone.
    private static func frameMeanSquares(of samples: [Float]) -> [Float] {
        guard samples.count >= frameLength else {
            guard !samples.isEmpty else { return [] }
            return [meanSquare(of: samples[...])]
        }
        var result: [Float] = []
        result.reserveCapacity(samples.count / frameLength)
        var start = samples.startIndex
        while start + frameLength <= samples.endIndex {
            result.append(meanSquare(of: samples[start ..< start + frameLength]))
            start += frameLength
        }
        return result
    }

    private static func meanSquare(of frame: ArraySlice<Float>) -> Float {
        guard !frame.isEmpty else { return 0 }
        let total = frame.reduce(into: Float(0)) { sum, sample in
            sum += sample * sample
        }
        return total / Float(frame.count)
    }
}

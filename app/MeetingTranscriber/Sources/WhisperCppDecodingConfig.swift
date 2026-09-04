import Foundation

/// The whisper.cpp decoding parameters this engine runs with, as a plain value.
///
/// It is a value type and not a settings surface: the engine was added to
/// reproduce one measured configuration, and every field below is a
/// transcription of what Meetily's `WhisperEngine::transcribe_audio_with_confidence`
/// passes to `whisper_full` on an Apple Silicon machine. Keeping them here — in
/// one literal, named, and commented — is what makes that claim checkable, and
/// what makes a later experiment a one-line edit instead of a hunt through the
/// C bridge.
///
/// ## What "Meetily parity" means, exactly
///
/// Meetily builds `FullParams::new(SamplingStrategy::BeamSearch { beam_size: 3,
/// patience: 1.0 })` and sets `temperature = 0.2`. That combination does NOT
/// run a beam search. `whisper_full` picks its decoder count as
///
///     case WHISPER_SAMPLING_BEAM_SEARCH:
///         if (t_cur > 0.0f) { n_decoders_cur = params.greedy.best_of; }
///         else              { n_decoders_cur = params.beam_search.beam_size; }
///
/// and `whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH)` leaves
/// `greedy.best_of` at its base default of `-1`, which `std::max(1, …)` turns
/// into a single decoder. So with any non-zero temperature the beam width is
/// ignored and the decode is single-path. Verified in whisper.cpp v1.7.1 (the
/// version Meetily links, via whisper-rs-sys 0.11.1) and unchanged in v1.9.2.
///
/// This is reproduced rather than corrected, on purpose. The comparison the
/// engine exists to win is against the configuration that was actually measured
/// as better on real Russian meeting audio, and "fixing" the beam width would
/// make the two runs different experiments. Setting `beamSize` with
/// `temperature: 0` is the obvious follow-up, and it is one field away.
struct WhisperCppDecodingConfig: Equatable, Sendable {
    /// Beam width requested from whisper.cpp. Inert while `temperature > 0` —
    /// see the type's discussion.
    var beamSize: Int32

    /// Beam-search patience. Documented as "not implemented" in whisper.h;
    /// carried because Meetily sets it and a future whisper.cpp may honour it.
    var patience: Float

    /// Initial decoding temperature, and therefore also the first rung of the
    /// fallback ladder: whisper.cpp retries a window at
    /// `temperature, +inc, … 1.0`, so 0.2 gives five attempts where 0.0 gives
    /// six.
    ///
    /// Non-zero temperature does not randomise this decode. With a single
    /// decoder the per-step choice is still the arg-max, and dividing the
    /// logits by 0.2 is a monotonic transform that cannot reorder them. What it
    /// does change is the *scale* of the log-probabilities the fallback
    /// thresholds are compared against: sharper logits mean a higher average
    /// log-probability and lower entropy, so `logprobThreshold` and
    /// `entropyThreshold` trip less often and the first pass is trusted more.
    var temperature: Float

    /// Increment for the temperature fallback ladder (whisper.cpp default;
    /// Meetily does not change it).
    var temperatureIncrement: Float

    /// whisper.cpp default; Meetily sets it explicitly to the same value.
    var entropyThreshold: Float

    /// whisper.cpp default; Meetily sets it explicitly to the same value.
    var logProbabilityThreshold: Float

    /// Meetily lowered this from whisper.cpp's 0.6 to 0.55, with the comment
    /// that 0.75 "rejected valid quiet speech". Below the threshold a window
    /// whose no-speech probability is high is dropped, so a lower value keeps
    /// more quiet speech and admits more silence.
    var noSpeechThreshold: Float

    /// whisper.cpp default; Meetily sets it explicitly to the same value.
    var maxInitialTimestamp: Float

    /// Suppress the blank token on the first decoding step (whisper.cpp
    /// default; Meetily sets it explicitly).
    var suppressBlank: Bool

    /// Suppress non-speech tokens — musical notes, `(applause)`, and similar.
    /// NOT a whisper.cpp default (it ships `false`); Meetily turns it on, and
    /// it is the single most audible difference from a stock whisper.cpp run on
    /// meeting audio.
    ///
    /// Named `suppress_nst` in whisper.cpp from v1.7.2 on, and
    /// `suppress_non_speech_tokens` before that. Same field.
    var suppressNonSpeechTokens: Bool

    /// Cap on characters per emitted segment, applied by
    /// `whisper_wrap_segment`. Only has an effect when `tokenTimestamps` is on,
    /// because the wrap needs per-token times to place the split.
    var maxSegmentLength: Int32

    /// Compute per-token timestamps. Meetily enables it; here it is also what
    /// makes `maxSegmentLength` live, and therefore what keeps segments short
    /// enough for diarization to attribute them.
    var tokenTimestamps: Bool

    /// Never force the whole window into one segment.
    var singleSegment: Bool

    /// Ask whisper.cpp NOT to emit timestamp tokens.
    ///
    /// **This is the one Meetily parameter this engine deliberately does not
    /// copy, and the reason is structural rather than a preference.** Meetily
    /// sets it to `true`, but Meetily also cuts its audio into VAD-bounded
    /// chunks of a few seconds and takes each chunk's timing from the chunk's
    /// own offset — it never reads a timestamp back out of whisper. This
    /// pipeline hands whole recordings to the engine and needs
    /// `TimestampedSegment.start`/`.end` per utterance, because those drive
    /// speaker diarization, the dual-track merge and the echo dedup.
    ///
    /// With `no_timestamps = true` whisper.cpp suppresses every timestamp token
    /// and then advances `seek` by a full 30 s window per decode
    /// (`seek_delta = 100*WHISPER_CHUNK_SIZE`), so a recording comes back as
    /// one segment per half-minute with a start time derived from a timestamp
    /// token that was never sampled. Diarization would collapse.
    ///
    /// Left as a field rather than hard-coded so the divergence is visible at
    /// the call site instead of being an unstated assumption in the bridge.
    var noTimestamps: Bool

    /// Condition each 30 s window on the text decoded before it.
    ///
    /// `true` — no conditioning — matching whisper.cpp's own default, which
    /// Meetily does not change. Note this is the opposite of the OpenAI
    /// reference implementation's `condition_on_previous_text=True`. Carrying
    /// context improves long-form coherence but is the classic route into a
    /// repetition loop that then poisons every later window.
    var noContext: Bool

    /// Decoding threads. whisper.cpp's own default is `min(4, cores)`, which is
    /// what Meetily effectively uses: it computes a thread count from its
    /// hardware profile and then never passes it to whisper (its own comment
    /// says "whisper.cpp may or may not expose thread control through params").
    /// Reproduced as the default rather than raised, because ggml's reductions
    /// are order-dependent and a different thread count is a different
    /// numerical result — which would make an A/B against Meetily inexact for
    /// a speed gain that the encoder, running on Metal, does not benefit from
    /// anyway.
    var threadCount: Int32

    /// The configuration Meetily runs on an Apple Silicon machine.
    ///
    /// Beam width and temperature come from its `PerformanceTier::High` branch,
    /// which is what an M-series Mac actually resolves to: the tier is chosen
    /// from `memory_gb >= 16 && cpu_cores >= 8`, and `detect_memory_gb()` reads
    /// the `MEMORY_GB` environment variable and otherwise returns a hard-coded
    /// 8. Nothing sets that variable, so a 16 GB M2 Pro lands on High
    /// (beam 3, temperature 0.2) and not on Ultra (beam 5, temperature 0.1).
    static let meetilyParity = Self(
        beamSize: 3,
        patience: 1.0,
        temperature: 0.2,
        temperatureIncrement: 0.2,
        entropyThreshold: 2.4,
        logProbabilityThreshold: -1.0,
        noSpeechThreshold: 0.55,
        maxInitialTimestamp: 1.0,
        suppressBlank: true,
        suppressNonSpeechTokens: true,
        maxSegmentLength: 200,
        tokenTimestamps: true,
        singleSegment: false,
        noTimestamps: false,
        noContext: true,
        threadCount: 4,
    )
}

/// Maps the app's existing Whisper language setting onto a whisper.cpp language
/// code.
///
/// Both engines are the same model family, so the app's one Whisper language
/// picker (`AppSettings.whisperLanguage`, `PickerLanguages.whisperKit`) is
/// reused verbatim and no second list exists. The codes agree for every entry
/// except one, and an unknown code is NOT harmless: whisper.cpp's
/// `whisper_lang_id` answers -1 and `whisper_full` then pushes
/// `whisper_token_lang(ctx, -1)` — the start-of-transcript token — into the
/// prompt instead of a language token, with no error returned anywhere. So the
/// mapping is checked before the call, not trusted.
enum WhisperCppLanguage {
    /// Codes the app's picker offers that whisper.cpp spells differently.
    ///
    /// `fil` is in the picker because WhisperKit accepts it; Whisper's own
    /// vocabulary only has Tagalog, as `tl`.
    static let aliases = ["fil": "tl"]

    /// Canonical whisper.cpp code for an app language setting, or nil for
    /// auto-detect.
    ///
    /// - Parameter setting: the stored value of `AppSettings.whisperLanguage`,
    ///   or the engine's already-optional `language`. Both an empty string and
    ///   nil are the app's auto-detect sentinel.
    static func code(for setting: String?) -> String? {
        guard let setting else { return nil }
        let trimmed = setting.trimmingCharacters(in: .whitespaces).lowercased()
        guard !trimmed.isEmpty, trimmed != "auto" else { return nil }
        return aliases[trimmed] ?? trimmed
    }
}

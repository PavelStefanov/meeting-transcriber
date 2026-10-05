#!/usr/bin/env bash
# Transcribe one audio file with four Russian-capable ASR setups, side by side:
#
#   1. GigaAM v3 (Sber, e2e RNNT, with punctuation)       — Python, .venv-gigaam
#   2. T-one (T-Bank, telephony CTC + KenLM)              — Python, .venv-tone
#   3. whisper.cpp, full f16 large-v3 — the app's engine  — app binary, --transcribe
#   4. WhisperKit "Large V3" from the app's model picker  — app binary, --transcribe
#
# Usage:
#   tools/asr-compare/compare.sh <audio file> [--lang ru] [--only gigaam,tone,whispercpp,whisperkit]
#       [--whisperkit-variant openai_whisper-large-v3]   (default: the picker's "Large V3")
#   tools/asr-compare/compare.sh --clean     # delete venvs + downloaded models (keeps out/)
#
# Everything this script creates stays in this folder and is gitignored:
#   .venv-gigaam/ .venv-tone/   Python environments (created on first run)
#   .cache/                     uv cache + GigaAM, T-one and Hugging Face downloads
#   out/<name>-<time>/          the transcripts, plus logs/ with per-engine output
# The two Whisper engines run through the app's own release build (the same
# `swift build -c release` run_app.sh does, so the build cache is shared), reuse
# the models the app already has under
# ~/Library/Application Support/MeetingTranscriber/models and never download.
#
# First run downloads about 7 GB: ~0.9 GB of Python packages, GigaAM's 450 MB
# checkpoint, and T-one's 144 MB acoustic model plus its 5.5 GB KenLM language
# model. TONE_DECODER=greedy skips the LM (less accurate; see tone_run.py).
# GigaAM and T-one are Russian-only; --lang only steers the Whisper engines.
#
# Transcripts of real meetings are recording content: out/ must never be
# committed, and nothing from it belongs in a commit message, PR or issue.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
APP="$REPO/app/MeetingTranscriber"

# Pinned so a rerun next month measures the same code.
GIGAAM_PKG="gigaam[torch] @ git+https://github.com/salute-developers/GigaAM@7447938d791c4f3e643386ee22c33777004293a5"
TONE_PKG="tone @ git+https://github.com/voicekit-team/T-one@3c5b6c015038173840e62cea99e10cdb1c759116"

export UV_CACHE_DIR="$HERE/.cache/uv"
export HF_HOME="$HERE/.cache/hf"
export XDG_CACHE_HOME="$HERE/.cache"
export TORCH_HOME="$HERE/.cache/torch"
export GIGAAM_DOWNLOAD_ROOT="$HERE/.cache/gigaam"
export HF_HUB_DISABLE_TELEMETRY=1

usage() { sed -n '9,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf "$HERE/.venv-gigaam" "$HERE/.venv-tone" "$HERE/.cache" "$HERE/work"
    echo "Removed venvs, caches and models. Transcripts in $HERE/out are kept."
    exit 0
fi

AUDIO=""
LANG_CODE="ru"
ONLY="gigaam,tone,whispercpp,whisperkit"
WK_VARIANT="openai_whisper-large-v3-v20240930"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang) LANG_CODE="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --whisperkit-variant) WK_VARIANT="$2"; shift 2 ;;
        -h|--help) usage ;;
        -*) echo "unknown option: $1" >&2; usage 1 ;;
        *) AUDIO="$1"; shift ;;
    esac
done
[[ -n "$AUDIO" ]] || usage 1
[[ -f "$AUDIO" ]] || { echo "no such file: $AUDIO" >&2; exit 1; }
AUDIO="$(cd "$(dirname "$AUDIO")" && pwd)/$(basename "$AUDIO")"
wants() { [[ ",$ONLY," == *",$1,"* ]]; }

command -v uv >/dev/null || { echo "uv is required: brew install uv" >&2; exit 1; }

ensure_venv() { # <dir> <package...>
    local dir="$1"; shift
    [[ -x "$dir/bin/python" ]] && return 0
    echo "Creating $(basename "$dir") (first run only)..."
    uv venv -q -p 3.12 "$dir"
    uv pip install -q -p "$dir/bin/python" "$@"
}

STEM="$(basename "${AUDIO%.*}")"
RUN="$HERE/out/$STEM-$(date +%Y%m%d-%H%M%S)"
WORK="$HERE/work/$$"
mkdir -p "$RUN/logs" "$WORK"
trap 'rm -rf "$WORK"; rmdir "$HERE/work" 2>/dev/null || true' EXIT

# One decode per rate, done once, with the converter macOS ships — the Python
# loaders would otherwise each want ffmpeg. GigaAM and both Whisper engines take
# 16 kHz; T-one is a telephony model and takes 8 kHz.
afconvert -f WAVE -d LEI16@16000 -c 1 "$AUDIO" "$WORK/in16k.wav"
afconvert -f WAVE -d LEI16@8000 -c 1 "$AUDIO" "$WORK/in8k.wav"
DURATION="$(afinfo "$WORK/in16k.wav" | awk '/estimated duration/ {printf "%.0f", $3}')"
echo "Audio: $STEM, ${DURATION}s, language $LANG_CODE"
echo "Output: $RUN"
echo

SUMMARY="$RUN/logs/summary.txt"
run_engine() { # <label> <output file> <command...>
    local label="$1" out="$2"; shift 2
    local log="$RUN/logs/$(basename "${out%.txt}").log"
    printf '%-40s ' "$label"
    local started=$SECONDS
    if "$@" >"$log" 2>&1; then
        local took=$((SECONDS - started))
        echo "ok   ${took}s"
        echo "$label: ok, ${took}s" >>"$SUMMARY"
    else
        local took=$((SECONDS - started))
        echo "FAIL ${took}s  (see $log)"
        echo "$label: FAILED after ${took}s" >>"$SUMMARY"
        rm -f "$out"
    fi
}

if wants gigaam; then
    ensure_venv "$HERE/.venv-gigaam" "$GIGAAM_PKG" silero-vad
    out="$RUN/1-gigaam-v3.txt"
    run_engine "1. GigaAM v3 e2e rnnt" "$out" \
        "$HERE/.venv-gigaam/bin/python" "$HERE/gigaam_run.py" "$WORK/in16k.wav" "$out"
fi

if wants tone; then
    ensure_venv "$HERE/.venv-tone" "$TONE_PKG" soundfile
    out="$RUN/2-t-one.txt"
    run_engine "2. T-one" "$out" \
        "$HERE/.venv-tone/bin/python" "$HERE/tone_run.py" "$WORK/in8k.wav" "$out"
fi

APP_BIN=""
app_binary() {
    [[ -n "$APP_BIN" ]] && return 0
    echo "Building the app (release)..."
    (cd "$APP" && swift build -c release >"$RUN/logs/app-build.log" 2>&1) \
        || { echo "app build failed, see $RUN/logs/app-build.log" >&2; return 1; }
    APP_BIN="$(cd "$APP" && swift build -c release --show-bin-path)/MeetingTranscriber"
}

swift_engine() { # <engine> <output file>
    "$APP_BIN" --transcribe "$1" "$WORK/in16k.wav" "$2" --lang "$LANG_CODE" \
        --whisperkit-variant "$WK_VARIANT"
}

if wants whispercpp || wants whisperkit; then
    if ! app_binary; then
        ONLY="${ONLY//whispercpp/}"
        ONLY="${ONLY//whisperkit/}"
    fi
fi

if wants whispercpp; then
    out="$RUN/3-whisper-cpp-large-v3.txt"
    run_engine "3. whisper.cpp large-v3 (f16)" "$out" swift_engine whisperCpp "$out"
fi

if wants whisperkit; then
    out="$RUN/4-whisperkit-${WK_VARIANT#openai_whisper-}.txt"
    run_engine "4. WhisperKit ${WK_VARIANT#openai_whisper-}" "$out" swift_engine whisperKit "$out"
fi

echo
echo "Done. Transcripts:"
ls -1 "$RUN"/*.txt 2>/dev/null | sed 's/^/  /' || echo "  (none — every engine failed, see $RUN/logs)"

#!/usr/bin/env bash
# Single source of truth for putting whisper.cpp into an app bundle.
#
# Unlike CLocalVQE, which is a static archive and therefore simply part of the
# executable, the whisper.cpp binary target is upstream's DYNAMIC
# `whisper.xcframework`. Two consequences, and a bundle that skips either one
# fails in a way that looks like nothing at all: the app is `LSUIElement`, so a
# process that dies in dyld before `main()` and one that started correctly look
# identical from the Dock.
#
#   1. The framework has to be inside the bundle, and the executable has to have
#      an rpath that finds it there. Its install name is
#      `@rpath/whisper.framework/Versions/Current/whisper`, and SwiftPM only
#      links the app against `@loader_path` (its own build directory), which
#      does not exist on the user's machine.
#   2. The framework has to be signed, by the same identity as the app. A
#      bundle signed with `--options runtime` enables Library Validation, which
#      requires embedded frameworks to share the app's Team ID; upstream's
#      artifact is signed by upstream, so dyld refuses to map it and the app
#      dies before `main()`. And even without hardened runtime, `codesign` seals
#      `Contents/Frameworks` into `CodeResources`, so `codesign --verify` on the
#      bundle fails outright if the nested framework carries no signature.
#
# Two scripts assemble a bundle — build_release.sh for distribution and
# run_app.sh for the dev app that e2e-app.sh deploys — and both need all of it,
# which is why it lives here instead of twice.
#
# Source this, don't execute it.

WHISPER_FRAMEWORK_NAME="whisper.framework"
# `@executable_path` and not `@loader_path`: the loader here is the executable
# itself, so the two resolve to the same directory, but this rpath is about
# where the EXECUTABLE is, and spelling it that way survives the framework
# later being loaded by something else in the bundle.
WHISPER_FRAMEWORK_RPATH="@executable_path/../Frameworks"

# whisper_framework_source <spm-dir> → path to the built whisper.framework
#
# Primary location is the SwiftPM build-products directory. That is not a guess:
# SwiftPM copies every binary dependency's library next to the product
# (`addBinaryDependencyCommands`) and passes that directory as the framework
# search path, which is also why `swift build` and `swift test` can run the
# linked binary at all.
#
# The fallback searches the artifact cache, because the products-directory copy
# is an implementation detail of SwiftPM's build plan and a layout change there
# would otherwise turn into "the app launches to nothing" rather than a build
# error.
whisper_framework_source() {
    local spm_dir="$1" candidate
    for config in release debug; do
        candidate="$spm_dir/.build/$config/$WHISPER_FRAMEWORK_NAME"
        if [ -d "$candidate" ]; then
            printf '%s' "$candidate"
            return 0
        fi
    done

    # macos-arm64_x86_64 is the slice name upstream publishes; matched on
    # "macos" so a rename of the architecture list does not break this.
    candidate="$(find "$spm_dir/.build/artifacts" \
        -type d -name "$WHISPER_FRAMEWORK_NAME" -path '*macos*' 2>/dev/null | head -1)"
    if [ -n "$candidate" ]; then
        printf '%s' "$candidate"
        return 0
    fi
    return 1
}

# install_whisper_framework <app-bundle> <spm-dir>
#
# Copies the framework into Contents/Frameworks and gives the executable an
# rpath that finds it. Fatal on failure in both callers: unlike the LocalVQE
# model, which resolves to "absent" and degrades, a missing whisper.framework
# means the bundle cannot start.
install_whisper_framework() {
    local bundle="$1" spm_dir="$2"
    local source frameworks binary
    [ -n "$bundle" ] || { echo "  ERROR: no app bundle given" >&2; return 1; }

    if ! source="$(whisper_framework_source "$spm_dir")"; then
        echo "  ERROR: $WHISPER_FRAMEWORK_NAME not found under $spm_dir/.build" >&2
        echo "  Run 'swift build' in $spm_dir first; SwiftPM copies the binary target there." >&2
        return 1
    fi

    frameworks="$bundle/Contents/Frameworks"
    mkdir -p "$frameworks" || return 1
    # Removed rather than copied over: a framework is a directory of symlinks,
    # and `ditto` onto an existing one merges instead of replacing, which would
    # leave a superseded `Versions/A` behind for `Versions/Current` to keep
    # pointing at.
    rm -rf "$frameworks/$WHISPER_FRAMEWORK_NAME" || return 1
    # ditto and not cp -R: it is the tool that preserves a bundle's symlink
    # farm and metadata exactly, which is what codesign then seals.
    ditto "$source" "$frameworks/$WHISPER_FRAMEWORK_NAME" || return 1

    binary="$bundle/Contents/MacOS/MeetingTranscriber"
    [ -f "$binary" ] || { echo "  ERROR: no executable at $binary" >&2; return 1; }
    # Idempotent: `install_name_tool -add_rpath` FAILS on a path the binary
    # already has, and run_app.sh reassembles a bundle in place.
    #
    # Matched inside the LC_RPATH stanza rather than against a whole line:
    # `otool -l` prints the value as `path <rpath> (offset N)`, so anchoring on
    # the end of the line would never match and the add would be attempted on
    # every run.
    if otool -l "$binary" | grep -A2 LC_RPATH | grep -qF "$WHISPER_FRAMEWORK_RPATH"; then
        echo "  whisper.cpp: rpath already present"
    else
        install_name_tool -add_rpath "$WHISPER_FRAMEWORK_RPATH" "$binary" || return 1
    fi

    echo "  whisper.cpp framework: $frameworks/$WHISPER_FRAMEWORK_NAME"
}

# sign_whisper_framework <app-bundle> <identity> [extra codesign args…]
#
# Signs the embedded framework. MUST run before the bundle itself is signed —
# code signing is inside-out, and a nested object signed afterwards invalidates
# the outer seal.
#
# A no-op when there is no identity: run_app.sh legitimately has none on a
# machine without a certificate, and there it leaves the whole bundle unsigned
# rather than half-signed.
#
# Not covered by build_release.sh's "sign all embedded libraries" loop, which
# matches `*.dylib` and `*.so` — a framework's binary has no extension at all.
sign_whisper_framework() {
    local bundle="$1" identity="$2"
    shift 2
    local framework="$bundle/Contents/Frameworks/$WHISPER_FRAMEWORK_NAME"
    [ -d "$framework" ] || { echo "  ERROR: no framework to sign at $framework" >&2; return 1; }
    if [ -z "$identity" ]; then
        echo "  whisper.cpp framework: left unsigned (no identity)"
        return 0
    fi
    codesign --force --sign "$identity" "$@" "$framework" || return 1
    echo "  whisper.cpp framework: signed"
}

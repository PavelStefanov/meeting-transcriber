// Process entry point. SwiftUI's App.main() is a protocol requirement, so a
// conforming type cannot intercept its own launch without recursing into
// itself; this separate @main enum makes the one pre-launch decision — divert
// into the LocalVQE selftest probe (scripts/localvqe-bundle-check.sh), into a
// one-file transcription (tools/asr-compare/compare.sh), or start the GUI —
// before any app state is constructed.
import SwiftUI

@main
enum AppLauncher {
    // Invoked by the @main synthesis, which the analyzer cannot see.
    // swiftlint:disable:next unused_declaration
    static func main() {
        // The selftest exists for the Homebrew build's bundle check and is
        // compiled out of the App Store variant entirely, like the debug RPC
        // server. Where present it is reachable only via this explicit argv
        // flag; it constructs no app state and exits before the GUI starts.
        #if !APPSTORE
            if let mode = LocalVQESelftest.parse(
                arguments: CommandLine.arguments, bundledModel: LocalVQEModel.resolve().path,
            ) {
                exit(LocalVQESelftest.run(mode))
            }
            switch TranscribeCommand.parse(arguments: CommandLine.arguments) {
            case let .success(command):
                command.runAndExit()

            case .failure:
                FileHandle.standardError.write(Data((TranscribeCommand.usage + "\n").utf8))
                exit(64)

            case nil:
                break
            }
        #endif
        MeetingTranscriberApp.main()
    }
}

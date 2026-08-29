import FluidAudio
import Foundation

private struct SegmentOutput: Encodable {
    let speaker: String
    let startSeconds: Float
    let endSeconds: Float
}

private struct DiarizationOutput: Encodable {
    let engine: String
    let sourceRevision: String
    let speakerCount: Int
    let segments: [SegmentOutput]
}

private enum ArgumentError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let value): value
        }
    }
}

@main
private struct MeetingDiarizer {
    private static let revision = "6428e29186573c6d33c598e25d460e6690bc0ee1"

    static func main() async {
        do {
            try await run(arguments: Array(CommandLine.arguments.dropFirst()))
        } catch {
            FileHandle.standardError.write(Data("meeting-diarizer failed: \(error)\n".utf8))
            Foundation.exit(1)
        }
    }

    private static func run(arguments: [String]) async throws {
        guard arguments.count >= 2 else {
            throw ArgumentError.message(
                "usage: meeting-diarizer INPUT.wav OUTPUT.json [--speakers N | --min-speakers N --max-speakers N]"
            )
        }

        let input = URL(fileURLWithPath: arguments[0])
        let output = URL(fileURLWithPath: arguments[1])
        var exactSpeakers: Int?
        var minimumSpeakers: Int?
        var maximumSpeakers: Int?
        var index = 2
        while index < arguments.count {
            let option = arguments[index]
            guard index + 1 < arguments.count, let value = Int(arguments[index + 1]), value > 0 else {
                throw ArgumentError.message("speaker options require a positive integer")
            }
            switch option {
            case "--speakers": exactSpeakers = value
            case "--min-speakers": minimumSpeakers = value
            case "--max-speakers": maximumSpeakers = value
            default: throw ArgumentError.message("unsupported option: \(option)")
            }
            index += 2
        }

        if exactSpeakers != nil && (minimumSpeakers != nil || maximumSpeakers != nil) {
            throw ArgumentError.message("exact speaker count cannot be combined with a range")
        }
        if let minimumSpeakers, let maximumSpeakers, minimumSpeakers > maximumSpeakers {
            throw ArgumentError.message("minimum speaker count cannot exceed maximum")
        }

        var config = OfflineDiarizerConfig()
        if let exactSpeakers {
            config = config.withSpeakers(exactly: exactSpeakers)
        } else if minimumSpeakers != nil || maximumSpeakers != nil {
            config = config.withSpeakers(min: minimumSpeakers, max: maximumSpeakers)
        }

        let manager = OfflineDiarizerManager(config: config)
        try await manager.prepareModels()
        let result = try await manager.process(input) { completed, total in
            guard total > 0 else { return }
            let percent = Int(Double(completed) / Double(total) * 100)
            FileHandle.standardError.write(Data("diarization progress: \(percent)%\n".utf8))
        }

        let segments = result.segments
            .map {
                SegmentOutput(
                    speaker: $0.speakerId,
                    startSeconds: $0.startTimeSeconds,
                    endSeconds: $0.endTimeSeconds
                )
            }
            .sorted {
                ($0.startSeconds, $0.endSeconds, $0.speaker)
                    < ($1.startSeconds, $1.endSeconds, $1.speaker)
            }
        let payload = DiarizationOutput(
            engine: "FluidAudio/CoreML",
            sourceRevision: revision,
            speakerCount: Set(segments.map(\.speaker)).count,
            segments: segments
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(payload)
        try data.write(to: output, options: [.atomic])
    }
}

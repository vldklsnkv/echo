// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MeetingDiarizer",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "meeting-diarizer", targets: ["MeetingDiarizer"])
    ],
    dependencies: [
        .package(path: "vendor/FluidAudio")
    ],
    targets: [
        .executableTarget(
            name: "MeetingDiarizer",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ]
        )
    ]
)

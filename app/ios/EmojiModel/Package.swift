// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "EmojiModel",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "EmojiModel", targets: ["EmojiModel"]),
    ],
    targets: [
        .target(
            name: "EmojiModel",
            // .process flattens the files into the bundle root; a nested "Resources"
            // folder makes codesign reject the iOS resource bundle.
            resources: [.process("Resources")]
        ),
        .testTarget(
            name: "EmojiModelTests",
            dependencies: ["EmojiModel"]
        ),
    ]
)

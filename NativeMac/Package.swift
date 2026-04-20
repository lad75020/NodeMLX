// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "NodeMLXNative",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "NodeMLXNative", targets: ["NodeMLXNative"])
    ],
    targets: [
        .executableTarget(
            name: "NodeMLXNative",
            path: "Sources/NodeMLXNative"
        )
    ]
)

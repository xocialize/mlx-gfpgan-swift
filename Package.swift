// swift-tools-version: 6.2
import PackageDescription

// mlx-gfpgan-swift — GFPGAN v1.4 blind face restoration for MLXEngine.
// U-Net degradation removal + StyleGAN2 (clean) decoder with channel-split SFT; 512² aligned crops.
// Upstream: TencentARC/GFPGAN — Apache-2.0, clean code path only (no NVIDIA-derived StyleGAN2 ops;
// decoder prior trained from scratch by BasicSR). See Sources/GFPGANMLXCore/GFPGAN.swift header.
let package = Package(
    name: "mlx-gfpgan-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "GFPGANMLXCore", targets: ["GFPGANMLXCore"]),
        .library(name: "MLXGFPGAN", targets: ["MLXGFPGAN"]),
        .executable(name: "gfpgan-gate", targets: ["GFPGANGate"]),
        .executable(name: "gfpgan-align-gate", targets: ["GFPGANAlignGate"]),
        .executable(name: "gfpgan-validate", targets: ["GFPGANValidate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.38.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "GFPGANMLXCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MLXGFPGAN",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "GFPGANMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXGFPGANTests",
            dependencies: [
                "GFPGANMLXCore", "MLXGFPGAN",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
        // Engine-driven authoritative footprint via MLXEngineTestKit (phys_footprint),
        // as opposed to the gate's --bench which reads the MLX pool + raw phys.
        .executableTarget(
            name: "GFPGANValidate",
            dependencies: [
                "MLXGFPGAN",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXEngineTestKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/Validate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "GFPGANAlignGate",
            dependencies: ["MLXGFPGAN"],
            path: "Sources/AlignGate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "GFPGANGate",
            dependencies: [
                "GFPGANMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/Gate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)

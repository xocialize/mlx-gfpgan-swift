import Foundation
import GFPGANMLXCore
import MLXToolKit

/// A GFPGAN checkpoint. v1.4 is the primary; v1.3 shares the identical architecture and key
/// set (a training-recipe sibling: softer, slightly higher fidelity on mild degradation) and
/// can be added as a second case with only a repo string when a corpus shows it earns a slot.
public enum GFPGANVariant: String, Codable, Sendable, CaseIterable {
    /// GFPGANv1.4 (`params_ema`) — the default release checkpoint: stronger restoration,
    /// better identity/detail balance on real-world degradation.
    case v1_4

    public var repo: String {
        switch self {
        case .v1_4: return "mlx-community/GFPGANv1.4-fp32"
        }
    }

    /// 349 MB at fp32. fp16 is DISQUALIFIED — the e2e dtype gate collapses (cosine ≈ −0.3,
    /// PSNR 8 dB: the high-magnitude-activation failure class). bf16 gates clean (48 dB) but
    /// per the Restormer precedent restoration stays fp32: the weights are small and the
    /// dtype buys ~175 MB on a ~350 MB model. Measure before changing.
    public var quant: Quant { .fp32 }
}

/// Init-time configuration for `GFPGANRestorePackage` (C9).
public struct GFPGANConfiguration: PackageConfiguration, ModelStorable {
    public var variant: GFPGANVariant

    /// Fresh Gaussian noise per run (upstream `randomize_noise=True` default) vs the
    /// checkpoint's stored noise buffers. **Stored is our default** — deterministic output,
    /// the parity-gate mode, and an upstream-supported path; the learned noise weights make
    /// the visual difference negligible.
    public var randomizeNoise: Bool

    /// Border feather (px, in the 512² crop) for the paste-back mask.
    public var pasteFeather: Int

    /// Faces whose eye distance is below this (px) are skipped — upstream's
    /// `eye_dist_threshold=5` guard against spurious detections.
    public var minEyeDistance: Float

    public var modelsRootDirectory: URL?
    public var weightsURL: URL?

    public init(variant: GFPGANVariant = .v1_4,
                randomizeNoise: Bool = false,
                pasteFeather: Int = 26,
                minEyeDistance: Float = 5,
                modelsRootDirectory: URL? = nil,
                weightsURL: URL? = nil) {
        self.variant = variant
        self.randomizeNoise = randomizeNoise
        self.pasteFeather = pasteFeather
        self.minEyeDistance = minEyeDistance
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsURL = weightsURL
    }

    var noiseMode: NoiseMode {
        randomizeNoise ? .random(seed: UInt64.random(in: 0 ..< UInt64.max)) : .stored
    }

    private enum CodingKeys: String, CodingKey {
        case variant, randomizeNoise, pasteFeather, minEyeDistance
    }
}

extension GFPGANConfiguration: QuantConfigured {
    public var quant: Quant { variant.quant }
}

extension GFPGANConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights", repo: variant.repo, revision: nil,
                      matching: ["model.safetensors"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsURL, FileManager.default.fileExists(atPath: weightsURL.path) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}

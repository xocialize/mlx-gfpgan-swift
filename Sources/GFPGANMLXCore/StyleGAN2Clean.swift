//
//  StyleGAN2Clean.swift
//  mlx-gfpgan-swift / GFPGANMLXCore
//
//  MLX-Swift port of gfpgan/archs/stylegan2_clean_arch.py — the StyleGAN2 generator
//  WITHOUT the NVIDIA-derived custom CUDA ops (that code path, and its non-commercial
//  license clause, never enters this port). Isomorphic: same classes, same key names,
//  same forward decomposition.
//
//  Layout: NHWC throughout. The converter maps every 4-D tensor with one (0,2,3,1)
//  transpose — conv kernels (O,I,kH,kW)→(O,kH,kW,I), the (1,C,1,1) biases →(1,1,1,C),
//  the stored noise buffers (1,1,H,W)→(1,H,W,1), the constant input (1,C,4,4)→(1,4,4,C) —
//  and the 5-D modulated-conv weights (1,O,I,k,k)→(1,O,k,k,I) via (0,1,3,4,2).
//

import Foundation
import MLX
import MLXNN
import MLXRandom

@inline(__always) func lrelu(_ x: MLXArray) -> MLXArray { leakyRelu(x, negativeSlope: 0.2) }

/// Normalize style codes: `x * rsqrt(mean(x², axis=1) + 1e-8)`.
public final class NormStyleCode: Module {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * rsqrt(mean(x * x, axis: 1, keepDims: true) + 1e-8)
    }
}

/// Modulated Conv2d used in StyleGAN2. No bias.
///
/// Weight shape is the converted `(1, O, kH, kW, I)`, so after per-sample style modulation
/// the kernel reshapes straight into MLX conv layout `(O, kH, kW, I)`.
public final class ModulatedConv2d: Module {
    public let inChannels: Int
    public let outChannels: Int
    public let kernelSize: Int
    public let demodulate: Bool
    public let sampleMode: String?
    public let eps: Float

    @ModuleInfo(key: "modulation") public var modulation: Linear
    @ParameterInfo(key: "weight") public var weight: MLXArray

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, numStyleFeat: Int,
                demodulate: Bool = true, sampleMode: String? = nil, eps: Float = 1e-8) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.demodulate = demodulate
        self.sampleMode = sampleMode
        self.eps = eps
        self._modulation.wrappedValue = Linear(numStyleFeat, inChannels, bias: true)
        self._weight.wrappedValue = MLXArray.zeros(
            [1, outChannels, kernelSize, kernelSize, inChannels])
    }

    /// - Parameters:
    ///   - x: `(b, h, w, c_in)`
    ///   - style: `(b, num_style_feat)`
    public func callAsFunction(_ x: MLXArray, _ style: MLXArray) -> MLXArray {
        let b = x.dim(0)
        // weight modulation: (1,O,k,k,I) * (b,1,1,1,I) -> (b,O,k,k,I)
        let s = modulation(style).reshaped([b, 1, 1, 1, inChannels])
        var w = weight * s

        if demodulate {
            let demod = rsqrt(w.square().sum(axes: [2, 3, 4]) + eps)   // (b, O)
            w = w * demod.reshaped([b, outChannels, 1, 1, 1])
        }

        var input = x
        if sampleMode == "upsample" {
            input = interpolateBilinear(input, scaleFactor: 2)
        } else if sampleMode == "downsample" {
            input = interpolateBilinear(input, scaleFactor: 0.5)
        }

        let pad = kernelSize / 2
        if b == 1 {
            return conv2d(input, w[0], padding: [pad, pad])
        }
        // General batch: per-sample kernels. Production is b == 1; this path exists for
        // completeness and mirrors upstream's groups=b trick without the layout gymnastics.
        let outs = (0 ..< b).map { i in
            conv2d(input[i ..< (i + 1)], w[i], padding: [pad, pad])
        }
        return concatenated(outs, axis: 0)
    }
}

/// Style conv: modulated conv → ×√2 → noise injection → bias → LeakyReLU(0.2).
public final class StyleConv: Module {
    @ModuleInfo(key: "modulated_conv") public var modulatedConv: ModulatedConv2d
    @ParameterInfo(key: "weight") public var weight: MLXArray       // noise strength, scalar (1,)
    @ParameterInfo(key: "bias") public var bias: MLXArray           // (1,1,1,C)

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, numStyleFeat: Int,
                demodulate: Bool = true, sampleMode: String? = nil) {
        self._modulatedConv.wrappedValue = ModulatedConv2d(
            inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize,
            numStyleFeat: numStyleFeat, demodulate: demodulate, sampleMode: sampleMode)
        self._weight.wrappedValue = MLXArray.zeros([1])
        self._bias.wrappedValue = MLXArray.zeros([1, 1, 1, outChannels])
    }

    /// `noise` is `(1_or_b, h, w, 1)`; pass the stored bank for determinism.
    public func callAsFunction(_ x: MLXArray, _ style: MLXArray, noise: MLXArray) -> MLXArray {
        var out = modulatedConv(x, style) * Float(2.0.squareRoot())   // "for conversion" — upstream
        out = out + weight * noise
        out = out + bias
        return lrelu(out)
    }
}

/// To RGB from features: 1×1 modulated conv (no demodulation) + bias, with a ×2-upsampled skip.
public final class ToRGB: Module {
    public let upsample: Bool
    @ModuleInfo(key: "modulated_conv") public var modulatedConv: ModulatedConv2d
    @ParameterInfo(key: "bias") public var bias: MLXArray           // (1,1,1,3)

    public init(inChannels: Int, numStyleFeat: Int, upsample: Bool = true) {
        self.upsample = upsample
        self._modulatedConv.wrappedValue = ModulatedConv2d(
            inChannels: inChannels, outChannels: 3, kernelSize: 1, numStyleFeat: numStyleFeat,
            demodulate: false, sampleMode: nil)
        self._bias.wrappedValue = MLXArray.zeros([1, 1, 1, 3])
    }

    public func callAsFunction(_ x: MLXArray, _ style: MLXArray, skip: MLXArray? = nil) -> MLXArray {
        var out = modulatedConv(x, style)
        out = out + bias
        if var s = skip {
            if upsample { s = interpolateBilinear(s, scaleFactor: 2) }
            out = out + s
        }
        return out
    }
}

/// Learned constant input, `(1, 4, 4, C)` broadcast over the batch.
public final class ConstantInput: Module {
    @ParameterInfo(key: "weight") public var weight: MLXArray

    public init(numChannel: Int, size: Int) {
        self._weight.wrappedValue = MLXArray.zeros([1, size, size, numChannel])
    }

    public func callAsFunction(_ batch: Int) -> MLXArray {
        broadcast(weight, to: [batch, weight.dim(1), weight.dim(2), weight.dim(3)])
    }
}

/// The style MLP — upstream is `Sequential(NormStyleCode, [Linear, LeakyReLU] × 8)`, so the
/// Linears sit at the odd Sequential indices and the state-dict keys are `style_mlp.1`,
/// `style_mlp.3`, … `style_mlp.15`. The safetensors keeps those upstream keys;
/// `GFPGANv1Clean.remapUpstreamKeys` maps them to `l1…l15` at load time, because
/// `ModuleParameters.unflattened` reads a numeric path component as an ARRAY index and
/// builds `[none, linear, none, …]` — structurally incompatible with a plain module.
///
/// ⚠️ DEAD AT INFERENCE for the shipping checkpoints: `GFPGANer` builds the clean arch with
/// `input_is_latent=True`, so the U-Net's `final_linear` output bypasses this MLP entirely.
/// It is ported (and its weights converted) to keep the key set isomorphic with upstream —
/// the strict load proves the checkpoint is the one we think it is.
public final class StyleMLP: Module {
    @ModuleInfo(key: "l1") public var l1: Linear
    @ModuleInfo(key: "l3") public var l3: Linear
    @ModuleInfo(key: "l5") public var l5: Linear
    @ModuleInfo(key: "l7") public var l7: Linear
    @ModuleInfo(key: "l9") public var l9: Linear
    @ModuleInfo(key: "l11") public var l11: Linear
    @ModuleInfo(key: "l13") public var l13: Linear
    @ModuleInfo(key: "l15") public var l15: Linear
    let norm = NormStyleCode()

    public init(numStyleFeat: Int) {
        self._l1.wrappedValue = Linear(numStyleFeat, numStyleFeat, bias: true)
        self._l3.wrappedValue = Linear(numStyleFeat, numStyleFeat, bias: true)
        self._l5.wrappedValue = Linear(numStyleFeat, numStyleFeat, bias: true)
        self._l7.wrappedValue = Linear(numStyleFeat, numStyleFeat, bias: true)
        self._l9.wrappedValue = Linear(numStyleFeat, numStyleFeat, bias: true)
        self._l11.wrappedValue = Linear(numStyleFeat, numStyleFeat, bias: true)
        self._l13.wrappedValue = Linear(numStyleFeat, numStyleFeat, bias: true)
        self._l15.wrappedValue = Linear(numStyleFeat, numStyleFeat, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = norm(x)
        for l in [l1, l3, l5, l7, l9, l11, l13, l15] {
            out = lrelu(l(out))
        }
        return out
    }
}

/// The 15 stored noise buffers (`noises.noise0 … noise14`), shipped inside the checkpoint.
/// Layer i has resolution `2^((i+5)/2)`: 4, 8, 8, 16, 16, …, 512, 512.
public final class NoiseBank: Module {
    @ParameterInfo(key: "noise0") public var noise0: MLXArray
    @ParameterInfo(key: "noise1") public var noise1: MLXArray
    @ParameterInfo(key: "noise2") public var noise2: MLXArray
    @ParameterInfo(key: "noise3") public var noise3: MLXArray
    @ParameterInfo(key: "noise4") public var noise4: MLXArray
    @ParameterInfo(key: "noise5") public var noise5: MLXArray
    @ParameterInfo(key: "noise6") public var noise6: MLXArray
    @ParameterInfo(key: "noise7") public var noise7: MLXArray
    @ParameterInfo(key: "noise8") public var noise8: MLXArray
    @ParameterInfo(key: "noise9") public var noise9: MLXArray
    @ParameterInfo(key: "noise10") public var noise10: MLXArray
    @ParameterInfo(key: "noise11") public var noise11: MLXArray
    @ParameterInfo(key: "noise12") public var noise12: MLXArray
    @ParameterInfo(key: "noise13") public var noise13: MLXArray
    @ParameterInfo(key: "noise14") public var noise14: MLXArray

    public init(numLayers: Int) {
        precondition(numLayers == 15, "NoiseBank is written for out_size=512 (15 layers)")
        func z(_ i: Int) -> MLXArray {
            let r = 1 << ((i + 5) / 2)
            return MLXArray.zeros([1, r, r, 1])
        }
        self._noise0.wrappedValue = z(0)
        self._noise1.wrappedValue = z(1)
        self._noise2.wrappedValue = z(2)
        self._noise3.wrappedValue = z(3)
        self._noise4.wrappedValue = z(4)
        self._noise5.wrappedValue = z(5)
        self._noise6.wrappedValue = z(6)
        self._noise7.wrappedValue = z(7)
        self._noise8.wrappedValue = z(8)
        self._noise9.wrappedValue = z(9)
        self._noise10.wrappedValue = z(10)
        self._noise11.wrappedValue = z(11)
        self._noise12.wrappedValue = z(12)
        self._noise13.wrappedValue = z(13)
        self._noise14.wrappedValue = z(14)
    }

    public subscript(_ i: Int) -> MLXArray {
        [noise0, noise1, noise2, noise3, noise4, noise5, noise6, noise7, noise8, noise9,
         noise10, noise11, noise12, noise13, noise14][i]
    }
}

/// How the generator sources its per-layer injected noise.
public enum NoiseMode: Sendable {
    /// The checkpoint's stored noise buffers — deterministic, and the mode every parity
    /// golden uses (`randomize_noise=False` upstream).
    case stored
    /// Fresh Gaussian noise per call (upstream's default). Seeded for reproducibility.
    case random(seed: UInt64)
}

/// StyleGAN2 generator (clean) with channel-split SFT modulation — the union of upstream's
/// `StyleGAN2GeneratorClean` and its `StyleGAN2GeneratorCSFT` subclass, since only the CSFT
/// variant is ever constructed by GFPGAN.
public final class StyleGAN2GeneratorCSFT: Module {
    public let numStyleFeat: Int
    public let logSize: Int
    public let numLayers: Int
    public let numLatent: Int
    public let sftHalf: Bool
    public let channels: [Int: Int]

    @ModuleInfo(key: "style_mlp") public var styleMLP: StyleMLP
    @ModuleInfo(key: "constant_input") public var constantInput: ConstantInput
    @ModuleInfo(key: "style_conv1") public var styleConv1: StyleConv
    @ModuleInfo(key: "to_rgb1") public var toRGB1: ToRGB
    @ModuleInfo(key: "style_convs") public var styleConvs: [StyleConv]
    @ModuleInfo(key: "to_rgbs") public var toRGBs: [ToRGB]
    @ModuleInfo(key: "noises") public var noises: NoiseBank

    public init(outSize: Int, numStyleFeat: Int = 512, numMlp: Int = 8,
                channelMultiplier: Int = 2, narrow: Float = 1, sftHalf: Bool = false) {
        precondition(numMlp == 8, "StyleMLP is written for num_mlp=8")
        self.numStyleFeat = numStyleFeat
        self.sftHalf = sftHalf

        var channels: [Int: Int] = [:]
        for (res, base) in [(4, 512), (8, 512), (16, 512), (32, 512),
                            (64, 256 * channelMultiplier), (128, 128 * channelMultiplier),
                            (256, 64 * channelMultiplier), (512, 32 * channelMultiplier),
                            (1024, 16 * channelMultiplier)] {
            channels[res] = Int(Float(base) * narrow)
        }
        self.channels = channels

        let logSize = Int(log2(Float(outSize)))
        self.logSize = logSize
        self.numLayers = (logSize - 2) * 2 + 1
        self.numLatent = logSize * 2 - 2

        self._styleMLP.wrappedValue = StyleMLP(numStyleFeat: numStyleFeat)
        self._constantInput.wrappedValue = ConstantInput(numChannel: channels[4]!, size: 4)
        self._styleConv1.wrappedValue = StyleConv(
            inChannels: channels[4]!, outChannels: channels[4]!, kernelSize: 3,
            numStyleFeat: numStyleFeat, demodulate: true, sampleMode: nil)
        self._toRGB1.wrappedValue = ToRGB(
            inChannels: channels[4]!, numStyleFeat: numStyleFeat, upsample: false)
        self._noises.wrappedValue = NoiseBank(numLayers: numLayers)

        var convs: [StyleConv] = []
        var rgbs: [ToRGB] = []
        var inCh = channels[4]!
        for i in 3 ... logSize {
            let outCh = channels[1 << i]!
            convs.append(StyleConv(inChannels: inCh, outChannels: outCh, kernelSize: 3,
                                   numStyleFeat: numStyleFeat, demodulate: true,
                                   sampleMode: "upsample"))
            convs.append(StyleConv(inChannels: outCh, outChannels: outCh, kernelSize: 3,
                                   numStyleFeat: numStyleFeat, demodulate: true,
                                   sampleMode: nil))
            rgbs.append(ToRGB(inChannels: outCh, numStyleFeat: numStyleFeat, upsample: true))
            inCh = outCh
        }
        self._styleConvs.wrappedValue = convs
        self._toRGBs.wrappedValue = rgbs
    }

    func resolvedNoise(_ mode: NoiseMode, batch: Int) -> [MLXArray] {
        switch mode {
        case .stored:
            return (0 ..< numLayers).map { noises[$0] }
        case .random(let seed):
            let key = MLXRandom.key(seed)
            let keys = MLXRandom.split(key: key, into: numLayers)
            return (0 ..< numLayers).map { i in
                let r = 1 << ((i + 5) / 2)
                return MLXRandom.normal([batch, r, r, 1], key: keys[i])
            }
        }
    }

    /// Forward for the CSFT generator.
    ///
    /// Implements the paths GFPGAN inference exercises: a single style entry, latent or
    /// style-space input, and provided/stored/random noise. (Truncation and two-style mixing
    /// are training/sampling-time features and are not ported.)
    ///
    /// - Parameters:
    ///   - styles: `(b, num_style_feat)` style code, or `(b, num_latent, num_style_feat)`
    ///     per-layer latents (the `different_w=True` case).
    ///   - conditions: the SFT `[scale0, shift0, scale1, shift1, …]` list.
    public func callAsFunction(styles: MLXArray, conditions: [MLXArray],
                               inputIsLatent: Bool = false,
                               noiseMode: NoiseMode = .stored) -> MLXArray {
        var style = styles
        if !inputIsLatent { style = styleMLP(style) }

        let b = style.dim(0)
        let noise = resolvedNoise(noiseMode, batch: b)

        var latent: MLXArray
        if style.ndim < 3 {
            latent = broadcast(style.expandedDimensions(axis: 1), to: [b, numLatent, numStyleFeat])
        } else {
            latent = style
        }

        var out = constantInput(b)
        out = styleConv1(out, latent[0..., 0], noise: noise[0])
        var skip = toRGB1(out, latent[0..., 1])

        var i = 1
        for level in 0 ..< toRGBs.count {
            out = styleConvs[2 * level](out, latent[0..., i], noise: noise[2 * level + 1])

            // the conditions may have fewer levels than the generator
            if i < conditions.count {
                if sftHalf {   // only apply SFT to half of the channels
                    let half = out.dim(-1) / 2
                    let outSame = out[.ellipsis, 0 ..< half]
                    let outSft = out[.ellipsis, half...] * conditions[i - 1] + conditions[i]
                    out = concatenated([outSame, outSft], axis: -1)
                } else {       // apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]
                }
            }

            out = styleConvs[2 * level + 1](out, latent[0..., i + 1], noise: noise[2 * level + 2])
            skip = toRGBs[level](out, latent[0..., i + 2], skip: skip)
            i += 2
            // Realize per level: one monolithic 512²-decoder graph otherwise materializes a
            // ~4 GB transient working set (and leaves ~2 GB of dirty driver pages behind);
            // bounding the graph per level keeps the peak at the widest single level.
            eval(out, skip)
            Memory.clearCache()
        }

        return skip
    }
}

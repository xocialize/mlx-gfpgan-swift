//
//  GFPGAN.swift
//  mlx-gfpgan-swift / GFPGANMLXCore
//
//  Role: MLX-Swift port of GFPGAN v1.3/v1.4 — blind face restoration on an aligned
//        512×512 crop. Degradation-removal U-Net + StyleGAN2 (clean) decoder modulated
//        by channel-split SFT.
//
//  Upstream: https://github.com/TencentARC/GFPGAN — Apache-2.0 with third-party carve-outs.
//            This port uses ONLY the "clean" code path (gfpganv1_clean_arch.py), which
//            contains no NVIDIA-derived StyleGAN2 code; the v1.3/v1.4 decoder prior was
//            trained from scratch by BasicSR (StyleGAN2_512_..._scratch_800k.pth), so the
//            weights carry no NVIDIA license taint either. Residual: trained on FFHQ
//            (dataset compilation CC-BY-NC-SA; industry-standard unsettled question).
//  Paper:    Wang et al., CVPR 2021. 87,143,276 parameters (v1.4, params_ema).
//
//  Conventions: NHWC; module keys mirror the upstream state dict exactly.
//

import Foundation
import MLX
import MLXNN

/// Residual block with bilinear up/downsampling: conv1 → lrelu → resample → conv2 → lrelu,
/// plus a resampled 1×1 skip.
public final class ResBlock: Module {
    public enum Mode: String, Sendable { case down, up }

    @ModuleInfo(key: "conv1") public var conv1: Conv2d
    @ModuleInfo(key: "conv2") public var conv2: Conv2d
    @ModuleInfo(key: "skip") public var skip: Conv2d
    public let scaleFactor: Float

    public init(inChannels: Int, outChannels: Int, mode: Mode = .down) {
        self._conv1.wrappedValue = WinogradFreeConv2d(
            inputChannels: inChannels, outputChannels: inChannels, kernelSize: 3, padding: 1)
        self._conv2.wrappedValue = WinogradFreeConv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._skip.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
        self.scaleFactor = mode == .down ? 0.5 : 2
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = lrelu(conv1(x))
        out = interpolateBilinear(out, scaleFactor: scaleFactor)
        out = lrelu(conv2(out))
        let sk = skip(interpolateBilinear(x, scaleFactor: scaleFactor))
        return out + sk
    }
}

/// One SFT condition head — upstream is `Sequential(Conv2d, LeakyReLU, Conv2d)`, so the convs
/// sit at Sequential indices 0 and 2 and the state-dict keys are `….0.weight` / `….2.weight`.
/// The safetensors keeps the upstream keys; `GFPGANv1Clean.remapUpstreamKeys` maps them to
/// `conv0`/`conv2` at load time (numeric path components read as array indices otherwise).
public final class ConditionBlock: Module {
    @ModuleInfo(key: "conv0") public var conv0: Conv2d
    @ModuleInfo(key: "conv2") public var conv2: Conv2d

    public init(channels: Int, outChannels: Int) {
        self._conv0.wrappedValue = WinogradFreeConv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        self._conv2.wrappedValue = WinogradFreeConv2d(
            inputChannels: channels, outputChannels: outChannels, kernelSize: 3, padding: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2(lrelu(conv0(x)))
    }
}

/// The GFPGAN architecture: U-Net + StyleGAN2 decoder with SFT (clean version).
public final class GFPGANv1Clean: Module, @unchecked Sendable {

    public struct Configuration: Sendable {
        public var outSize = 512
        public var numStyleFeat = 512
        /// v1.3/v1.4 use 2 (the `GFPGANer(arch: 'clean')` default). The U-Net halves it
        /// again internally (`unet_narrow = 0.5`).
        public var channelMultiplier = 2
        public var numMlp = 8
        /// True for every shipping checkpoint: the U-Net's `final_linear` output IS the
        /// per-layer latent; the decoder's style MLP is bypassed.
        public var inputIsLatent = true
        /// True for every shipping checkpoint: one 512-vector per decoder layer (16 total).
        public var differentW = true
        public var narrow: Float = 1
        /// True for every shipping checkpoint: SFT modulates half the channels.
        public var sftHalf = true

        /// The released v1.3/v1.4 configs are all defaults.
        public init() {}
    }

    public let configuration: Configuration
    public let logSize: Int
    let unetChannels: [Int: Int]

    @ModuleInfo(key: "conv_body_first") public var convBodyFirst: Conv2d
    @ModuleInfo(key: "conv_body_down") public var convBodyDown: [ResBlock]
    @ModuleInfo(key: "final_conv") public var finalConv: Conv2d
    @ModuleInfo(key: "conv_body_up") public var convBodyUp: [ResBlock]
    @ModuleInfo(key: "toRGB") public var toRGB: [Conv2d]
    @ModuleInfo(key: "final_linear") public var finalLinear: Linear
    @ModuleInfo(key: "stylegan_decoder") public var styleganDecoder: StyleGAN2GeneratorCSFT
    @ModuleInfo(key: "condition_scale") public var conditionScale: [ConditionBlock]
    @ModuleInfo(key: "condition_shift") public var conditionShift: [ConditionBlock]

    public init(_ cfg: Configuration = Configuration()) {
        self.configuration = cfg

        let unetNarrow = cfg.narrow * 0.5
        var ch: [Int: Int] = [:]
        for (res, base) in [(4, 512), (8, 512), (16, 512), (32, 512),
                            (64, 256 * cfg.channelMultiplier), (128, 128 * cfg.channelMultiplier),
                            (256, 64 * cfg.channelMultiplier), (512, 32 * cfg.channelMultiplier),
                            (1024, 16 * cfg.channelMultiplier)] {
            ch[res] = Int(Float(base) * unetNarrow)
        }
        self.unetChannels = ch

        let logSize = Int(log2(Float(cfg.outSize)))
        self.logSize = logSize
        let firstOutSize = 1 << logSize

        self._convBodyFirst.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: ch[firstOutSize]!, kernelSize: 1)

        var down: [ResBlock] = []
        var inCh = ch[firstOutSize]!
        for i in stride(from: logSize, to: 2, by: -1) {
            let outCh = ch[1 << (i - 1)]!
            down.append(ResBlock(inChannels: inCh, outChannels: outCh, mode: .down))
            inCh = outCh
        }
        self._convBodyDown.wrappedValue = down

        self._finalConv.wrappedValue = WinogradFreeConv2d(
            inputChannels: inCh, outputChannels: ch[4]!, kernelSize: 3, padding: 1)

        var up: [ResBlock] = []
        var rgb: [Conv2d] = []
        var scale: [ConditionBlock] = []
        var shift: [ConditionBlock] = []
        inCh = ch[4]!
        for i in 3 ... logSize {
            let outCh = ch[1 << i]!
            up.append(ResBlock(inChannels: inCh, outChannels: outCh, mode: .up))
            rgb.append(Conv2d(inputChannels: outCh, outputChannels: 3, kernelSize: 1))
            let sftOut = cfg.sftHalf ? outCh : outCh * 2
            scale.append(ConditionBlock(channels: outCh, outChannels: sftOut))
            shift.append(ConditionBlock(channels: outCh, outChannels: sftOut))
            inCh = outCh
        }
        self._convBodyUp.wrappedValue = up
        self._toRGB.wrappedValue = rgb
        self._conditionScale.wrappedValue = scale
        self._conditionShift.wrappedValue = shift

        let linearOut = cfg.differentW ? (logSize * 2 - 2) * cfg.numStyleFeat : cfg.numStyleFeat
        self._finalLinear.wrappedValue = Linear(ch[4]! * 4 * 4, linearOut)

        self._styleganDecoder.wrappedValue = StyleGAN2GeneratorCSFT(
            outSize: cfg.outSize, numStyleFeat: cfg.numStyleFeat, numMlp: cfg.numMlp,
            channelMultiplier: cfg.channelMultiplier, narrow: cfg.narrow, sftHalf: cfg.sftHalf)
        super.init()
        if let route = GFPGANConvRoute.environmentOverride { convRoute = route }
    }

    /// Route for the in-window 3×3 convs — the U-Net/SFT `Conv2d`s and the StyleGAN2 modulated
    /// convs (WinogradFreeConv2d.swift). Default `.conv3d`: exact, and no slower at these shapes.
    public var convRoute: GFPGANConvRoute {
        get {
            modules().lazy.compactMap { ($0 as? WinogradFreeConv2d)?.route }.first ?? .conv3d
        }
        set {
            for m in modules() {
                if let c = m as? WinogradFreeConv2d { c.route = newValue }
                if let c = m as? ModulatedConv2d { c.convRoute = newValue }
            }
        }
    }

    /// Restore an aligned face crop.
    ///
    /// - Parameters:
    ///   - x: `(b, 512, 512, 3)` RGB in **[-1, 1]** (`(img/255 - 0.5) / 0.5`).
    ///   - noiseMode: `.stored` (deterministic, the parity mode) or `.random` (upstream default).
    ///   - tap: gate-only observer, called with the oracle's golden names per stage.
    /// - Returns: `(b, 512, 512, 3)` RGB in [-1, 1], unclamped (clamp at the consumer, as
    ///   upstream's `tensor2img(min_max=(-1, 1))` does).
    public func callAsFunction(_ x: MLXArray, noiseMode: NoiseMode = .stored,
                               tap: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        var conditions: [MLXArray] = []
        var unetSkips: [MLXArray] = []

        // encoder
        var feat = lrelu(convBodyFirst(x))
        tap?("feat_first", feat)
        for i in 0 ..< (logSize - 2) {
            feat = convBodyDown[i](feat)
            unetSkips.insert(feat, at: 0)
            tap?("down\(i)", feat)
        }
        feat = lrelu(finalConv(feat))
        tap?("final_conv", feat)

        // style code — ⚠️ upstream flattens NCHW `(C, 4, 4)` C-major; from NHWC we must
        // transpose back before the flatten or `final_linear`'s columns are permuted.
        var styleCode = finalLinear(feat.transposed(0, 3, 1, 2).reshaped([feat.dim(0), -1]))
        tap?("style_code", styleCode)
        if configuration.differentW {
            styleCode = styleCode.reshaped([styleCode.dim(0), -1, configuration.numStyleFeat])
        }

        // decode
        for i in 0 ..< (logSize - 2) {
            feat = feat + unetSkips[i]
            feat = convBodyUp[i](feat)
            tap?("up\(i)", feat)
            let scale = conditionScale[i](feat)
            conditions.append(scale)
            let shift = conditionShift[i](feat)
            conditions.append(shift)
            tap?("scale\(i)", scale)
            tap?("shift\(i)", shift)
            eval(feat, scale, shift)   // per-level graph boundary — see the decoder note
            Memory.clearCache()
        }

        // decoder
        return styleganDecoderForward(styleCode, conditions: conditions, noiseMode: noiseMode,
                                      tap: tap)
    }

    /// The decoder call, with gate taps mirroring the oracle. Kept next to the main forward
    /// so the two can never drift: the un-tapped path IS this path.
    private func styleganDecoderForward(_ styleCode: MLXArray, conditions: [MLXArray],
                                        noiseMode: NoiseMode,
                                        tap: ((String, MLXArray) -> Void)?) -> MLXArray {
        let dec = styleganDecoder
        guard tap != nil else {
            return dec(styles: styleCode, conditions: conditions,
                       inputIsLatent: configuration.inputIsLatent, noiseMode: noiseMode)
        }

        // tapped replica of StyleGAN2GeneratorCSFT.callAsFunction (live path only)
        var style = styleCode
        if !configuration.inputIsLatent { style = dec.styleMLP(style) }
        let b = style.dim(0)
        let noise = dec.resolvedNoise(noiseMode, batch: b)

        var latent = style
        if latent.ndim < 3 {
            latent = broadcast(latent.expandedDimensions(axis: 1),
                               to: [b, dec.numLatent, dec.numStyleFeat])
        }

        var out = dec.constantInput(b)
        out = dec.styleConv1(out, latent[0..., 0], noise: noise[0])
        tap?("dec_conv1", out)
        var skip = dec.toRGB1(out, latent[0..., 1])

        var i = 1
        for level in 0 ..< dec.toRGBs.count {
            out = dec.styleConvs[2 * level](out, latent[0..., i], noise: noise[2 * level + 1])
            if i < conditions.count {
                if dec.sftHalf {
                    let half = out.dim(-1) / 2
                    let outSame = out[.ellipsis, 0 ..< half]
                    let outSft = out[.ellipsis, half...] * conditions[i - 1] + conditions[i]
                    out = concatenated([outSame, outSft], axis: -1)
                } else {
                    out = out * conditions[i - 1] + conditions[i]
                }
            }
            out = dec.styleConvs[2 * level + 1](out, latent[0..., i + 1],
                                                noise: noise[2 * level + 2])
            skip = dec.toRGBs[level](out, latent[0..., i + 2], skip: skip)
            tap?("dec_out\(level)", out)
            tap?("dec_skip\(level)", skip)
            i += 2
        }
        return skip
    }

    /// The converted safetensors mirrors the upstream state-dict keys exactly, including the
    /// two nn.Sequential spots whose child keys are bare indices (`style_mlp.1`,
    /// `condition_scale.N.{0,2}`). `ModuleParameters.unflattened` reads a numeric component
    /// as an ARRAY index, so those need named Swift keys and this mechanical remap.
    public static func remapUpstreamKeys(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(arrays.count)
        for (k, v) in arrays {
            var key = k
            if key.hasPrefix("stylegan_decoder.style_mlp.") {
                key = key.replacingOccurrences(of: "style_mlp.", with: "style_mlp.l")
            } else if key.hasPrefix("condition_scale.") || key.hasPrefix("condition_shift.") {
                // condition_scale.N.0.weight -> condition_scale.N.conv0.weight
                var parts = key.split(separator: ".").map(String.init)
                if parts.count == 4 { parts[2] = "conv" + parts[2] }
                key = parts.joined(separator: ".")
            }
            out[key] = v
        }
        return out
    }

    /// Loads converted safetensors weights under the strict verifier.
    public func loadWeights(from url: URL) throws {
        let arrays = Self.remapUpstreamKeys(try MLX.loadArrays(url: url))
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        eval(self)
    }
}

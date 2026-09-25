// Route for 3×3 convs inside mlx's Winograd conv2d window (mlx-swift ≤ 0.31.6 Metal numerics).
//
// mlx's Metal conv2d (mlx/backend/metal/conv.cpp `dispatch_conv_2D_gpu`) runs a Winograd
// F(6×6,3×3) kernel when ALL of these hold: kernel 3×3, stride 1, dilation 1, groups 1,
// C % 32 == 0, O % 32 == 0, C + O ≥ 256, N·H·W ≥ 4096. On M5 that path is lossy — relL2 per
// conv against an exact reference: fp32 6.4e-3 (its batched GEMM runs TF32, MLX_ENABLE_TF32
// defaults on, and the output transform amplifies that ~8×), bf16 5.8e-2, fp16 7.5e-3. Every
// other conv path is exact-class (fp32 ~1e-6; bf16 1.7e-3 = output rounding). conv3d with
// kT = 1 is the same conv on the implicit-GEMM path: exact, but 1.3–4× slower than Winograd at
// 256/512-channel shapes.
//
// GFPGAN v1.4 hits the window in 21 convs per 512² face: U-Net ResBlocks and SFT condition heads
// (128/256 ch at 64²/128²) and six StyleGAN2 modulated convs (512/256/128 ch at 64²…256²; a dense
// conv2d at batch 1). Measured GPU vs CPU (production fp32): relL2 1.0e-3, max 2 of 255 levels —
// all of it Winograd (no TF32-eligible matmuls here). Default: see `GFPGANv1Clean.convRoute`.
// Probe: `swift test --filter WinogradProbeTests` (weight-free). Removal: when the probe reports
// raw conv2d exact on a new mlx-swift pin, go back to plain Conv2d.
// `GFPGAN_CONV_ROUTE=winograd|conv3d|fp32Winograd` overrides the defaults (validation).

import Foundation
import MLX
import MLXNN

/// How a 3×3 conv inside mlx's Winograd window runs. Shapes outside the window always take plain
/// conv2d, which is mlx's exact implicit-GEMM path.
public enum GFPGANConvRoute: String, Sendable {
    /// mlx's default Winograd kernel — fastest; on M5 ~6.4e-3 relL2 per conv in fp32, ~5.8e-2 in bf16.
    case winograd
    /// conv3d with kT = 1 on the implicit-GEMM path — exact; 1.3–4× slower than Winograd.
    case conv3d
    /// Half-precision input upcast to fp32 for the Winograd kernel and the result cast back:
    /// ~6.8e-3 per conv instead of bf16's ~5.8e-2, at fp32-Winograd speed. `.winograd` for fp32.
    case fp32Winograd

    /// `GFPGAN_CONV_ROUTE` = winograd | conv3d | fp32Winograd, if set.
    static var environmentOverride: GFPGANConvRoute? {
        getenv("GFPGAN_CONV_ROUTE").flatMap { GFPGANConvRoute(rawValue: String(cString: $0)) }
    }
}

final class WinogradFreeConv2d: Conv2d {
    var route: GFPGANConvRoute = .conv3d

    /// mlx's Winograd dispatch predicate, evaluated on the actual input (NHWC).
    static func takesWinograd(
        input x: MLXArray, weight: MLXArray, stride: (Int, Int), dilation: (Int, Int),
        groups: Int
    ) -> Bool {
        guard x.ndim == 4, groups == 1, stride == (1, 1), dilation == (1, 1),
            weight.dim(1) == 3, weight.dim(2) == 3
        else { return false }
        let (c, o) = (x.dim(3), weight.dim(0))
        return c % 32 == 0 && o % 32 == 0 && c + o >= 256 && x.dim(0) * x.dim(1) * x.dim(2) >= 4096
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        Self.conv(
            x, weight: weight, bias: bias, stride: stride, padding: padding, dilation: dilation,
            groups: groups, route: route)
    }

    /// The routed conv for weights held outside a Conv2d module (e.g. the UNet's `Upsample2D`).
    static func conv(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray?, stride: (Int, Int) = (1, 1),
        padding: (Int, Int), dilation: (Int, Int) = (1, 1), groups: Int = 1,
        route: GFPGANConvRoute
    ) -> MLXArray {
        func plain() -> MLXArray {
            let y = conv2d(
                x, weight, stride: .init(stride), padding: .init(padding),
                dilation: .init(dilation), groups: groups)
            return bias.map { y + $0 } ?? y
        }
        guard route != .winograd,
            takesWinograd(
                input: x, weight: weight, stride: stride, dilation: dilation, groups: groups)
        else { return plain() }
        if route == .fp32Winograd {
            guard x.dtype != .float32 else { return plain() }
            var y = conv2d(
                x.asType(.float32), weight.asType(.float32), stride: .init(stride),
                padding: .init(padding), dilation: .init(dilation), groups: groups)
            if let bias { y = y + bias.asType(.float32) }
            return y.asType(x.dtype)
        }
        let y = conv3d(
            x.expandedDimensions(axis: 1), weight.expandedDimensions(axis: 1),
            stride: [1, 1, 1], padding: [0, padding.0, padding.1]
        ).squeezed(axis: 1)
        return bias.map { y + $0 } ?? y
    }
}

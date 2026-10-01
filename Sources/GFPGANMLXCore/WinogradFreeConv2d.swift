// Route for 3×3 convs inside mlx's Winograd conv2d window.
//
// mlx's Metal conv2d runs a Winograd F(6×6,3×3) kernel for 3×3 / stride 1 / dilation 1 / groups 1
// convs with C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path is lossy
// (its batched GEMM runs TF32): relL2 per conv vs an exact reference ~6.4e-3 fp32, ~5.8e-2 bf16.
// The routing itself lives in MLXExactConv (mlx-exact-conv-swift), shared across the fleet; its
// exact path holds on mlx-swift 0.31.x and 0.32.x, where the old kT = 1 conv3d trick stopped
// working (mlx#3785 sends small-depth conv3d back to Winograd).
//
// GFPGAN v1.4 hits the window in 21 convs per 512² face: U-Net ResBlocks and SFT condition heads
// (128/256 ch at 64²/128²) and six StyleGAN2 modulated convs (512/256/128 ch at 64²…256²; a dense
// conv2d at batch 1). Measured GPU vs CPU (production fp32): raw relL2 1.0e-3, max 2 of 255
// levels — all of it Winograd (no TF32-eligible matmuls here). Default: see `GFPGANv1Clean.convRoute`.
// Gate: `GFPGAN_LANE=1` GPULaneTests. Removal: when mlx exposes an exact conv opt-out (mlx#4595)
// or its Winograd stops being lossy, go back to plain Conv2d.
// `GFPGAN_CONV_ROUTE=winograd|conv3d|fp32Winograd` overrides the defaults (validation).

import Foundation
import MLX
import MLXExactConv
import MLXNN

/// How a 3×3 conv inside mlx's Winograd window runs. Shapes outside the window always take plain
/// conv2d, which is mlx's exact implicit-GEMM path.
public enum GFPGANConvRoute: String, Sendable {
    /// mlx's default Winograd kernel — fastest; on M5 ~6.4e-3 relL2 per conv in fp32, ~5.8e-2 in bf16.
    case winograd
    /// Exact implicit-GEMM path (MLXExactConv); the name is kept from the original kT = 1 conv3d route.
    case conv3d
    /// Half-precision input upcast to fp32 for the Winograd kernel and the result cast back:
    /// ~6.8e-3 per conv instead of bf16's ~5.8e-2, at fp32-Winograd speed. `.winograd` for fp32.
    case fp32Winograd

    /// `GFPGAN_CONV_ROUTE` = winograd | conv3d | fp32Winograd, if set.
    static var environmentOverride: GFPGANConvRoute? {
        getenv("GFPGAN_CONV_ROUTE").flatMap { GFPGANConvRoute(rawValue: String(cString: $0)) }
    }

    var exactConvRoute: ExactConvRoute {
        switch self {
        case .winograd: .winograd
        case .conv3d: .exact
        case .fp32Winograd: .fp32Winograd
        }
    }
}

final class WinogradFreeConv2d: Conv2d {
    var route: GFPGANConvRoute = .conv3d

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
        ExactConv.conv2d(
            x, weight: weight, bias: bias, stride: stride, padding: padding, dilation: dilation,
            groups: groups, route: route.exactConvRoute)
    }
}

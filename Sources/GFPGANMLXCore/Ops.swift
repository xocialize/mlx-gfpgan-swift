//
//  Ops.swift
//  mlx-gfpgan-swift / GFPGANMLXCore
//
//  Hand-rolled NHWC bilinear resize — lifted from the parity-locked implementation in
//  mlx-sea-raft-swift (Ops.swift), which matches torch `F.interpolate(mode='bilinear')`.
//  GFPGAN's clean arch uses bilinear/align_corners=False EVERYWHERE it resamples: the
//  U-Net ResBlocks (×2 and ×0.5), the StyleGAN2 upsample StyleConvs, and the ToRGB skip
//  chain. There is no upfirdn2d in the clean architecture.
//

import Foundation
import MLX

private func sampleCoords(out: Int, inSize: Int, alignCorners: Bool) -> MLXArray {
    let dst = MLXArray(Array(0..<out).map { Float($0) })
    if alignCorners {
        let scale = out > 1 ? Float(inSize - 1) / Float(out - 1) : 0
        return dst * scale
    }
    let scale = Float(inSize) / Float(out)
    return (dst + 0.5) * scale - 0.5
}

private func bilinear1D(_ x: MLXArray, axis: Int, out: Int, alignCorners: Bool) -> MLXArray {
    let inSize = x.shape[axis]
    if inSize == out { return x }
    let src = sampleCoords(out: out, inSize: inSize, alignCorners: alignCorners)
    let i0f = floor(src)
    let w1 = src - i0f, w0 = 1.0 - (src - i0f)
    let i0 = clip(i0f, min: 0, max: Float(inSize - 1)).asType(.int32)
    let i1 = clip(i0f + 1, min: 0, max: Float(inSize - 1)).asType(.int32)
    let g0 = take(x, i0, axis: axis)
    let g1 = take(x, i1, axis: axis)
    var shape = [Int](repeating: 1, count: x.ndim)
    shape[axis] = out
    return g0 * w0.reshaped(shape).asType(x.dtype) + g1 * w1.reshaped(shape).asType(x.dtype)
}

/// Bilinear resize (NHWC) — `F.interpolate(mode: "bilinear")`. Degenerate targets clamp to 1.
public func interpolateBilinear(_ x: MLXArray, scaleFactor: Float,
                                alignCorners: Bool = false) -> MLXArray {
    let H = x.shape[1], W = x.shape[2]
    let oH = max(1, Int((Float(H) * scaleFactor).rounded()))
    let oW = max(1, Int((Float(W) * scaleFactor).rounded()))
    var out = bilinear1D(x, axis: 1, out: oH, alignCorners: alignCorners)
    out = bilinear1D(out, axis: 2, out: oW, alignCorners: alignCorners)
    return out
}

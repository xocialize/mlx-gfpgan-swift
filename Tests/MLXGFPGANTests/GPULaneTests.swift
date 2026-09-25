// GPU-lane gate for mlx's lossy Winograd conv2d window (Sources/GFPGANMLXCore/
// WinogradFreeConv2d.swift): the production fp32 forward (stored noise) on the aligned 512² face
// golden input, on the CPU stream (exact-class reference) and on the GPU with the conv route on
// (default) and off (raw). The S0–S3 parity gates pin the CPU device and the GPU modes carry no
// fp32 threshold, so the GPU lane was never gated before this.
//
// Measured 2026-09-24 (M5 Max, mlx-swift 0.31.6): raw GPU vs CPU relL2 1.0e-3, max 2 of 255
// levels — all Winograd (GFPGAN has no TF32-eligible matmuls); with MLX_ENABLE_TF32=0 1.0e-6.
//
// Run: GFPGAN_LANE=1 swift test -c release -Xswiftc -enable-testing --filter GPULaneTests
// Overrides: GFPGAN_WEIGHTS (model.safetensors), GFPGAN_FACE (aligned 512² face PNG).

import CoreGraphics
import Foundation
import ImageIO
import MLX
import GFPGANMLXCore
import XCTest

final class GPULaneTests: XCTestCase {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    static func stats(_ a: MLXArray, _ ref: MLXArray) -> (rel: Float, text: String) {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        let q = { (x: MLXArray) in clip((x + 1) / 2, min: 0, max: 1) * 255 }  // production truncation
        let lv = abs(floor(q(a)) - floor(q(ref)))
        let mx = abs(d).max()
        eval(rel, mx, lv)
        return (rel.item(Float.self), String(format: "relL2 %.2e  maxAbs %.2e  8-bit max %d levels, %.2f%% px > 2",
            rel.item(Float.self), mx.item(Float.self), Int(lv.max().item(Float.self)),
            100 * mean(lv .> 2).item(Float.self)))
    }

    func testFaceGPUvsCPU() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["GFPGAN_LANE"] == "1", "set GFPGAN_LANE=1 to run")
        let weights = env["GFPGAN_WEIGHTS"]
            ?? "/Volumes/Satechi/Models/models/mlx-community/GFPGANv1.4-fp32/model.safetensors"
        let face = URL(fileURLWithPath: env["GFPGAN_FACE"]
            ?? Self.root.appendingPathComponent("oracle/goldens/full_face_in.png").path)
        let model = GFPGANv1Clean()
        try model.loadWeights(from: URL(fileURLWithPath: weights))

        guard let src = CGImageSourceCreateWithURL(face as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "GFPGAN", code: 1) }
        let (w, h) = (cg.width, cg.height)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        let rgb = (0..<(w * h * 3)).map { i in Float(rgba[(i / 3) * 4 + i % 3]) / 127.5 - 1 }
        let x = MLXArray(rgb, [1, h, w, 3])

        let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = model(x, noiseMode: .stored)
            eval(r)
            return r
        }
        Memory.clearCache()
        var t: [GFPGANConvRoute: [Double]] = [:]
        var out: [GFPGANConvRoute: MLXArray] = [:]
        for _ in 0..<3 {
            for route in [GFPGANConvRoute.conv3d, .winograd] {
                model.convRoute = route
                var y = model(x, noiseMode: .stored)
                eval(y)
                let t0 = Date()
                for _ in 0..<3 { y = model(x, noiseMode: .stored); eval(y) }
                t[route, default: []].append(Date().timeIntervalSince(t0) / 3 * 1000)
                out[route] = y
            }
        }
        model.convRoute = .conv3d
        let sR = Self.stats(out[.conv3d]!, ref), sW = Self.stats(out[.winograd]!, ref)
        func med(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        print("[\(face.lastPathComponent) \(w)×\(h), fp32, GPU vs CPU lane]")
        print(String(format: "  conv3d route   %@  %6.1f ms", sR.text, med(t[.conv3d]!)))
        print(String(format: "  raw Winograd   %@  %6.1f ms", sW.text, med(t[.winograd]!)))
        if getenv("MLX_ENABLE_TF32").map({ String(cString: $0) }) == "0" {
            XCTAssertLessThan(sR.rel, 1e-4, "conv3d route vs CPU lane (TF32 off)")
        } else {
            // No TF32-eligible matmuls here: the route alone makes the GPU lane exact-class.
            XCTAssertLessThan(sR.rel, 1e-4, "conv3d route vs CPU lane")
            XCTAssertLessThan(sR.rel, sW.rel / 3, "route vs raw Winograd, against the CPU lane")
        }
    }
}

//
//  main.swift
//  mlx-gfpgan-swift / GFPGANGate
//
//  Parity gates against the PyTorch oracle. Executable, not a test target — the SPM test
//  product's metallib is unreliable for GPU work.
//
//  All parity gates run randomize_noise=False (the checkpoint's stored noise buffers), the
//  mode the goldens were generated in. S3 checks every intermediate tap, so a break
//  localizes to the exact stage without a Python twin.
//

import Foundation
import GFPGANMLXCore
import MLX
import MLXNN

private let _unbuffered: Void = { setvbuf(stdout, nil, _IONBF, 0) }()

func fail(_ msg: String) -> Never { _ = _unbuffered; print("❌ \(msg)"); exit(1) }

func loadedModel(_ path: String) -> GFPGANv1Clean {
    let model = GFPGANv1Clean()
    do { try model.loadWeights(from: URL(fileURLWithPath: path)) }
    catch { fail("weight load failed: \(error)") }
    return model
}

func g(_ dir: String, _ name: String) -> MLXArray {
    do { return try loadNPY("\(dir)/\(name).npy") } catch { fail("golden \(name): \(error)") }
}

/// Key contract: Swift module tree ↔ converted checkpoint, strict both ways.
func gateS0(_ weightsPath: String) {
    _ = _unbuffered
    print("=== S0 · key contract ===\n")
    let model = GFPGANv1Clean()
    var swiftKeys: [String: [Int]] = [:]; var total = 0
    for (k, v) in model.parameters().flattened() { swiftKeys[k] = v.shape; total += v.size }
    print("Swift module tree : \(swiftKeys.count) tensors, \(total) params")
    guard let raw = try? MLX.loadArrays(url: URL(fileURLWithPath: weightsPath)) else {
        fail("could not load \(weightsPath)")
    }
    let loaded = GFPGANv1Clean.remapUpstreamKeys(raw)
    print("Checkpoint        : \(loaded.count) tensors, \(loaded.values.reduce(0) { $0 + $1.size }) params\n")
    let sk = Set(swiftKeys.keys), ck = Set(loaded.keys)
    let missing = sk.subtracting(ck).sorted(), unused = ck.subtracting(sk).sorted()
    if !missing.isEmpty { print("MISSING (\(missing.count)):"); missing.prefix(15).forEach { print("   \($0)  \(swiftKeys[$0]!)") } }
    if !unused.isEmpty { print("UNUSED (\(unused.count)):"); unused.prefix(15).forEach { print("   \($0)  \(loaded[$0]!.shape)") } }
    var mismatch: [(String, [Int], [Int])] = []
    for k in sk.intersection(ck) where swiftKeys[k]! != loaded[k]!.shape {
        mismatch.append((k, swiftKeys[k]!, loaded[k]!.shape))
    }
    if !mismatch.isEmpty {
        print("SHAPE MISMATCH (\(mismatch.count)):")
        for (k, a, b) in mismatch.prefix(15) { print("   \(k)\n     swift \(a) vs ckpt \(b)") }
    }
    guard missing.isEmpty, unused.isEmpty, mismatch.isEmpty else { fail("S0 FAILED") }
    do { try model.update(parameters: ModuleParameters.unflattened(loaded), verify: .all) }
    catch { fail("S0 FAILED at update(verify: .all): \(error)") }
    print("✅ S0 PASSED — \(swiftKeys.count) tensors, \(total) params, strict update clean.")
}

/// Primitives: the raw bilinear op, then the modulated-conv family.
func gateS1(_ dir: String, _ w: String) -> Bool {
    print("=== S1 · primitives ===\n")
    let r = GateReport("S1")
    let model = loadedModel(w)
    let dec = model.styleganDecoder

    r.check("bilinear_up2", toNCHW(interpolateBilinear(toNHWC(g(dir, "bilinear_in")), scaleFactor: 2)),
            g(dir, "bilinear_up2_out"), tol: 1e-6)
    r.check("bilinear_down2", toNCHW(interpolateBilinear(toNHWC(g(dir, "bilinear_in")), scaleFactor: 0.5)),
            g(dir, "bilinear_down2_out"), tol: 1e-6)

    let xm = toNHWC(g(dir, "modconv_in"))
    let sm = g(dir, "modconv_style")
    r.check("modconv_demod", toNCHW(dec.styleConv1.modulatedConv(xm, sm)),
            g(dir, "modconv_demod_out"), tol: 1e-5)
    r.check("modconv_plain", toNCHW(dec.toRGB1.modulatedConv(xm, sm)),
            g(dir, "modconv_plain_out"), tol: 1e-5)

    r.check("styleconv", toNCHW(dec.styleConv1(xm, sm, noise: dec.noises[0])),
            g(dir, "styleconv_out"), tol: 1e-5)
    r.check("torgb1", toNCHW(dec.toRGB1(xm, sm, skip: nil)),
            g(dir, "torgb1_out"), tol: 1e-5)
    r.check("torgb_up",
            toNCHW(dec.toRGBs[0](toNHWC(g(dir, "torgb_up_in")), g(dir, "torgb_up_style"),
                                 skip: toNHWC(g(dir, "torgb_up_skip")))),
            g(dir, "torgb_up_out"), tol: 1e-5)
    return r.summarize()
}

/// Blocks: the U-Net ResBlock in both resample directions.
func gateS2(_ dir: String, _ w: String) -> Bool {
    print("=== S2 · blocks ===\n")
    let r = GateReport("S2")
    let model = loadedModel(w)
    r.check("resblock_down", toNCHW(model.convBodyDown[0](toNHWC(g(dir, "resblock_down_in")))),
            g(dir, "resblock_down_out"), tol: 1e-5)
    r.check("resblock_up", toNCHW(model.convBodyUp[0](toNHWC(g(dir, "resblock_up_in")))),
            g(dir, "resblock_up_out"), tol: 1e-5)
    return r.summarize()
}

/// Full model with per-stage taps — every intermediate the oracle dumped, both inputs.
func gateS3(_ dir: String, _ w: String) -> Bool {
    print("=== S3 · full model (per-stage taps) ===\n")
    let r = GateReport("S3")
    let model = loadedModel(w)

    for tag in ["full_rand", "full_face"] {
        print("  — \(tag) —")
        var taps: [String: MLXArray] = [:]
        let image = model(toNHWC(g(dir, "\(tag)_in"))) { name, arr in taps[name] = arr }
        eval(image)

        // Stage taps in pipeline order. style_code is NCHW-free (b, 8192): compare directly.
        var names = ["feat_first"]
        names += (0 ..< 7).map { "down\($0)" }
        names += ["final_conv", "style_code"]
        for i in 0 ..< 7 { names += ["up\(i)", "scale\(i)", "shift\(i)"] }
        names += ["dec_conv1"]
        for l in 0 ..< 7 { names += ["dec_out\(l)", "dec_skip\(l)"] }

        for name in names {
            guard let got = taps[name] else { fail("missing tap \(name)") }
            let want = g(dir, "\(tag)_\(name)")
            let cmp = name == "style_code" ? got : toNCHW(got)
            // Tolerance widens with depth: the decoder compounds ~30 convs and 7 resamples.
            let tol: Float = name.hasPrefix("dec_") ? 5e-4 : 1e-4
            r.check("\(tag).\(name)", cmp, want, tol: tol)
        }
        r.check("\(tag).image", toNCHW(image), g(dir, "\(tag)_image"), tol: 5e-4)

        // The tapped decoder path is a replica of the untapped one — prove they agree.
        let untapped = model(toNHWC(g(dir, "\(tag)_in")))
        eval(untapped)
        let p = parity(untapped, image)
        print("  \(p.maxAbs == 0 ? "✅" : "❌") tapped == untapped forward (max_abs=\(p.maxAbs))")
        if p.maxAbs != 0 { return false }
    }
    return r.summarize()
}

/// e2e at a candidate publish dtype, GPU stream — the dtype-choice gate.
func gateDtype(_ dir: String, _ w: String, dtype: DType, label: String) {
    _ = _unbuffered
    print("=== DTYPE · \(label) e2e (GPU stream) ===\n")
    let model = loadedModel(w)
    let params = model.parameters().mapValues { $0.asType(dtype) }
    model.update(parameters: params)
    eval(model)

    for tag in ["full_rand", "full_face"] {
        let x = toNHWC(g(dir, "\(tag)_in")).asType(dtype)
        let out = model(x)
        eval(out)
        let want = g(dir, "\(tag)_image")
        let got = toNCHW(out.asType(.float32))
        let p = parity(got, want)
        // PSNR in the display [-1,1] → [0,1] domain, clamped like tensor2img.
        let a = clip((got + 1) / 2, min: 0, max: 1)
        let b = clip((want + 1) / 2, min: 0, max: 1)
        let mse = MLX.mean(MLX.square(a - b)).item(Float.self)
        let psnr = mse > 0 ? 10 * log10(1.0 / mse) : Float.infinity
        print(String(format: "  %@  cos=%.6f  rel=%.3e  PSNR=%.2f dB vs fp32 golden",
                     tag, p.cosine, p.relative, psnr))
    }
}

/// Split footprint on the GPU stream. Face restore is fixed-size 512² — no tiling story.
func gateBench(_ w: String) {
    _ = _unbuffered
    print("=== BENCH · split footprint (GPU stream) ===\n")
    let base = physFootprintBytes()
    let model = loadedModel(w)
    MLX.Memory.clearCache()
    let floor = physFootprintBytes()
    print("  post-load floor : \(gb(floor))  → resident ≈ \(gb(floor > base ? floor - base : 0))")
    print("  (weights are 87,143,276 params @ fp32 = 348.6 MB)\n")

    MLX.Memory.clearCache()
    MLX.Memory.peakMemory = 0
    let x = MLXArray.zeros([1, 512, 512, 3], dtype: .float32)
    let t0 = Date()
    let out = model(x)
    eval(out)
    let dt = Date().timeIntervalSince(t0)
    let mlxPeak = MLX.Memory.peakMemory
    let phys = physFootprintBytes()
    print(String(format: "  512x512: MLX peak %@   phys %@   activation ≈ %@   %.2fs",
                 gb(mlxPeak), gb(phys), gb(phys > floor ? phys - floor : 0), dt))
    MLX.Memory.clearCache()
}

func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(max(0, b)) / 1e9) }
func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

let args = Array(CommandLine.arguments.dropFirst())
guard let mode = args.first else {
    print("usage: gfpgan-gate --s0 <weights> | --s1|--s2|--s3|--all <goldens> <weights> "
        + "| --fp16|--bf16 <goldens> <weights> | --bench <weights>")
    exit(2)
}
switch mode {
case "--bench":
    guard args.count >= 2 else { fail("--bench needs a weights path") }
    Device.setDefault(device: .gpu)
    gateBench(args[1])
case "--fp16", "--bf16":
    guard args.count >= 3 else { fail("\(mode) needs <goldens> <weights>") }
    Device.setDefault(device: .gpu)
    gateDtype(args[1], args[2], dtype: mode == "--fp16" ? .float16 : .bfloat16,
              label: mode == "--fp16" ? "fp16" : "bf16")
case "--s0":
    Device.setDefault(device: .cpu)
    guard args.count >= 2 else { fail("--s0 needs a weights path") }
    gateS0(args[1])
case "--s1", "--s2", "--s3", "--all":
    Device.setDefault(device: .cpu)
    guard args.count >= 3 else { fail("\(mode) needs <goldens> <weights>") }
    let (dir, w) = (args[1], args[2])
    var ok = true
    if mode == "--s1" || mode == "--all" { ok = gateS1(dir, w) && ok; print("") }
    if mode == "--s2" || mode == "--all" { ok = gateS2(dir, w) && ok; print("") }
    if mode == "--s3" || mode == "--all" { ok = gateS3(dir, w) && ok }
    if !ok { exit(1) }
default: fail("unknown mode \(mode)")
}

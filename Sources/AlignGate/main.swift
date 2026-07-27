//
//  main.swift
//  mlx-gfpgan-swift / GFPGANAlignGate
//
//  The #43b agreement gate for the substituted alignment seam: Apple Vision (FaceAlign)
//  vs the facexlib ground truth baked once offline by `oracle/gen_align_fixtures.py`.
//
//  A substituted preprocessing signal is a distribution shift against the crops GFPGAN was
//  trained on, so it is measured, not assumed. Metrics per face:
//    - crop-quadrilateral IoU: the 512² crop's preimage in image space under each affine
//      (the SELECTION agreement — what pixels each pipeline feeds the model)
//    - mean 5-point landmark distance, normalized by the interocular distance
//  Gate: every ground-truth face matched, IoU ≥ 0.75. Landmark distance is reported
//  (informational — the FFHQ template absorbs small landmark noise into a similar crop).
//
//  Usage:  swift run gfpgan-align-gate oracle/align_fixtures
//

import CoreGraphics
import Foundation
import ImageIO
import MLXGFPGAN

setvbuf(stdout, nil, _IONBF, 0)

struct FixtureFace: Codable {
    let landmarks5: [[Double]]
    let affine: [[Double]]      // 2x3, image -> 512² crop, cv2 convention
}
struct Fixture: Codable {
    let image: String
    let width: Int
    let height: Int
    let faces: [FixtureFace]
}

func fail(_ msg: String) -> Never { print("❌ \(msg)"); exit(1) }

// MARK: - Polygon IoU (convex quads via Sutherland–Hodgman + shoelace)

func polygonArea(_ pts: [CGPoint]) -> CGFloat {
    guard pts.count >= 3 else { return 0 }
    var a: CGFloat = 0
    for i in 0 ..< pts.count {
        let p = pts[i], q = pts[(i + 1) % pts.count]
        a += p.x * q.y - q.x * p.y
    }
    return abs(a) / 2
}

func clip(_ subject: [CGPoint], edgeA: CGPoint, edgeB: CGPoint) -> [CGPoint] {
    func inside(_ p: CGPoint) -> Bool {
        (edgeB.x - edgeA.x) * (p.y - edgeA.y) - (edgeB.y - edgeA.y) * (p.x - edgeA.x) >= 0
    }
    func intersection(_ p: CGPoint, _ q: CGPoint) -> CGPoint {
        let a1 = edgeB.y - edgeA.y, b1 = edgeA.x - edgeB.x
        let c1 = a1 * edgeA.x + b1 * edgeA.y
        let a2 = q.y - p.y, b2 = p.x - q.x
        let c2 = a2 * p.x + b2 * p.y
        let det = a1 * b2 - a2 * b1
        guard abs(det) > 1e-12 else { return p }
        return CGPoint(x: (b2 * c1 - b1 * c2) / det, y: (a1 * c2 - a2 * c1) / det)
    }
    var out: [CGPoint] = []
    for i in 0 ..< subject.count {
        let cur = subject[i], prev = subject[(i + subject.count - 1) % subject.count]
        if inside(cur) {
            if !inside(prev) { out.append(intersection(prev, cur)) }
            out.append(cur)
        } else if inside(prev) {
            out.append(intersection(prev, cur))
        }
    }
    return out
}

func quadIoU(_ a: [CGPoint], _ b: [CGPoint]) -> CGFloat {
    // Ensure counter-clockwise winding for the clipper.
    func ccw(_ p: [CGPoint]) -> [CGPoint] {
        var s: CGFloat = 0
        for i in 0 ..< p.count {
            let u = p[i], v = p[(i + 1) % p.count]
            s += (v.x - u.x) * (v.y + u.y)
        }
        return s > 0 ? p.reversed() : p
    }
    let pa = ccw(a), pb = ccw(b)
    var inter = pa
    for i in 0 ..< pb.count {
        inter = clip(inter, edgeA: pb[i], edgeB: pb[(i + 1) % pb.count])
        if inter.isEmpty { return 0 }
    }
    let ai = polygonArea(inter)
    let union = polygonArea(pa) + polygonArea(pb) - ai
    return union > 0 ? ai / union : 0
}

// MARK: - Gate

let fixturesRoot = CommandLine.arguments.count > 1 ? CommandLine.arguments[1]
    : "oracle/align_fixtures"
let oracleRoot = URL(fileURLWithPath: fixturesRoot).deletingLastPathComponent()

guard let dirs = try? FileManager.default.contentsOfDirectory(atPath: fixturesRoot).sorted(),
      !dirs.isEmpty else { fail("no fixtures under \(fixturesRoot)") }

let cropCorners = [CGPoint(x: 0, y: 0), CGPoint(x: 512, y: 0),
                   CGPoint(x: 512, y: 512), CGPoint(x: 0, y: 512)]

var allPassed = true
var totalMatched = 0, totalGT = 0

for dir in dirs {
    let base = "\(fixturesRoot)/\(dir)"
    guard let jsonData = FileManager.default.contents(atPath: "\(base)/faces.json"),
          let fixture = try? JSONDecoder().decode(Fixture.self, from: jsonData) else { continue }

    let imgPath = oracleRoot.appendingPathComponent(fixture.image).path
    guard let data = FileManager.default.contents(atPath: imgPath),
          let src = CGImageSourceCreateWithData(data as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        fail("cannot load \(imgPath)")
    }

    let detected: [AlignedFace]
    do { detected = try FaceAlign.detectFaces(in: cg) }
    catch { fail("Vision detection failed on \(dir): \(error)") }

    print("— \(dir) (\(fixture.width)x\(fixture.height)): "
        + "facexlib \(fixture.faces.count) face(s), Vision \(detected.count) —")
    totalGT += fixture.faces.count

    for (gi, gt) in fixture.faces.enumerated() {
        // facexlib affine: crop = A·image; invert to place the crop quad in image space.
        let A = gt.affine
        let t = CGAffineTransform(a: A[0][0], b: A[1][0], c: A[0][1], d: A[1][1],
                                  tx: A[0][2], ty: A[1][2])
        let gtQuad = cropCorners.map { $0.applying(t.inverted()) }
        let gtLm = gt.landmarks5.map { CGPoint(x: $0[0], y: $0[1]) }
        let interocular = hypot(gtLm[1].x - gtLm[0].x, gtLm[1].y - gtLm[0].y)

        // match by landmark centroid distance
        func centroid(_ pts: [CGPoint]) -> CGPoint {
            CGPoint(x: pts.reduce(0) { $0 + $1.x } / CGFloat(pts.count),
                    y: pts.reduce(0) { $0 + $1.y } / CGFloat(pts.count))
        }
        let gc = centroid(gtLm)
        guard let match = detected.min(by: {
            let c0 = centroid($0.landmarks), c1 = centroid($1.landmarks)
            return hypot(c0.x - gc.x, c0.y - gc.y) < hypot(c1.x - gc.x, c1.y - gc.y)
        }) else {
            print("  ❌ face\(gi): no Vision detection to match")
            allPassed = false
            continue
        }
        let mc = centroid(match.landmarks)
        guard hypot(mc.x - gc.x, mc.y - gc.y) < interocular * 2 else {
            print("  ❌ face\(gi): nearest Vision face is \(Int(hypot(mc.x - gc.x, mc.y - gc.y)))px away — unmatched")
            allPassed = false
            continue
        }
        totalMatched += 1

        let ourQuad = cropCorners.map { $0.applying(match.transform.inverted()) }
        let iou = quadIoU(gtQuad, ourQuad)
        let lmDist = zip(gtLm, match.landmarks)
            .map { hypot($0.x - $1.x, $0.y - $1.y) }
            .reduce(0, +) / 5 / interocular
        let ok = iou >= 0.75
        if !ok { allPassed = false }
        print(String(format: "  %@ face%d: crop IoU %.3f   landmark dist %.3f×interocular",
                     ok ? "✅" : "❌", gi, iou, lmDist))

        // our crop, for the eyeball pair with facexlib's face<gi>.png
        if let crop = FaceAlign.warp(cg, transform: match.transform, width: 512, height: 512) {
            let url = URL(fileURLWithPath: "\(base)/face\(gi).vision.png")
            if let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString,
                                                          1, nil) {
                CGImageDestinationAddImage(dest, crop, nil)
                CGImageDestinationFinalize(dest)
            }
        }
    }
}

print("")
print("matched \(totalMatched)/\(totalGT) ground-truth faces")
if allPassed && totalMatched == totalGT {
    print("✅ ALIGN GATE PASSED")
} else {
    print("❌ ALIGN GATE FAILED")
    exit(1)
}

// The pre-norm transformer encoder, ported from EmojiModel.forward in emoji.js.
//
// Precision follows the JS: activations live in Float (its Float32Array
// buffers) while every accumulator is Double (a plain JS number), rounded to
// Float only where the JS stores into a typed array.

import Accelerate
import Foundation

struct ModelConfig: Codable, Sendable {
    let vocab_size: Int
    let n_emoji: Int
    let d_model: Int
    let n_heads: Int
    let n_layers: Int
    let d_ff: Int
    let max_len: Int
}

/// A linear layer stored transposed in Double, so vDSP_mmulD computes
/// x @ W^T with a Double accumulator like the JS inner loop.
struct Linear: Sendable {
    let wt: [Double]      // [dIn, dOut]
    let bias: [Double]
    let dIn: Int
    let dOut: Int

    init(weight: ArraySlice<Float>, bias: ArraySlice<Float>, dIn: Int, dOut: Int) {
        var wt = [Double](repeating: 0, count: dIn * dOut)
        let base = weight.startIndex
        for j in 0..<dOut {
            for i in 0..<dIn { wt[i * dOut + j] = Double(weight[base + j * dIn + i]) }
        }
        self.wt = wt
        self.bias = bias.map(Double.init)
        self.dIn = dIn
        self.dOut = dOut
    }

    func apply(_ x: [Float], rows: Int, into out: inout [Float]) {
        let xd = x.withUnsafeBufferPointer { src -> [Double] in
            var xd = [Double](repeating: 0, count: rows * dIn)
            vDSP_vspdp(src.baseAddress!, 1, &xd, 1, vDSP_Length(rows * dIn))
            return xd
        }
        var yd = [Double](repeating: 0, count: rows * dOut)
        vDSP_mmulD(xd, 1, wt, 1, &yd, 1, vDSP_Length(rows), vDSP_Length(dOut), vDSP_Length(dIn))
        out.withUnsafeMutableBufferPointer { o in
            for r in 0..<rows {
                let yo = r * dOut
                for j in 0..<dOut { o[yo + j] = Float(bias[j] + yd[yo + j]) }
            }
        }
    }
}

struct Norm: Sendable {
    let gamma: [Float]
    let beta: [Float]

    func apply(_ x: inout [Float], d: Int, rows: Int) {
        x.withUnsafeMutableBufferPointer { x in
            for r in 0..<rows {
                let o = r * d
                var mean = 0.0
                for i in 0..<d { mean += Double(x[o + i]) }
                mean /= Double(d)
                var varc = 0.0
                for i in 0..<d { let v = Double(x[o + i]) - mean; varc += v * v }
                let inv = 1 / (varc / Double(d) + 1e-5).squareRoot()
                for i in 0..<d {
                    x[o + i] = Float((Double(x[o + i]) - mean) * inv * Double(gamma[i]) + Double(beta[i]))
                }
            }
        }
    }
}

struct Layer: Sendable {
    let norm1: Norm
    let qkv: Linear
    let proj: Linear
    let norm2: Norm
    let ff0: Linear
    let ff2: Linear
}

/// Abramowitz & Stegun 7.1.26, the same approximation as the JS (not libm's erf).
@inline(__always)
func erfApprox(_ x: Double) -> Double {
    let sign: Double = x < 0 ? -1 : 1
    let x = abs(x)
    let t = 1 / (1 + 0.3275911 * x)
    let y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t * exp(-x * x)
    return sign * y
}

@inline(__always)
func gelu(_ x: Double) -> Double {
    0.5 * x * (1 + erfApprox(x / 2.0.squareRoot()))
}

struct Encoder: Sendable {
    let cfg: ModelConfig
    let tokenEmb: [Float]
    let posEmb: [Float]
    let layers: [Layer]
    let norm: Norm
    let head: Linear

    init(cfg: ModelConfig, tensors: [String: ArraySlice<Float>]) throws {
        func t(_ name: String) throws -> ArraySlice<Float> {
            guard let v = tensors[name] else { throw EmojiModelError.missingTensor(name) }
            return v
        }
        let D = cfg.d_model, F = cfg.d_ff
        self.cfg = cfg
        tokenEmb = Array(try t("token_emb.weight"))
        posEmb = Array(try t("pos_emb.weight"))
        layers = try (0..<cfg.n_layers).map { l in
            let p = "layers.\(l)."
            return Layer(
                norm1: Norm(gamma: Array(try t(p + "norm1.weight")), beta: Array(try t(p + "norm1.bias"))),
                qkv: Linear(weight: try t(p + "qkv.weight"), bias: try t(p + "qkv.bias"), dIn: D, dOut: 3 * D),
                proj: Linear(weight: try t(p + "proj.weight"), bias: try t(p + "proj.bias"), dIn: D, dOut: D),
                norm2: Norm(gamma: Array(try t(p + "norm2.weight")), beta: Array(try t(p + "norm2.bias"))),
                ff0: Linear(weight: try t(p + "ff.0.weight"), bias: try t(p + "ff.0.bias"), dIn: D, dOut: F),
                ff2: Linear(weight: try t(p + "ff.2.weight"), bias: try t(p + "ff.2.bias"), dIn: F, dOut: D)
            )
        }
        norm = Norm(gamma: Array(try t("norm.weight")), beta: Array(try t("norm.bias")))
        head = Linear(weight: try t("head.weight"), bias: try t("head.bias"), dIn: D, dOut: cfg.n_emoji)
    }

    /// Sigmoid probabilities over the emoji vocabulary for one token sequence.
    func forward(_ ids: [Int32]) -> [Float] {
        let D = cfg.d_model, H = cfg.n_heads, F = cfg.d_ff
        let dh = D / H
        let scale = 1 / Double(dh).squareRoot()
        let T = ids.count

        var x = [Float](repeating: 0, count: T * D)
        for p in 0..<T {
            let tokOff = Int(ids[p]) * D, posOff = p * D, o = p * D
            for i in 0..<D { x[o + i] = tokenEmb[tokOff + i] + posEmb[posOff + i] }
        }

        var h = [Float](repeating: 0, count: T * D)
        var qkv = [Float](repeating: 0, count: T * 3 * D)
        var att = [Float](repeating: 0, count: T * D)
        var projd = [Float](repeating: 0, count: T * D)
        var ff1 = [Float](repeating: 0, count: T * F)
        var ff2 = [Float](repeating: 0, count: T * D)
        var probs = [Double](repeating: 0, count: T)

        for layer in layers {
            h = x
            layer.norm1.apply(&h, d: D, rows: T)
            layer.qkv.apply(h, rows: T, into: &qkv)
            attention(qkv: qkv, att: &att, probs: &probs, T: T, D: D, H: H, dh: dh, scale: scale)

            layer.proj.apply(att, rows: T, into: &projd)
            for i in 0..<(T * D) { x[i] += projd[i] }

            h = x
            layer.norm2.apply(&h, d: D, rows: T)
            layer.ff0.apply(h, rows: T, into: &ff1)
            for i in 0..<(T * F) { ff1[i] = Float(gelu(Double(ff1[i]))) }
            layer.ff2.apply(ff1, rows: T, into: &ff2)
            for i in 0..<(T * D) { x[i] += ff2[i] }
        }

        norm.apply(&x, d: D, rows: T)

        var pooled = [Float](repeating: 0, count: D)
        for p in 0..<T { for i in 0..<D { pooled[i] += x[p * D + i] } }
        for i in 0..<D { pooled[i] /= Float(T) }

        var logits = [Float](repeating: 0, count: cfg.n_emoji)
        head.apply(pooled, rows: 1, into: &logits)
        return logits.map { Float(1 / (1 + exp(-Double($0)))) }
    }

    private func attention(qkv: [Float], att: inout [Float], probs: inout [Double],
                           T: Int, D: Int, H: Int, dh: Int, scale: Double) {
        qkv.withUnsafeBufferPointer { qkv in
            att.withUnsafeMutableBufferPointer { att in
                probs.withUnsafeMutableBufferPointer { probs in
                    for head in 0..<H {
                        let hOff = head * dh
                        for q in 0..<T {
                            let qo = q * 3 * D + hOff
                            var mx = -Double.infinity
                            for k in 0..<T {
                                let ko = k * 3 * D + D + hOff
                                var dot = 0.0
                                for i in 0..<dh { dot += Double(qkv[qo + i]) * Double(qkv[ko + i]) }
                                dot *= scale
                                probs[k] = dot
                                if dot > mx { mx = dot }
                            }
                            var sum = 0.0
                            for k in 0..<T { probs[k] = exp(probs[k] - mx); sum += probs[k] }
                            let ao = q * D + hOff
                            for i in 0..<dh { att[ao + i] = 0 }
                            for k in 0..<T {
                                let w = probs[k] / sum, vo = k * 3 * D + 2 * D + hOff
                                // `att[..] += w * v` in JS: Double math, stored back to Float32.
                                for i in 0..<dh { att[ao + i] = Float(Double(att[ao + i]) + w * Double(qkv[vo + i])) }
                            }
                        }
                    }
                }
            }
        }
    }
}

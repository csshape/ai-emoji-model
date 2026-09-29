// On-device emoji suggestions: a Swift port of report/model/emoji.js.
// Parity with the JS is tested against app/fixtures/parity.json; if you change
// the JS, change this and rerun app/sync_model.sh and `swift test`.

import Foundation

public struct Suggestion: Hashable, Sendable {
    public let emoji: String
    public let score: Double
}

public enum EmojiModelError: Error {
    case missingResource(String)
    case missingTensor(String)
    case badWeights
    case badKeywords
}

private struct ModelMeta: Decodable {
    struct Tensor: Decodable {
        let name: String
        let shape: [Int]
        let offset: Int
    }
    let config: ModelConfig
    let tensors: [Tensor]
    let pieces: [String]
    let scores: [Double]
    let emoji: [String]
    let prior: [Double]?
}

/// Immutable after loading, so one instance can serve concurrent callers.
public final class EmojiModel: Sendable {
    public static let maxEmoji = 5
    public static let absThreshold = 0.08
    public static let relRatio = 0.35
    public static let priorAlpha = 0.25
    // The model learns mood; naming a thing is closer to a lookup, and Unicode
    // already wrote that lookup down.
    public static let keywordWeight = 8.0

    public let emoji: [String]
    public let parameterCount: Int
    private let encoder: Encoder
    private let tokenizer: SpTokenizer
    private let prior: [Float]?
    private let keywords: KeywordTable
    private let emojiIndex: [UTF16Units: Int]

    public init(meta metaData: Data, weights: Data, keywords keywordData: Data?) throws {
        let meta = try JSONDecoder().decode(ModelMeta.self, from: metaData)
        guard weights.count % 4 == 0 else { throw EmojiModelError.badWeights }
        // The .bin is float32 little-endian, which is also every Apple CPU's order.
        var all = [Float](repeating: 0, count: weights.count / 4)
        _ = all.withUnsafeMutableBytes { weights.copyBytes(to: $0) }
        var tensors: [String: ArraySlice<Float>] = [:]
        var count = 0
        for spec in meta.tensors {
            let size = spec.shape.reduce(1, *)
            guard spec.offset + size <= all.count else { throw EmojiModelError.badWeights }
            tensors[spec.name] = all[spec.offset..<(spec.offset + size)]
            count += size
        }
        encoder = try Encoder(cfg: meta.config, tensors: tensors)
        tokenizer = SpTokenizer(pieces: meta.pieces, scores: meta.scores)
        emoji = meta.emoji
        parameterCount = count
        prior = meta.prior.map { $0.map(Float.init) }   // Float32Array.from(meta.prior)
        keywords = try keywordData.map(KeywordTable.init(json:)) ?? KeywordTable()
        var index: [UTF16Units: Int] = [:]
        for (i, e) in meta.emoji.enumerated() { index[Array(e.utf16)] = i }
        emojiIndex = index
    }

    /// Loads model.json, model.bin and keywords.json from the package's resources.
    public static func load() throws -> EmojiModel {
        func url(_ name: String, _ ext: String) throws -> URL {
            guard let u = Bundle.module.url(forResource: name, withExtension: ext)
            else { throw EmojiModelError.missingResource("\(name).\(ext)") }
            return u
        }
        let keywords = try? Data(contentsOf: url("keywords", "json"))
        return try EmojiModel(
            meta: Data(contentsOf: url("model", "json")),
            weights: Data(contentsOf: url("model", "bin"), options: .mappedIfSafe),
            keywords: keywords
        )
    }

    /// Token ids after cleaning, BOS included.
    public func tokenize(_ text: String) -> [Int32] {
        tokenizer.encode(cleanText(text), maxLength: encoder.cfg.max_len)
    }

    /// Raw sigmoid outputs, before prior correction and keyword boost.
    public func probabilities(_ text: String) -> [Float] {
        encoder.forward(tokenize(text))
    }

    public func predict(_ text: String,
                        alpha: Double = EmojiModel.priorAlpha,
                        keywordWeight: Double = EmojiModel.keywordWeight) -> [Suggestion] {
        var p = probabilities(text)
        let n = p.count
        var flag: String?
        if keywordWeight > 0, !keywords.isEmpty {
            var flagScore = 0.0
            for (e, score) in keywords.hits(cleanText(text)) {
                if let j = emojiIndex[Array(e.utf16)] {
                    p[j] = Float(Double(p[j]) + keywordWeight * score)
                } else if isFlag(e), score > flagScore
                            || (score == flagScore && flag.map { e.utf16.lexicographicallyPrecedes($0.utf16) } == true) {
                    // Flags are outside the model's vocabulary; a named country
                    // goes first. Ties go to the smaller string in UTF-16 order
                    // (JS `<`), not to table order.
                    flag = e
                    flagScore = score
                }
            }
        }

        // Array.prototype.sort is stable, so ties keep index order.
        var order = Array(0..<n)
        if let prior, alpha > 0 {
            let adj = (0..<n).map { Double(p[$0]) / pow(max(Double(prior[$0]), 1e-6), alpha) }
            order.sort { adj[$0] != adj[$1] ? adj[$0] > adj[$1] : $0 < $1 }
        } else {
            order.sort { p[$0] != p[$1] ? p[$0] > p[$1] : $0 < $1 }
        }
        order = Array(order.prefix(Self.maxEmoji))

        let top = Double(p[order[0]])
        var out = [Suggestion(emoji: emoji[order[0]], score: top)]
        for k in order.dropFirst() {
            let s = Double(p[k])
            if s >= Self.absThreshold, s >= Self.relRatio * top {
                out.append(Suggestion(emoji: emoji[k], score: s))
            }
        }
        if let flag { out.insert(Suggestion(emoji: flag, score: top), at: 0) }
        return Array(out.prefix(Self.maxEmoji))
    }
}

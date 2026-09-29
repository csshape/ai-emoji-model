// SentencePiece unigram (Viterbi over the piece lattice), ported from emoji.js.

let padID: Int32 = 0
let unkID: Int32 = 1
let bosID: Int32 = 2
private let spaceMark: UInt16 = 0x2581   // "▁"

struct SpTokenizer: Sendable {
    // Keyed by UTF-16 units, not String: the JS slices with `substr` on code
    // units, and Swift String equality would also merge canonically equivalent
    // pieces that the JS Map keeps apart.
    private let index: [UTF16Units: Int32]
    private let scores: [Double]
    private let maxLen: Int
    private let unkPenalty: Double

    init(pieces: [String], scores: [Double]) {
        var index = [UTF16Units: Int32](minimumCapacity: pieces.count)
        var maxLen = 1
        for (i, piece) in pieces.enumerated() {
            let units = Array(piece.utf16)
            index[units] = Int32(i)          // later duplicates win, as with Map.set
            maxLen = max(maxLen, units.count)
        }
        self.index = index
        self.scores = scores
        self.maxLen = maxLen
        self.unkPenalty = (scores.min() ?? 0) - 10
    }

    /// Returns BOS plus piece ids, truncated to `maxLength` (no padding: the
    /// encoder masks padding away, so it is simply not materialised).
    func encode(_ text: String, maxLength: Int) -> [Int32] {
        var s: UTF16Units = [spaceMark]
        for u in normalizeForSp(text) { s.append(u == 0x20 ? spaceMark : u) }
        let n = s.count
        var best = [Double](repeating: -.infinity, count: n + 1)
        var from = [Int](repeating: -1, count: n + 1)
        var pieceAt = [Int32](repeating: -1, count: n + 1)
        best[0] = 0

        for i in 0..<n {
            if best[i] == -.infinity { continue }
            let limit = min(maxLen, n - i)
            var matched = false
            for len in stride(from: limit, through: 1, by: -1) {
                guard let id = index[Array(s[i..<(i + len)])] else { continue }
                matched = true
                let cand = best[i] + scores[Int(id)]
                if cand > best[i + len] {
                    best[i + len] = cand
                    from[i + len] = i
                    pieceAt[i + len] = id
                }
            }
            if !matched {
                // Unknown character: consume one code unit as <unk>, so an
                // unknown astral character becomes two <unk> tokens, as in JS.
                let cand = best[i] + unkPenalty
                if cand > best[i + 1] {
                    best[i + 1] = cand
                    from[i + 1] = i
                    pieceAt[i + 1] = unkID
                }
            }
        }

        var out: [Int32] = []
        var i = n
        while i > 0 {
            if from[i] < 0 { break }
            out.append(pieceAt[i])
            i = from[i]
        }
        out.reverse()
        return Array(([bosID] + out).prefix(maxLength))
    }
}

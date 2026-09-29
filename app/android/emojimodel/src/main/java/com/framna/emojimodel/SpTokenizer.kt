package com.framna.emojimodel

internal const val PAD_ID = 0
internal const val UNK_ID = 1
internal const val BOS_ID = 2
private const val SPACE = "▁"

/** SentencePiece unigram: Viterbi over the piece lattice, as in emoji.js. */
internal class SpTokenizer(pieces: List<String>, private val scores: DoubleArray) {
    private val index = HashMap<String, Int>(pieces.size * 2)
    private val maxPieceLen: Int
    private val unkPenalty: Double = scores.min() - 10

    init {
        var m = 1
        // Later duplicates win, like Map.set in JS.
        pieces.forEachIndexed { i, p ->
            index[p] = i
            if (p.length > m) m = p.length
        }
        maxPieceLen = m
    }

    /** Returns BOS + pieces, truncated to [maxLen]; unpadded since the model drops padding anyway. */
    fun encode(text: String, maxLen: Int): IntArray {
        val s = SPACE + normalizeForSp(text).split(" ").joinToString(SPACE)
        val n = s.length
        val best = DoubleArray(n + 1) { Double.NEGATIVE_INFINITY }
        val from = IntArray(n + 1) { -1 }
        val pieceAt = IntArray(n + 1) { -1 }
        best[0] = 0.0

        for (i in 0 until n) {
            if (best[i] == Double.NEGATIVE_INFINITY) continue
            val limit = minOf(maxPieceLen, n - i)
            var matched = false
            // Lengths are UTF-16 code units, the same unit JS substr counts in.
            for (len in limit downTo 1) {
                val id = index[s.substring(i, i + len)] ?: continue
                matched = true
                val cand = best[i] + scores[id]
                if (cand > best[i + len]) {
                    best[i + len] = cand
                    from[i + len] = i
                    pieceAt[i + len] = id
                }
            }
            if (!matched) {
                // Unknown character: consume one code unit as <unk>, half a surrogate pair included.
                val cand = best[i] + unkPenalty
                if (cand > best[i + 1]) {
                    best[i + 1] = cand
                    from[i + 1] = i
                    pieceAt[i + 1] = UNK_ID
                }
            }
        }

        val out = ArrayList<Int>()
        var i = n
        while (i > 0) {
            if (from[i] < 0) break
            out.add(pieceAt[i])
            i = from[i]
        }
        out.reverse()

        val len = minOf(out.size + 1, maxLen)
        val ids = IntArray(len)
        ids[0] = BOS_ID
        for (k in 1 until len) ids[k] = out[k - 1]
        return ids
    }
}

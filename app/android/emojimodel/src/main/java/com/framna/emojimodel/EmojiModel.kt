package com.framna.emojimodel

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

const val MAX_EMOJI = 5
const val ABS_THRESHOLD = 0.08
const val REL_RATIO = 0.35
const val PRIOR_ALPHA = 0.25
const val KEYWORD_WEIGHT = 8.0

private val KEY_WORD_RE = Regex("[a-zA-ZæøåÆØÅ]{2,}")

data class Suggestion(val emoji: String, val score: Float)

data class ModelConfig(
    val dModel: Int,
    val nHeads: Int,
    val nLayers: Int,
    val dFf: Int,
    val maxLen: Int,
    val nEmoji: Int,
)

/**
 * Kotlin port of report/model/emoji.js. Pure JVM so it runs in unit tests; see
 * [EmojiModelAssets] for loading it on Android.
 *
 * Numeric types follow the JS exactly: buffers that are Float32Array there are
 * FloatArray here, and every accumulator that is a plain JS number is a Double that
 * is only rounded to Float when stored, as JS does on assignment into a Float32Array.
 */
class EmojiModel(modelJson: String, weights: ByteArray, keywordsJson: String? = null) {
    val config: ModelConfig
    val emoji: List<String>
    val parameterCount: Int
    private val prior: FloatArray?
    private val keywords: LinkedHashMap<String, List<String>>
    private val emojiIndex: Map<String, Int>
    private val tok: SpTokenizer
    private val t: Map<String, FloatArray>

    init {
        val meta = Json.parseToJsonElement(modelJson).jsonObject
        val cfg = meta.getValue("config").jsonObject
        config = ModelConfig(
            dModel = cfg.int("d_model"),
            nHeads = cfg.int("n_heads"),
            nLayers = cfg.int("n_layers"),
            dFf = cfg.int("d_ff"),
            maxLen = cfg.int("max_len"),
            nEmoji = cfg.int("n_emoji"),
        )
        emoji = meta.getValue("emoji").jsonArray.map { it.jsonPrimitive.content }
        prior = meta["prior"]?.let { p ->
            val arr = p.jsonArray
            FloatArray(arr.size) { arr[it].jsonPrimitive.double.toFloat() }
        }
        val scores = meta.getValue("scores").jsonArray.let { a -> DoubleArray(a.size) { a[it].jsonPrimitive.double } }
        tok = SpTokenizer(meta.getValue("pieces").jsonArray.map { it.jsonPrimitive.content }, scores)

        val all = FloatArray(weights.size / 4)
        ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(all)
        parameterCount = all.size
        t = meta.getValue("tensors").jsonArray.associate { el ->
            val spec = el.jsonObject
            val size = spec.getValue("shape").jsonArray.fold(1) { a, d -> a * d.jsonPrimitive.int }
            val offset = spec.int("offset")
            spec.getValue("name").jsonPrimitive.content to all.copyOfRange(offset, offset + size)
        }

        // kotlinx keeps JSON key order, which is what Object.entries walks (the table
        // has no integer-like keys, the one case where JS would reorder). Order matters
        // because each emoji's score is a floating-point sum over keys.
        keywords = LinkedHashMap()
        if (keywordsJson != null) {
            for ((k, v) in Json.parseToJsonElement(keywordsJson).jsonObject) {
                keywords[k] = v.jsonArray.map { it.jsonPrimitive.content }
            }
        }
        emojiIndex = HashMap<String, Int>().also { m -> emoji.forEachIndexed { i, e -> m[e] = i } }
    }

    /** Token ids fed to the encoder, BOS included, for parity checks. */
    fun tokenize(text: String): IntArray = tok.encode(cleanText(text), config.maxLen)

    fun probs(text: String): FloatArray = forward(tokenize(text))

    fun predict(
        text: String,
        alpha: Double = PRIOR_ALPHA,
        keywordWeight: Double = KEYWORD_WEIGHT,
    ): List<Suggestion> {
        val p = probs(text)
        val n = p.size
        var flag: String? = null
        if (keywordWeight > 0 && keywords.isNotEmpty()) {
            var flagScore = 0.0
            for ((e, score) in keywordHits(cleanText(text))) {
                val j = emojiIndex[e]
                if (j != null) {
                    p[j] = (p[j] + keywordWeight * score).toFloat()
                } else if (isFlag(e) && (score > flagScore || (score == flagScore && flag != null && e < flag))) {
                    // Flags are outside the model's vocabulary: a named country goes first.
                    // Ties go to the smaller string; String.compareTo compares UTF-16 code
                    // units like JS `<`. JS `e < null` is false, hence the null check.
                    flag = e
                    flagScore = score
                }
            }
        }

        var order = (0 until n).toList()
        // List.sortedWith is stable, like Array.prototype.sort; the comparator uses the
        // sign of the difference, as the JS one does.
        order = if (prior != null && alpha > 0) {
            val adj = DoubleArray(n) { p[it] / max(prior[it].toDouble(), 1e-6).pow(alpha) }
            order.sortedWith { a, b -> sign(adj[b] - adj[a]) }
        } else {
            order.sortedWith { a, b -> sign(p[b].toDouble() - p[a].toDouble()) }
        }
        order = order.take(MAX_EMOJI)

        val top = p[order[0]]
        val out = ArrayList<Suggestion>()
        out.add(Suggestion(emoji[order[0]], top))
        for (k in 1 until order.size) {
            val s = p[order[k]]
            if (s >= ABS_THRESHOLD && s >= REL_RATIO * top) out.add(Suggestion(emoji[order[k]], s))
        }
        if (flag != null) out.add(0, Suggestion(flag, top))
        return out.take(MAX_EMOJI)
    }

    private fun keywordHits(text: String): LinkedHashMap<String, Double> {
        val hits = LinkedHashMap<String, Double>()
        val words = KEY_WORD_RE.findAll(text.lowercase()).map { it.value }.toList()
        for ((key, emojis) in keywords) {
            if (words.none { it.startsWith(key) }) continue
            for (e in emojis) hits[e] = (hits[e] ?: 0.0) + key.length * 0.01
        }
        return hits
    }

    private fun forward(ids: IntArray): FloatArray {
        val d = config.dModel
        val h = config.nHeads
        val f = config.dFf
        val dh = d / h
        val scale = 1 / sqrt(dh.toDouble())
        val len = ids.size // padding is masked everywhere, so it is never materialised

        val tokEmb = t.getValue("token_emb.weight")
        val posEmb = t.getValue("pos_emb.weight")
        val x = FloatArray(len * d)
        for (p in 0 until len) {
            val tokOff = ids[p] * d
            val o = p * d
            // A single float+float rounded once is the same whether done in float or double.
            for (i in 0 until d) x[o + i] = tokEmb[tokOff + i] + posEmb[o + i]
        }

        val hb = FloatArray(len * d)
        val qkv = FloatArray(len * 3 * d)
        val att = FloatArray(len * d)
        val projd = FloatArray(len * d)
        val ff1 = FloatArray(len * f)
        val ff2 = FloatArray(len * d)
        val probs = DoubleArray(len)

        for (l in 0 until config.nLayers) {
            val pre = "layers.$l."
            x.copyInto(hb)
            layerNorm(hb, t.getValue(pre + "norm1.weight"), t.getValue(pre + "norm1.bias"), d, len)
            linear(hb, t.getValue(pre + "qkv.weight"), t.getValue(pre + "qkv.bias"), len, d, 3 * d, qkv)

            for (head in 0 until h) {
                val hOff = head * dh
                for (q in 0 until len) {
                    val qo = q * 3 * d + hOff
                    var mx = Double.NEGATIVE_INFINITY
                    for (k in 0 until len) {
                        val ko = k * 3 * d + d + hOff
                        var dot = 0.0
                        for (i in 0 until dh) dot += qkv[qo + i].toDouble() * qkv[ko + i]
                        dot *= scale
                        probs[k] = dot
                        if (dot > mx) mx = dot
                    }
                    var sum = 0.0
                    for (k in 0 until len) {
                        probs[k] = exp(probs[k] - mx)
                        sum += probs[k]
                    }
                    val ao = q * d + hOff
                    for (i in 0 until dh) att[ao + i] = 0f
                    for (k in 0 until len) {
                        val w = probs[k] / sum
                        val vo = k * 3 * d + 2 * d + hOff
                        // `att[i] += w * v` in JS rounds to float32 after every step.
                        for (i in 0 until dh) att[ao + i] = (att[ao + i] + w * qkv[vo + i]).toFloat()
                    }
                }
            }

            linear(att, t.getValue(pre + "proj.weight"), t.getValue(pre + "proj.bias"), len, d, d, projd)
            for (i in 0 until len * d) x[i] += projd[i]

            x.copyInto(hb)
            layerNorm(hb, t.getValue(pre + "norm2.weight"), t.getValue(pre + "norm2.bias"), d, len)
            linear(hb, t.getValue(pre + "ff.0.weight"), t.getValue(pre + "ff.0.bias"), len, d, f, ff1)
            for (i in 0 until len * f) ff1[i] = gelu(ff1[i].toDouble()).toFloat()
            linear(ff1, t.getValue(pre + "ff.2.weight"), t.getValue(pre + "ff.2.bias"), len, f, d, ff2)
            for (i in 0 until len * d) x[i] += ff2[i]
        }

        layerNorm(x, t.getValue("norm.weight"), t.getValue("norm.bias"), d, len)

        // pooled is a Float32Array in JS, so the running sum is float32 too.
        val pooled = FloatArray(d)
        for (p in 0 until len) for (i in 0 until d) pooled[i] += x[p * d + i]
        for (i in 0 until d) pooled[i] /= len.toFloat()

        val n = config.nEmoji
        val logits = FloatArray(n)
        linear(pooled, t.getValue("head.weight"), t.getValue("head.bias"), 1, d, n, logits)
        return FloatArray(n) { (1.0 / (1.0 + exp(-logits[it].toDouble()))).toFloat() }
    }
}

private fun JsonObject.int(key: String) = getValue(key).jsonPrimitive.int

private fun sign(v: Double): Int = if (v < 0) -1 else if (v > 0) 1 else 0

internal fun isFlag(e: String): Boolean {
    if (e.codePointCount(0, e.length) != 2) return false
    var i = 0
    while (i < e.length) {
        val c = e.codePointAt(i)
        if (c < 0x1F1E6 || c > 0x1F1FF) return false
        i += Character.charCount(c)
    }
    return true
}

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 -- the approximation emoji.js uses,
// kept instead of an exact erf so both ports round the same way.
private fun erf(x0: Double): Double {
    val sign = if (x0 < 0) -1 else 1
    val x = abs(x0)
    val t = 1 / (1 + 0.3275911 * x)
    val y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
        0.284496736) * t + 0.254829592) * t * exp(-x * x)
    return sign * y
}

private val SQRT2 = sqrt(2.0)

private fun gelu(x: Double) = 0.5 * x * (1 + erf(x / SQRT2))

private fun layerNorm(x: FloatArray, gamma: FloatArray, beta: FloatArray, d: Int, rows: Int) {
    for (r in 0 until rows) {
        val o = r * d
        var mean = 0.0
        for (i in 0 until d) mean += x[o + i]
        mean /= d
        var varc = 0.0
        for (i in 0 until d) {
            val v = x[o + i] - mean
            varc += v * v
        }
        val inv = 1 / sqrt(varc / d + 1e-5)
        for (i in 0 until d) x[o + i] = ((x[o + i] - mean) * inv * gamma[i] + beta[i]).toFloat()
    }
}

// y[rows, dOut] = x[rows, dIn] @ W[dOut, dIn]^T + b, accumulated in double as in JS.
private fun linear(x: FloatArray, w: FloatArray, b: FloatArray?, rows: Int, dIn: Int, dOut: Int, out: FloatArray) {
    for (r in 0 until rows) {
        val xo = r * dIn
        val yo = r * dOut
        for (j in 0 until dOut) {
            var acc = b?.get(j)?.toDouble() ?: 0.0
            val wo = j * dIn
            for (i in 0 until dIn) acc += x[xo + i].toDouble() * w[wo + i]
            out[yo + j] = acc.toFloat()
        }
    }
}

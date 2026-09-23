// Runs the emoji encoder in the browser -- no ML runtime, just typed arrays.
// Mirrors emoji_model/model.py and emoji_model/infer.py; if you change one,
// change the other and rerun `node report/model/check.mjs`.

const PAD_ID = 0, UNK_ID = 1, BOS_ID = 2;
const SPACE = "▁";
const MAX_EMOJI = 5;
const ABS_THRESHOLD = 0.08, REL_RATIO = 0.35, PRIOR_ALPHA = 0.25;
// The model learns mood; naming a thing is closer to a lookup, and Unicode
// already wrote that lookup down. Mirrors emoji_model/keywords.py.
const KEYWORD_WEIGHT = 8.0;
const KEY_WORD_RE = /[a-zA-ZæøåÆØÅ]{2,}/g;

// --- text normalisation -----------------------------------------------------
// SentencePiece was trained with normalization_rule_name="nmt_nfkc_cf", i.e.
// NFKC plus case folding plus NMT whitespace cleanup.
const URL_RE = /https?:\/\/\S+|www\.\S+/g;
const MENTION_RE = /@\w+/g;
const SUBREDDIT_RE = /\/?r\/\w+/g;
const REPEAT_RE = /(.)\1{3,}/g;

export function cleanText(text) {
  let t = text.normalize("NFKC");
  t = t.replace(URL_RE, " <url> ");
  t = t.replace(MENTION_RE, " <user> ");
  t = t.replace(SUBREDDIT_RE, " <sub> ");
  t = t.replace(REPEAT_RE, "$1$1$1");
  t = t.replace(/\s+/g, " ");
  return t.trim();
}

function normalizeForSp(text) {
  return text.normalize("NFKC").toLowerCase().replace(/\s+/g, " ").trim();
}

// --- SentencePiece unigram (Viterbi over the piece lattice) ------------------
class SpTokenizer {
  constructor(pieces, scores) {
    this.pieces = pieces;
    this.scores = scores;
    this.index = new Map();
    this.maxLen = 1;
    for (let i = 0; i < pieces.length; i++) {
      this.index.set(pieces[i], i);
      if (pieces[i].length > this.maxLen) this.maxLen = pieces[i].length;
    }
    this.unkPenalty = Math.min(...scores) - 10;
  }

  encode(text, maxLen) {
    const s = SPACE + normalizeForSp(text).split(" ").join(SPACE);
    const n = s.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const from = new Int32Array(n + 1).fill(-1);
    const pieceAt = new Int32Array(n + 1).fill(-1);
    best[0] = 0;

    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      const limit = Math.min(this.maxLen, n - i);
      let matched = false;
      for (let len = limit; len >= 1; len--) {
        const id = this.index.get(s.substr(i, len));
        if (id === undefined) continue;
        matched = true;
        const cand = best[i] + this.scores[id];
        if (cand > best[i + len]) {
          best[i + len] = cand;
          from[i + len] = i;
          pieceAt[i + len] = id;
        }
      }
      if (!matched) {
        // Unknown character: consume one code unit as <unk>.
        const cand = best[i] + this.unkPenalty;
        if (cand > best[i + 1]) {
          best[i + 1] = cand;
          from[i + 1] = i;
          pieceAt[i + 1] = UNK_ID;
        }
      }
    }

    const out = [];
    for (let i = n; i > 0; i = from[i]) {
      if (from[i] < 0) break;
      out.push(pieceAt[i]);
    }
    out.reverse();

    const ids = [BOS_ID, ...out].slice(0, maxLen);
    const padded = new Int32Array(maxLen);
    for (let i = 0; i < ids.length; i++) padded[i] = ids[i];
    return { ids: padded, length: ids.length };
  }
}

// --- math -------------------------------------------------------------------
function erf(x) {
  // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return sign * y;
}
const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));

function layerNorm(x, gamma, beta, d, rows) {
  for (let r = 0; r < rows; r++) {
    const o = r * d;
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[o + i];
    mean /= d;
    let varc = 0;
    for (let i = 0; i < d; i++) { const v = x[o + i] - mean; varc += v * v; }
    const inv = 1 / Math.sqrt(varc / d + 1e-5);
    for (let i = 0; i < d; i++) x[o + i] = (x[o + i] - mean) * inv * gamma[i] + beta[i];
  }
}

// y[rows, dOut] = x[rows, dIn] @ W[dOut, dIn]^T + b
function linear(x, W, b, rows, dIn, dOut, out) {
  for (let r = 0; r < rows; r++) {
    const xo = r * dIn, yo = r * dOut;
    for (let j = 0; j < dOut; j++) {
      let acc = b ? b[j] : 0;
      const wo = j * dIn;
      for (let i = 0; i < dIn; i++) acc += x[xo + i] * W[wo + i];
      out[yo + j] = acc;
    }
  }
}

// --- model ------------------------------------------------------------------
function keywordHits(text, table) {
  const hits = new Map();
  const words = text.toLowerCase().match(KEY_WORD_RE) || [];
  for (const [key, emojis] of Object.entries(table)) {
    if (!words.some(w => w.startsWith(key))) continue;
    for (const e of emojis) hits.set(e, (hits.get(e) ?? 0) + key.length * 0.01);
  }
  return hits;
}

export class EmojiModel {
  constructor(meta, buffer, keywords) {
    this.cfg = meta.config;
    this.emoji = meta.emoji;
    this.prior = meta.prior ? Float32Array.from(meta.prior) : null;
    this.keywords = keywords ?? {};
    this.tok = new SpTokenizer(meta.pieces, meta.scores);
    const all = new Float32Array(buffer);
    this.t = {};
    for (const spec of meta.tensors) {
      const size = spec.shape.reduce((a, b) => a * b, 1);
      this.t[spec.name] = all.subarray(spec.offset, spec.offset + size);
    }
  }

  static async load(base, keywordsUrl = "./model/keywords.json") {
    const [meta, buf, kw] = await Promise.all([
      fetch(`${base}.json`).then(r => r.json()),
      fetch(`${base}.bin`).then(r => r.arrayBuffer()),
      fetch(keywordsUrl).then(r => r.json()).catch(() => ({})),
    ]);
    return new EmojiModel(meta, buf, kw);
  }

  forward(ids, length) {
    const { d_model: D, n_heads: H, d_ff: F, n_layers: L } = this.cfg;
    const dh = D / H, scale = 1 / Math.sqrt(dh);
    const T = length;                     // padding is masked everywhere, so drop it
    const t = this.t;

    let x = new Float32Array(T * D);
    for (let p = 0; p < T; p++) {
      const tokOff = ids[p] * D, posOff = p * D, o = p * D;
      for (let i = 0; i < D; i++) x[o + i] = t["token_emb.weight"][tokOff + i] + t["pos_emb.weight"][posOff + i];
    }

    const h = new Float32Array(T * D);
    const qkv = new Float32Array(T * 3 * D);
    const att = new Float32Array(T * D);
    const projd = new Float32Array(T * D);
    const ff1 = new Float32Array(T * F);
    const ff2 = new Float32Array(T * D);
    const probs = new Float64Array(T);

    for (let l = 0; l < L; l++) {
      const p = `layers.${l}.`;
      h.set(x);
      layerNorm(h, t[p + "norm1.weight"], t[p + "norm1.bias"], D, T);
      linear(h, t[p + "qkv.weight"], t[p + "qkv.bias"], T, D, 3 * D, qkv);

      for (let head = 0; head < H; head++) {
        const hOff = head * dh;
        for (let q = 0; q < T; q++) {
          const qo = q * 3 * D + hOff;
          let max = -Infinity;
          for (let k = 0; k < T; k++) {
            const ko = k * 3 * D + D + hOff;
            let dot = 0;
            for (let i = 0; i < dh; i++) dot += qkv[qo + i] * qkv[ko + i];
            dot *= scale;
            probs[k] = dot;
            if (dot > max) max = dot;
          }
          let sum = 0;
          for (let k = 0; k < T; k++) { probs[k] = Math.exp(probs[k] - max); sum += probs[k]; }
          const ao = q * D + hOff;
          for (let i = 0; i < dh; i++) att[ao + i] = 0;
          for (let k = 0; k < T; k++) {
            const w = probs[k] / sum, vo = k * 3 * D + 2 * D + hOff;
            for (let i = 0; i < dh; i++) att[ao + i] += w * qkv[vo + i];
          }
        }
      }

      linear(att, t[p + "proj.weight"], t[p + "proj.bias"], T, D, D, projd);
      for (let i = 0; i < T * D; i++) x[i] += projd[i];

      h.set(x);
      layerNorm(h, t[p + "norm2.weight"], t[p + "norm2.bias"], D, T);
      linear(h, t[p + "ff.0.weight"], t[p + "ff.0.bias"], T, D, F, ff1);
      for (let i = 0; i < T * F; i++) ff1[i] = gelu(ff1[i]);
      linear(ff1, t[p + "ff.2.weight"], t[p + "ff.2.bias"], T, F, D, ff2);
      for (let i = 0; i < T * D; i++) x[i] += ff2[i];
    }

    layerNorm(x, t["norm.weight"], t["norm.bias"], D, T);

    const pooled = new Float32Array(D);
    for (let p = 0; p < T; p++) for (let i = 0; i < D; i++) pooled[i] += x[p * D + i];
    for (let i = 0; i < D; i++) pooled[i] /= T;

    const n = this.cfg.n_emoji;
    const logits = new Float32Array(n);
    linear(pooled, t["head.weight"], t["head.bias"], 1, D, n, logits);

    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-logits[i]));
    return out;
  }

  probs(text) {
    const { ids, length } = this.tok.encode(cleanText(text), this.cfg.max_len);
    return this.forward(ids, length);
  }

  predict(text, alpha = PRIOR_ALPHA, keywordWeight = KEYWORD_WEIGHT) {
    const p = this.probs(text);
    const n = p.length;
    if (keywordWeight > 0 && Object.keys(this.keywords).length) {
      const index = new Map(this.emoji.map((e, i) => [e, i]));
      for (const [e, score] of keywordHits(cleanText(text), this.keywords)) {
        const j = index.get(e);
        if (j !== undefined) p[j] = p[j] + keywordWeight * score;
      }
    }
    let order = Array.from({ length: n }, (_, i) => i);
    if (this.prior && alpha > 0) {
      const adj = new Float64Array(n);
      for (let i = 0; i < n; i++) adj[i] = p[i] / Math.pow(Math.max(this.prior[i], 1e-6), alpha);
      order.sort((a, b) => adj[b] - adj[a]);
    } else {
      order.sort((a, b) => p[b] - p[a]);
    }
    order = order.slice(0, MAX_EMOJI);
    const top = p[order[0]];
    const out = [{ emoji: this.emoji[order[0]], score: top }];
    for (let k = 1; k < order.length; k++) {
      const s = p[order[k]];
      if (s >= ABS_THRESHOLD && s >= REL_RATIO * top) out.push({ emoji: this.emoji[order[k]], score: s });
    }
    return out;
  }
}

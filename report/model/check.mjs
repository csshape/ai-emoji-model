// Compare the browser implementation against PyTorch on the same phrases.
// Usage: node report/model/check.mjs <name> <phrases.json>
import { readFileSync } from "node:fs";
import { EmojiModel, cleanText } from "./emoji.js";

const name = process.argv[2] ?? "base";
const meta = JSON.parse(readFileSync(`report/model/${name}.json`, "utf8"));
const bin = readFileSync(`report/model/${name}.bin`);
const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const model = new EmojiModel(meta, buf);

const phrases = JSON.parse(readFileSync(process.argv[3], "utf8"));
const out = {};
for (const p of phrases) {
  const { ids, length } = model.tok.encode(cleanText(p), model.cfg.max_len);
  const probs = model.probs(p);
  let top = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
  out[p] = {
    tokens: Array.from(ids.slice(0, length)),
    emoji: model.predict(p).map(r => r.emoji).join(""),
    maxProb: Number(probs[top].toFixed(6)),
    topIdx: top,
  };
}
console.log(JSON.stringify(out));

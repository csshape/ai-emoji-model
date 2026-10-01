// The keyword dictionary boost, ported from keywordHits() in emoji.js.

import Foundation

struct KeywordTable: Sendable {
    /// In file order: hit scores are summed per emoji in table order, and a
    /// flag tie-break compares those sums with `===`, so the order of Double
    /// additions has to match `Object.entries` exactly.
    let entries: [(key: UTF16Units, emoji: [String])]

    var isEmpty: Bool { entries.isEmpty }

    init(entries: [(key: UTF16Units, emoji: [String])] = []) {
        self.entries = entries
    }

    init(json data: Data) throws {
        var parser = OrderedJSON(bytes: Array(data))
        entries = try parser.parseStringArrayObject().map { (Array($0.key.utf16), $0.value) }
    }

    /// Emoji to summed score (`key.length * 0.01` per matching key), in the
    /// order a JS Map would iterate them: first insertion first.
    func hits(_ text: String) -> [(emoji: String, score: Double)] {
        let words = keyWords(jsLowercased(text))
        // Each word counts only its longest matching key, so "bee" stays quiet
        // inside "beer". Mirrors keywords.fired_keys.
        var fired = Set<UTF16Units>()
        for w in words {
            var best: UTF16Units?
            for (key, _) in entries where w.starts(with: key) && key.count > (best?.count ?? -1) {
                best = key
            }
            if let best { fired.insert(best) }
        }
        var order: [String] = []
        var slot: [UTF16Units: Int] = [:]
        var scores: [Double] = []
        for (key, emojis) in entries {
            guard fired.contains(key) else { continue }
            let add = Double(key.count) * 0.01
            for e in emojis {
                let k = Array(e.utf16)
                if let s = slot[k] {
                    scores[s] = scores[s] + add
                } else {
                    slot[k] = scores.count
                    order.append(e)
                    scores.append(0 + add)
                }
            }
        }
        return zip(order, scores).map { ($0, $1) }
    }
}

/// text.match(/[a-zA-ZæøåÆØÅ]{2,}/g)
private func keyWords(_ text: String) -> [UTF16Units] {
    func isKeyChar(_ c: UInt16) -> Bool {
        switch c {
        case 0x41...0x5A, 0x61...0x7A, 0xE6, 0xF8, 0xE5, 0xC6, 0xD8, 0xC5: return true
        default: return false
        }
    }
    let t = Array(text.utf16)
    var words: [UTF16Units] = []
    var i = 0
    while i < t.count {
        guard isKeyChar(t[i]) else { i += 1; continue }
        var j = i
        while j < t.count, isKeyChar(t[j]) { j += 1 }
        if j - i >= 2 { words.append(Array(t[i..<j])) }
        i = j
    }
    return words
}

/// Two regional indicator symbols, i.e. a country flag.
func isFlag(_ e: String) -> Bool {
    let cps = Array(e.unicodeScalars)
    return cps.count == 2 && cps.allSatisfy { (0x1F1E6...0x1F1FF).contains($0.value) }
}

/// Just enough JSON to read `{"key": ["emoji", ...], ...}` with key order kept;
/// Foundation's decoders hand back unordered dictionaries.
private struct OrderedJSON {
    let bytes: [UInt8]
    var pos = 0

    init(bytes: [UInt8]) { self.bytes = bytes }

    mutating func parseStringArrayObject() throws -> [(key: String, value: [String])] {
        var out: [(key: String, value: [String])] = []
        try expect(UInt8(ascii: "{"))
        if peek() == UInt8(ascii: "}") { pos += 1; return out }
        while true {
            let key = try parseString()
            try expect(UInt8(ascii: ":"))
            try expect(UInt8(ascii: "["))
            var values: [String] = []
            if peek() == UInt8(ascii: "]") {
                pos += 1
            } else {
                while true {
                    values.append(try parseString())
                    let c = try next()
                    if c == UInt8(ascii: "]") { break }
                    guard c == UInt8(ascii: ",") else { throw EmojiModelError.badKeywords }
                }
            }
            out.append((key, values))
            let c = try next()
            if c == UInt8(ascii: "}") { return out }
            guard c == UInt8(ascii: ",") else { throw EmojiModelError.badKeywords }
        }
    }

    private mutating func skipSpace() {
        while pos < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[pos]) { pos += 1 }
    }

    private mutating func peek() -> UInt8? {
        skipSpace()
        return pos < bytes.count ? bytes[pos] : nil
    }

    private mutating func next() throws -> UInt8 {
        guard let c = peek() else { throw EmojiModelError.badKeywords }
        pos += 1
        return c
    }

    private mutating func expect(_ c: UInt8) throws {
        guard try next() == c else { throw EmojiModelError.badKeywords }
    }

    private mutating func parseString() throws -> String {
        try expect(UInt8(ascii: "\""))
        var units: UTF16Units = []
        var raw: [UInt8] = []
        func flushRaw() {
            units.append(contentsOf: String(decoding: raw, as: UTF8.self).utf16)
            raw.removeAll()
        }
        while pos < bytes.count {
            let c = bytes[pos]
            pos += 1
            switch c {
            case UInt8(ascii: "\""):
                flushRaw()
                return String(decoding: units, as: UTF16.self)
            case UInt8(ascii: "\\"):
                flushRaw()
                guard pos < bytes.count else { throw EmojiModelError.badKeywords }
                let e = bytes[pos]
                pos += 1
                switch e {
                case UInt8(ascii: "n"): units.append(0x0A)
                case UInt8(ascii: "t"): units.append(0x09)
                case UInt8(ascii: "r"): units.append(0x0D)
                case UInt8(ascii: "b"): units.append(0x08)
                case UInt8(ascii: "f"): units.append(0x0C)
                case UInt8(ascii: "u"):
                    guard pos + 4 <= bytes.count,
                          let v = UInt16(String(decoding: bytes[pos..<(pos + 4)], as: UTF8.self), radix: 16)
                    else { throw EmojiModelError.badKeywords }
                    units.append(v)          // surrogate halves pair up in the UTF-16 buffer
                    pos += 4
                default: units.append(UInt16(e))   // \" \\ \/
                }
            default:
                raw.append(c)
            }
        }
        throw EmojiModelError.badKeywords
    }
}

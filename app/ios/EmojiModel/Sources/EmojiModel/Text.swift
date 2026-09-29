// Text normalisation, ported from report/model/emoji.js.
//
// The JS works on UTF-16 strings and its regexes run without the `u` flag, so
// everything here operates on UTF-16 code units. The regexes are hand-written
// scanners rather than NSRegularExpression: ICU's `\w`, `\s` and `.` are
// Unicode-aware and `.` matches whole code points, which would collapse
// "😂😂😂😂" where the JS leaves it alone.

typealias UTF16Units = [UInt16]

private enum Unit {
    static let space: UInt16 = 0x20
    static let at: UInt16 = 0x40            // @
    static let slash: UInt16 = 0x2F         // /
    static let lowerR: UInt16 = 0x72        // r
    static let lowerS: UInt16 = 0x73        // s
}

/// JS `\s` (and what `String.prototype.trim` strips): WhiteSpace plus LineTerminator.
@inline(__always)
func isJSWhitespace(_ c: UInt16) -> Bool {
    switch c {
    case 0x09...0x0D, 0x20, 0xA0, 0x1680, 0x2000...0x200A,
         0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF:
        return true
    default:
        return false
    }
}

/// JS `.` without the `s` flag stops only at these four.
@inline(__always)
private func isJSLineTerminator(_ c: UInt16) -> Bool {
    c == 0x0A || c == 0x0D || c == 0x2028 || c == 0x2029
}

/// JS `\w` is ASCII-only.
@inline(__always)
private func isJSWordChar(_ c: UInt16) -> Bool {
    (c >= 0x30 && c <= 0x39) || (c >= 0x41 && c <= 0x5A) || (c >= 0x61 && c <= 0x7A) || c == 0x5F
}

private func hasPrefix(_ t: UTF16Units, at i: Int, _ ascii: String) -> Bool {
    var j = i
    for u in ascii.utf16 {
        guard j < t.count, t[j] == u else { return false }
        j += 1
    }
    return true
}

private func run(_ t: UTF16Units, from i: Int, while pred: (UInt16) -> Bool) -> Int {
    var j = i
    while j < t.count, pred(t[j]) { j += 1 }
    return j
}

/// Global replace with a matcher that returns the match end at `i`, or nil.
/// Mirrors `String.prototype.replace(/.../g, ...)`: leftmost match, resume after it.
private func replaceAll(_ t: UTF16Units, with replacement: String, matchAt: (UTF16Units, Int) -> Int?) -> UTF16Units {
    let rep = Array(replacement.utf16)
    var out = UTF16Units()
    out.reserveCapacity(t.count)
    var i = 0
    while i < t.count {
        if let end = matchAt(t, i) {
            out.append(contentsOf: rep)
            i = end
        } else {
            out.append(t[i])
            i += 1
        }
    }
    return out
}

/// /https?:\/\/\S+|www\.\S+/
private func matchURL(_ t: UTF16Units, _ i: Int) -> Int? {
    if hasPrefix(t, at: i, "http") {
        var j = i + 4
        if j < t.count, t[j] == Unit.lowerS { j += 1 }
        // Backtracking `s?` to empty would need ':' where 's' is, so one try suffices.
        if hasPrefix(t, at: j, "://") {
            let end = run(t, from: j + 3) { !isJSWhitespace($0) }
            if end > j + 3 { return end }
        }
    }
    if hasPrefix(t, at: i, "www.") {
        let end = run(t, from: i + 4) { !isJSWhitespace($0) }
        if end > i + 4 { return end }
    }
    return nil
}

/// /@\w+/
private func matchMention(_ t: UTF16Units, _ i: Int) -> Int? {
    guard t[i] == Unit.at else { return nil }
    let end = run(t, from: i + 1, while: isJSWordChar)
    return end > i + 1 ? end : nil
}

/// /\/?r\/\w+/
private func matchSubreddit(_ t: UTF16Units, _ i: Int) -> Int? {
    func tail(_ j: Int) -> Int? {
        guard j + 1 < t.count, t[j] == Unit.lowerR, t[j + 1] == Unit.slash else { return nil }
        let end = run(t, from: j + 2, while: isJSWordChar)
        return end > j + 2 ? end : nil
    }
    if t[i] == Unit.slash, let end = tail(i + 1) { return end }
    return tail(i)
}

/// /(.)\1{3,}/g -> "$1$1$1", on code units, so a surrogate pair never counts as a repeat.
private func collapseRepeats(_ t: UTF16Units) -> UTF16Units {
    var out = UTF16Units()
    out.reserveCapacity(t.count)
    var i = 0
    while i < t.count {
        let c = t[i]
        let end = isJSLineTerminator(c) ? i + 1 : run(t, from: i) { $0 == c }
        if end - i >= 4 {
            out.append(contentsOf: [c, c, c])
        } else {
            out.append(contentsOf: t[i..<end])
        }
        i = end
    }
    return out
}

/// .replace(/\s+/g, " ").trim()
private func squashWhitespace(_ t: UTF16Units) -> UTF16Units {
    var out = UTF16Units()
    out.reserveCapacity(t.count)
    var i = 0
    while i < t.count {
        if isJSWhitespace(t[i]) {
            out.append(Unit.space)
            i = run(t, from: i, while: isJSWhitespace)
        } else {
            out.append(t[i])
            i += 1
        }
    }
    // After squashing, whitespace at either end is exactly one space.
    if out.first == Unit.space { out.removeFirst() }
    if out.last == Unit.space { out.removeLast() }
    return out
}

func nfkc(_ s: String) -> String {
    s.precomposedStringWithCompatibilityMapping
}

func string(_ units: UTF16Units) -> String {
    String(decoding: units, as: UTF16.self)
}

/// `String.prototype.toLowerCase()`: full default case mapping plus the one
/// context-sensitive rule, Final_Sigma. Swift's `lowercased()` maps scalar by
/// scalar and always turns "Σ" into "σ".
func jsLowercased(_ s: String) -> String {
    let scalars = Array(s.unicodeScalars)
    var out = String.UnicodeScalarView()
    for (i, u) in scalars.enumerated() {
        if u.value < 0x80 {
            out.append(u.value >= 0x41 && u.value <= 0x5A ? Unicode.Scalar(u.value + 0x20)! : u)
        } else if u.value == 0x03A3, isFinalSigma(scalars, i) {
            out.append(Unicode.Scalar(0x03C2)!)
        } else {
            out.append(contentsOf: u.properties.lowercaseMapping.unicodeScalars)
        }
    }
    return String(out)
}

private func isFinalSigma(_ s: [Unicode.Scalar], _ i: Int) -> Bool {
    var j = i - 1
    while j >= 0, s[j].properties.isCaseIgnorable { j -= 1 }
    guard j >= 0, s[j].properties.isCased else { return false }
    j = i + 1
    while j < s.count, s[j].properties.isCaseIgnorable { j += 1 }
    return !(j < s.count && s[j].properties.isCased)
}

/// cleanText() in emoji.js.
public func cleanText(_ text: String) -> String {
    var t = Array(nfkc(text).utf16)
    t = replaceAll(t, with: " <url> ", matchAt: matchURL)
    t = replaceAll(t, with: " <user> ", matchAt: matchMention)
    t = replaceAll(t, with: " <sub> ", matchAt: matchSubreddit)
    t = collapseRepeats(t)
    return string(squashWhitespace(t))
}

/// normalizeForSp() in emoji.js, returned as code units for the tokenizer.
func normalizeForSp(_ text: String) -> UTF16Units {
    squashWhitespace(Array(jsLowercased(nfkc(text)).utf16))
}

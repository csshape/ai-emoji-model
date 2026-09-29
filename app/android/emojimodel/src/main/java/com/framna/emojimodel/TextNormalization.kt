package com.framna.emojimodel

import java.text.Normalizer

// Mirrors cleanText / normalizeForSp in report/model/emoji.js. Every regex here is
// spelled out so it means what the JS one means, not what java.util.regex defaults to.

// JS `\s` (and String.prototype.trim) is this exact Unicode set. Java's `\s` is ASCII
// only, and `(?U)\s` adds U+0085 and drops U+FEFF, so neither matches.
private const val JS_WS = "\\t\\n\\u000B\\f\\r \\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000\\uFEFF"
private val WS_RUN = Regex("[$JS_WS]+")

// `\w` is ASCII in both JS and Java (no UNICODE_CHARACTER_CLASS), so it is left as is.
private val URL_RE = Regex("https?://[^$JS_WS]+|www\\.[^$JS_WS]+")
private val MENTION_RE = Regex("@\\w+")
private val SUBREDDIT_RE = Regex("/?r/\\w+")

internal fun isJsWhitespace(c: Char): Boolean = when (c) {
    '\t', '\n', '\u000B', '\u000C', '\r', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', '　', '﻿' -> true
    else -> c in ' '..' '
}

// Kotlin's trim() uses Char.isWhitespace, which is a different set from JS trim().
internal fun jsTrim(s: String): String {
    var start = 0
    var end = s.length
    while (start < end && isJsWhitespace(s[start])) start++
    while (end > start && isJsWhitespace(s[end - 1])) end--
    return s.substring(start, end)
}

// JS `.` excludes exactly these; Java's `.` also excludes U+0085.
private fun isJsLineTerminator(c: Char) = c == '\n' || c == '\r' || c == ' ' || c == ' '

/**
 * `t.replace(/(.)\1{3,}/g, "$1$1$1")`, done by hand: without the `u` flag JS `.`
 * matches one UTF-16 code unit, so a run of the same emoji (alternating surrogates)
 * is never collapsed, whereas a Java regex `.` would match whole code points and
 * collapse it.
 */
internal fun collapseRepeats(t: String): String {
    val sb = StringBuilder(t.length)
    var i = 0
    while (i < t.length) {
        val c = t[i]
        var j = i + 1
        while (j < t.length && t[j] == c) j++
        if (j - i >= 4 && !isJsLineTerminator(c)) {
            sb.append(c).append(c).append(c)
        } else {
            sb.append(t, i, j)
        }
        i = j
    }
    return sb.toString()
}

fun cleanText(text: String): String {
    var t = Normalizer.normalize(text, Normalizer.Form.NFKC)
    t = URL_RE.replace(t, " <url> ")
    t = MENTION_RE.replace(t, " <user> ")
    t = SUBREDDIT_RE.replace(t, " <sub> ")
    t = collapseRepeats(t)
    t = WS_RUN.replace(t, " ")
    return jsTrim(t)
}

// lowercase() is Locale.ROOT, like JS toLowerCase(): "ß" stays "ß", no Turkish dotless i.
internal fun normalizeForSp(text: String): String =
    jsTrim(WS_RUN.replace(Normalizer.normalize(text, Normalizer.Form.NFKC).lowercase(), " "))

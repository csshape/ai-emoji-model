package com.framna.emojimodel

import android.content.Context

/** The only Android-aware piece: reads the bundled model from this library's assets. */
object EmojiModelAssets {
    fun load(context: Context): EmojiModel {
        val assets = context.assets
        val json = assets.open("model.json").use { it.readBytes().decodeToString() }
        val bin = assets.open("model.bin").use { it.readBytes() }
        val keywords = runCatching { assets.open("keywords.json").use { it.readBytes().decodeToString() } }.getOrNull()
        return EmojiModel(json, bin, keywords)
    }
}

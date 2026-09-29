package com.framna.emojichat

import android.content.Context
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** app/fixtures/demo_conversation.json, copied into the assets by the build. */
data class DemoConversation(
    val friend: String,
    val seed: List<Pair<Boolean, String>>,
    val replies: List<String>,
) {
    companion object {
        fun load(context: Context): DemoConversation {
            val text = context.assets.open("demo_conversation.json").use { it.readBytes().decodeToString() }
            val root = Json.parseToJsonElement(text).jsonObject
            return DemoConversation(
                friend = root.getValue("friend").jsonPrimitive.content,
                seed = root.getValue("seed").jsonArray.map {
                    val m = it.jsonObject
                    (m.getValue("from").jsonPrimitive.content == "me") to m.getValue("text").jsonPrimitive.content
                },
                replies = root.getValue("replies").jsonArray.map { it.jsonPrimitive.content },
            )
        }
    }
}

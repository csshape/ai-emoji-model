package com.framna.emojimodel

import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * app/fixtures/parity.json is what report/model/emoji.js answers (see app/sync_model.sh);
 * this port has to give the same answers. Gradle runs tests from the module directory.
 */
class EmojiModelParityTest {
    companion object {
        private const val ASSETS = "src/main/assets"
        lateinit var model: EmojiModel

        @BeforeClass
        @JvmStatic
        fun load() {
            model = EmojiModel(
                File("$ASSETS/model.json").readText(),
                File("$ASSETS/model.bin").readBytes(),
                File("$ASSETS/keywords.json").readText(),
            )
        }
    }

    @Test
    fun matchesJavaScriptReference() {
        val fixture = Json.parseToJsonElement(File("../../fixtures/parity.json").readText()).jsonObject
        assertTrue("empty fixture", fixture.isNotEmpty())
        val failures = mutableListOf<String>()
        var worstDelta = 0.0
        for ((phrase, el) in fixture) {
            val want = el.jsonObject
            val wantTokens = want.getValue("tokens").jsonArray.map { it.jsonPrimitive.int }
            val wantEmoji = want.getValue("emoji").jsonPrimitive.content
            val wantMax = want.getValue("maxProb").jsonPrimitive.double
            val wantTop = want.getValue("topIdx").jsonPrimitive.int

            val tokens = model.tokenize(phrase).toList()
            val probs = model.probs(phrase)
            var top = 0
            for (i in 1 until probs.size) if (probs[i] > probs[top]) top = i
            val emoji = model.predict(phrase).joinToString("") { it.emoji }
            val delta = abs(probs[top] - wantMax)
            worstDelta = maxOf(worstDelta, delta)

            val q = Json.encodeToString(String.serializer(), phrase)
            if (tokens != wantTokens) failures += "$q tokens $tokens != $wantTokens"
            if (emoji != wantEmoji) failures += "$q emoji $emoji != $wantEmoji"
            if (top != wantTop) failures += "$q topIdx $top != $wantTop"
            if (delta > 2e-4) failures += "$q maxProb ${probs[top]} != $wantMax"
        }
        println("parity: ${fixture.size} phrases, worst maxProb delta %.2e".format(worstDelta))
        assertEquals(failures.joinToString("\n"), 0, failures.size)
    }

    @Test
    fun predictsFastEnough() {
        val phrases = listOf(
            "I'm so tired", "Pizza tonight?", "Happy birthday!!", "Jeg skal til Grønland",
            "Congrats on the new job", "a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j k l m n o p q r s t u v w x y z",
        )
        repeat(20) { phrases.forEach { model.predict(it) } }
        val rounds = 20
        val start = System.nanoTime()
        repeat(rounds) { phrases.forEach { model.predict(it) } }
        val avgMs = (System.nanoTime() - start) / 1e6 / (rounds * phrases.size)
        println("predict: %.2f ms average".format(avgMs))
        assertTrue("average predict $avgMs ms", avgMs < 50)
    }
}

package com.framna.emojichat

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.framna.emojimodel.EmojiModel
import com.framna.emojimodel.EmojiModelAssets
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ChatMessage(val id: Long, val fromMe: Boolean, val text: String, val reaction: String? = null)

private const val SUGGEST_DEBOUNCE_MS = 120L
private const val REPLY_DELAY_MS = 1200L
private const val TYPING_AFTER_MS = 350L

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val conversation = DemoConversation.load(app)
    private val model = viewModelScope.async(Dispatchers.Default) { EmojiModelAssets.load(app) }
    private var nextId = 0L
    private var nextReply = 0
    private val draftText = MutableStateFlow("")

    val friend: String = conversation.friend
    val messages = mutableStateListOf<ChatMessage>()

    var draft by mutableStateOf(TextFieldValue(""))
        private set
    var suggestions by mutableStateOf<List<String>>(emptyList())
        private set
    var lastLatencyMs by mutableStateOf<Double?>(null)
        private set
    var parameterCount by mutableStateOf<Int?>(null)
        private set
    private var pendingReplies by mutableIntStateOf(0)
    val friendTyping: Boolean get() = pendingReplies > 0

    /** The friend message whose reaction bar is open, and what the model offers for it. */
    var reactionTarget by mutableStateOf<Long?>(null)
        private set
    var reactionOptions by mutableStateOf<List<String>>(emptyList())
        private set

    init {
        conversation.seed.forEach { (fromMe, text) -> messages += ChatMessage(nextId++, fromMe, text) }
        viewModelScope.launch {
            parameterCount = model.await().parameterCount
        }
        viewModelScope.launch {
            // collectLatest cancels the pending delay on every keystroke: a debounce.
            draftText.collectLatest { text ->
                if (text.isBlank()) {
                    suggestions = emptyList()
                    return@collectLatest
                }
                delay(SUGGEST_DEBOUNCE_MS)
                suggestions = suggest(text)
            }
        }
    }

    fun onDraftChange(value: TextFieldValue) {
        draft = value
        draftText.value = value.text
    }

    fun appendEmoji(emoji: String) {
        val text = draft.text
        val sep = if (text.isNotEmpty() && text.last().isLetterOrDigit()) " " else ""
        val next = text + sep + emoji
        onDraftChange(TextFieldValue(next, TextRange(next.length)))
    }

    fun send() {
        val text = draft.text.trim()
        if (text.isEmpty()) return
        messages += ChatMessage(nextId++, fromMe = true, text = text)
        onDraftChange(TextFieldValue(""))
        val reply = conversation.replies[nextReply++ % conversation.replies.size]
        viewModelScope.launch {
            delay(TYPING_AFTER_MS)
            pendingReplies++
            delay(REPLY_DELAY_MS - TYPING_AFTER_MS)
            pendingReplies--
            messages += ChatMessage(nextId++, fromMe = false, text = reply)
        }
    }

    fun openReactions(id: Long) {
        val message = messages.firstOrNull { it.id == id } ?: return
        reactionTarget = id
        reactionOptions = emptyList()
        viewModelScope.launch {
            val options = suggest(message.text)
            if (reactionTarget == id) reactionOptions = options
        }
    }

    fun dismissReactions() {
        reactionTarget = null
    }

    fun react(emoji: String) {
        val id = reactionTarget ?: return
        val i = messages.indexOfFirst { it.id == id }
        if (i >= 0) {
            val m = messages[i]
            messages[i] = m.copy(reaction = if (m.reaction == emoji) null else emoji)
        }
        reactionTarget = null
    }

    private suspend fun suggest(text: String): List<String> {
        val m: EmojiModel = model.await()
        val (result, ms) = withContext(Dispatchers.Default) {
            val start = System.nanoTime()
            val r = m.predict(text)
            r to (System.nanoTime() - start) / 1e6
        }
        lastLatencyMs = ms
        return result.map { forDisplay(it.emoji) }
    }
}

/**
 * The model's vocabulary drops U+FE0F, so "❤" would render as a small text-style glyph;
 * asking for emoji presentation makes single-code-point emoji look like the rest.
 */
private fun forDisplay(emoji: String): String =
    if (emoji.codePointCount(0, emoji.length) == 1) emoji + "️" else emoji

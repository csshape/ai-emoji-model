import EmojiModel
import Foundation
import Observation

struct DemoConversation: Decodable {
    struct Line: Decodable {
        let from: String
        let text: String
    }
    let friend: String
    let seed: [Line]
    let replies: [String]

    static func bundled() -> DemoConversation {
        guard let url = Bundle.main.url(forResource: "demo_conversation", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let convo = try? JSONDecoder().decode(DemoConversation.self, from: data)
        else { return DemoConversation(friend: "Friend", seed: [], replies: ["👋"]) }
        return convo
    }
}

struct Message: Identifiable, Equatable {
    let id = UUID()
    let fromMe: Bool
    let text: String
    var reaction: String?
}

/// Runs the model off the main thread and times each call for the footer.
final class SuggestionEngine: Sendable {
    private let model: EmojiModel

    init(model: EmojiModel) { self.model = model }

    var parameterCount: Int { model.parameterCount }

    func suggest(_ text: String) async -> (emoji: [String], ms: Double) {
        await Task.detached(priority: .userInitiated) { [model] in
            let start = DispatchTime.now().uptimeNanoseconds
            let emoji = model.predict(text).map(\.emoji)
            let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
            return (emoji, ms)
        }.value
    }
}

@MainActor
@Observable
final class ChatModel {
    let friend: String
    private(set) var messages: [Message]
    private(set) var friendIsTyping = false
    private(set) var suggestions: [String] = []
    private(set) var lastLatencyMs: Double?
    private(set) var parameterCount: Int?
    private(set) var loadFailed = false

    /// The friend message whose reaction bar is open, with the model's picks for it.
    private(set) var reactingTo: Message.ID?
    private(set) var reactionChoices: [String] = []

    var draft = "" {
        didSet { if draft != oldValue { scheduleSuggestions() } }
    }

    private let replies: [String]
    private var nextReply = 0
    private var engine: SuggestionEngine?
    private var suggestTask: Task<Void, Never>?

    init(conversation: DemoConversation = .bundled()) {
        friend = conversation.friend
        replies = conversation.replies
        messages = conversation.seed.map { Message(fromMe: $0.from == "me", text: $0.text) }
    }

    var isReady: Bool { engine != nil }

    func load() async {
        guard engine == nil else { return }
        do {
            // Parsing 6.5 MB of weights takes a moment; keep it off the main thread.
            let model = try await Task.detached(priority: .userInitiated) { try EmojiModel.load() }.value
            let engine = SuggestionEngine(model: model)
            self.engine = engine
            parameterCount = engine.parameterCount
            // Warm up so the first real keystroke shows a representative latency.
            _ = await engine.suggest("hello")
            scheduleSuggestions(debounce: false)
        } catch {
            loadFailed = true
        }
    }

    private func scheduleSuggestions(debounce: Bool = true) {
        suggestTask?.cancel()
        let text = draft
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            suggestions = []
            return
        }
        guard let engine else { return }
        suggestTask = Task { [weak self] in
            if debounce { try? await Task.sleep(for: .milliseconds(120)) }
            guard !Task.isCancelled else { return }
            let (emoji, ms) = await engine.suggest(text)
            guard !Task.isCancelled, let self, self.draft == text else { return }
            self.suggestions = emoji
            self.lastLatencyMs = ms
        }
    }

    func appendToDraft(_ emoji: String) {
        draft += emoji
    }

    func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        messages.append(Message(fromMe: true, text: text))
        draft = ""
        closeReactions()
        guard !replies.isEmpty else { return }
        let reply = replies[nextReply % replies.count]
        nextReply += 1
        Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(350))
            self?.friendIsTyping = true
            try? await Task.sleep(for: .milliseconds(850))
            guard let self else { return }
            self.friendIsTyping = false
            self.messages.append(Message(fromMe: false, text: reply))
        }
    }

    func toggleReactions(for message: Message) {
        if reactingTo == message.id {
            closeReactions()
            return
        }
        reactingTo = message.id
        reactionChoices = []
        guard let engine else { return }
        Task { [weak self] in
            let (emoji, ms) = await engine.suggest(message.text)
            guard let self, self.reactingTo == message.id else { return }
            self.reactionChoices = emoji
            self.lastLatencyMs = ms
        }
    }

    func closeReactions() {
        reactingTo = nil
        reactionChoices = []
    }

    /// Plays a fixed conversation so a screen recording shows the same thing
    /// every time: typing one character at a time (slow enough that each
    /// keystroke gets its own suggestions), picking the model's first
    /// suggestion, sending, and reacting to the reply with the model's first pick.
    func playDemoScript() async {
        let lines = ["Beer after work?", "Happy birthday!!", "Going to Spain next week", "My cat is sick"]
        try? await Task.sleep(for: .seconds(1.5))
        for line in lines {
            for ch in line {
                draft.append(ch)
                try? await Task.sleep(for: .milliseconds(140))
            }
            guard await waitFor({ self.suggestions.first != nil }) else { continue }
            try? await Task.sleep(for: .milliseconds(900))
            if let pick = suggestions.first { draft += " " + pick }
            try? await Task.sleep(for: .milliseconds(800))
            let count = messages.count
            send()
            guard await waitFor({ self.messages.count > count + 1 }, timeout: 5) else { continue }
            try? await Task.sleep(for: .milliseconds(900))
            guard let reply = messages.last, !reply.fromMe else { continue }
            toggleReactions(for: reply)
            guard await waitFor({ !self.reactionChoices.isEmpty }) else { continue }
            try? await Task.sleep(for: .milliseconds(1100))
            if let pick = reactionChoices.first { react(pick, to: reply.id) }
            try? await Task.sleep(for: .milliseconds(1200))
        }
    }

    private func waitFor(_ condition: @escaping () -> Bool, timeout: Double = 3) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() > deadline { return false }
            try? await Task.sleep(for: .milliseconds(50))
        }
        return true
    }

    func react(_ emoji: String, to id: Message.ID) {
        guard let i = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[i].reaction = messages[i].reaction == emoji ? nil : emoji
        closeReactions()
    }
}

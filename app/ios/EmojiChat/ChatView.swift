import SwiftUI

struct ChatView: View {
    @State private var chat = ChatModel()
    @FocusState private var composerFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                transcript
                bottomBar
            }
            .background(Color(.systemBackground))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) { header }
            }
            .toolbarBackground(.visible, for: .navigationBar)
        }
        .task {
            await chat.load()
            applyLaunchArguments()
        }
    }

    private var header: some View {
        VStack(spacing: 1) {
            Text(chat.friend).font(.headline)
            Text("Emoji suggestions by an on-device model")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 6) {
                    ForEach(chat.messages) { message in
                        MessageRow(
                            message: message,
                            isReacting: chat.reactingTo == message.id,
                            choices: chat.reactionChoices,
                            onTap: { chat.toggleReactions(for: message) },
                            onReact: { chat.react($0, to: message.id) }
                        )
                        .id(message.id)
                    }
                    if chat.friendIsTyping {
                        TypingIndicator()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .transition(.opacity.combined(with: .scale(scale: 0.8, anchor: .bottomLeading)))
                    }
                    Color.clear.frame(height: 1).id(bottomID)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 12)
            }
            .defaultScrollAnchor(.bottom)
            .scrollDismissesKeyboard(.interactively)
            .contentShape(Rectangle())
            .onTapGesture {
                chat.closeReactions()
                composerFocused = false
            }
            .onChange(of: chat.messages.count) { scrollToBottom(proxy) }
            .onChange(of: chat.friendIsTyping) { scrollToBottom(proxy) }
            .onChange(of: composerFocused) { scrollToBottom(proxy) }
        }
    }

    private let bottomID = "bottom"

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(bottomID, anchor: .bottom) }
    }

    // MARK: Composer

    private var bottomBar: some View {
        VStack(spacing: 8) {
            SuggestionStrip(
                suggestions: chat.suggestions,
                draftIsEmpty: chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                isReady: chat.isReady,
                loadFailed: chat.loadFailed,
                onPick: chat.appendToDraft
            )
            composer
            footer
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 4)
        .background(.bar)
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message", text: $chat.draft, axis: .vertical)
                .lineLimit(1...5)
                .focused($composerFocused)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .strokeBorder(Color(.separator).opacity(0.5), lineWidth: 0.5)
                )
                .submitLabel(.send)
                .onSubmit(chat.send)

            Button(action: chat.send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 32))
                    .symbolRenderingMode(.palette)
                    .foregroundStyle(.white, canSend ? Color.accentColor : Color(.systemGray3))
            }
            .disabled(!canSend)
            .accessibilityLabel("Send")
        }
    }

    private var canSend: Bool {
        !chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var footer: some View {
        Text(footerText)
            .font(.caption2)
            .monospacedDigit()
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity)
    }

    private var footerText: String {
        let params = chat.parameterCount.map { String(format: "%.1fM params", Double($0) / 1e6) } ?? "loading model"
        let latency = chat.lastLatencyMs.map { String(format: "%.0f ms", max($0, 1)) } ?? "– ms"
        return "On-device · \(params) · \(latency)"
    }

    /// `-demoDraft "text"`, `-demoSend YES`, `-demoReaction "😂"` and
    /// `-demoReact YES` put the app in a state worth a screenshot without
    /// driving the keyboard.
    private func applyLaunchArguments() {
        let defaults = UserDefaults.standard
        if let draft = defaults.string(forKey: "demoDraft") { chat.draft = draft }
        if let emoji = defaults.string(forKey: "demoReaction"), !emoji.isEmpty, let last = chat.messages.last(where: { !$0.fromMe }) {
            chat.react(emoji, to: last.id)
        }
        if defaults.bool(forKey: "demoSend") { chat.send() }
        if defaults.bool(forKey: "demoReact"), let last = chat.messages.last(where: { !$0.fromMe }) {
            chat.toggleReactions(for: last)
        }
    }
}

// MARK: - Pieces

private struct MessageRow: View {
    let message: Message
    let isReacting: Bool
    let choices: [String]
    let onTap: () -> Void
    let onReact: (String) -> Void

    var body: some View {
        VStack(alignment: message.fromMe ? .trailing : .leading, spacing: 4) {
            if isReacting {
                ReactionBar(choices: choices, current: message.reaction, onPick: onReact)
                    .transition(.scale(scale: 0.6, anchor: .bottomLeading).combined(with: .opacity))
            }
            Bubble(message: message)
                .onTapGesture { if !message.fromMe { onTap() } }
                .onLongPressGesture { if !message.fromMe { onTap() } }
                .accessibilityHint(message.fromMe ? "" : "Double-tap to react")
        }
        .frame(maxWidth: .infinity, alignment: message.fromMe ? .trailing : .leading)
        .padding(.top, message.reaction == nil ? 0 : 10)
        .animation(.spring(response: 0.3, dampingFraction: 0.75), value: isReacting)
        .animation(.spring(response: 0.3, dampingFraction: 0.6), value: message.reaction)
    }
}

private struct Bubble: View {
    let message: Message

    var body: some View {
        Text(message.text)
            .font(.body)
            .foregroundStyle(message.fromMe ? Color.white : Color.primary)
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .background(
                message.fromMe ? Color.accentColor : Color(.secondarySystemBackground),
                in: RoundedRectangle(cornerRadius: 18, style: .continuous)
            )
            .overlay(alignment: .topTrailing) {
                if let reaction = message.reaction {
                    Text(reaction)
                        .font(.body)
                        .padding(5)
                        .background(Color(.systemBackground), in: Circle())
                        .overlay(Circle().strokeBorder(Color(.separator).opacity(0.6), lineWidth: 0.5))
                        .shadow(color: .black.opacity(0.12), radius: 2, y: 1)
                        .offset(x: 12, y: -14)
                        .transition(.scale.combined(with: .opacity))
                        .accessibilityLabel("Reaction \(reaction)")
                }
            }
            .containerRelativeFrame(.horizontal, alignment: message.fromMe ? .trailing : .leading) { width, _ in
                width * 0.78
            }
    }
}

private struct ReactionBar: View {
    let choices: [String]
    let current: String?
    let onPick: (String) -> Void

    var body: some View {
        HStack(spacing: 2) {
            if choices.isEmpty {
                ProgressView().padding(.horizontal, 20).frame(height: 40)
            } else {
                ForEach(choices, id: \.self) { emoji in
                    Button { onPick(emoji) } label: {
                        Text(emoji)
                            .font(.title2)
                            .frame(minWidth: 40, minHeight: 40)
                            .background(emoji == current ? Color.accentColor.opacity(0.2) : .clear, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("React with \(emoji)")
                }
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(.regularMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(Color(.separator).opacity(0.5), lineWidth: 0.5))
        .shadow(color: .black.opacity(0.15), radius: 8, y: 3)
    }
}

private struct SuggestionStrip: View {
    let suggestions: [String]
    let draftIsEmpty: Bool
    let isReady: Bool
    let loadFailed: Bool
    let onPick: (String) -> Void

    var body: some View {
        Group {
            if loadFailed {
                hint("Could not load the emoji model")
            } else if !isReady {
                hint("Loading the emoji model…")
            } else if draftIsEmpty || suggestions.isEmpty {
                hint("Start typing to get emoji suggestions")
            } else {
                HStack(spacing: 8) {
                    ForEach(suggestions, id: \.self) { emoji in
                        Button { onPick(emoji) } label: {
                            Text(emoji)
                                .font(.title)
                                .frame(maxWidth: .infinity, minHeight: 48)
                                .background(Color(.secondarySystemBackground),
                                            in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                        .buttonStyle(ChipStyle())
                        .accessibilityLabel("Add \(emoji)")
                        .transition(.scale(scale: 0.7).combined(with: .opacity))
                    }
                    // Keep chips the same width whether the model offers one emoji or five.
                    ForEach(suggestions.count..<5, id: \.self) { _ in
                        Color.clear.frame(maxWidth: .infinity).frame(height: 48)
                    }
                }
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .animation(.snappy(duration: 0.2), value: suggestions)
    }

    private func hint(_ text: String) -> some View {
        Text(text)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, minHeight: 48)
    }
}

private struct ChipStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.9 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

private struct TypingIndicator: View {
    var body: some View {
        TimelineView(.animation) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            HStack(spacing: 5) {
                ForEach(0..<3) { i in
                    Circle()
                        .fill(Color.secondary)
                        .frame(width: 8, height: 8)
                        .opacity(0.35 + 0.65 * max(0, sin((t * 2 * .pi / 1.2) - Double(i) * 0.8)))
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .accessibilityLabel("Typing")
    }
}

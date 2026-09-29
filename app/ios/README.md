# EmojiChat (iOS)

A SwiftUI chat demo that suggests emoji with the on-device model — a Swift port
of `report/model/emoji.js` with no ML runtime (Accelerate only).

- `EmojiModel/` — Swift package: tokenizer, encoder, keyword boost, flag rule.
  Resources are copied in by `app/sync_model.sh`; don't edit them by hand.
- `EmojiChat/` — the app. `project.yml` is the XcodeGen spec; the generated
  `EmojiChat.xcodeproj` is checked in. The script comes from
  `app/fixtures/demo_conversation.json`.

## Test

```sh
cd app/ios/EmojiModel && swift test
```

The parity test checks every phrase in `app/fixtures/parity.json` against the
JS output (tokens, emoji, top index exactly; max probability within 2e-4) and
a timing test keeps a prediction under 50 ms. Point `PARITY_FIXTURE` at any
other `check.mjs` output to test more phrases.

## Build and run

```sh
cd app/ios
xcodegen            # only after editing project.yml
xcodebuild -project EmojiChat.xcodeproj -scheme EmojiChat \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build
```

Or open `EmojiChat.xcodeproj` and run. Needs iOS 17 or later.

For screenshots without a keyboard, launch arguments set up state:
`-demoDraft "text"`, `-demoSend YES`, `-demoReaction "😂"` (on the last friend
message) and `-demoReact YES` (opens its reaction bar).

# Emoji Chat (Android)

A Compose chat demo that runs the emoji model on-device. `:emojimodel` is a 1:1 Kotlin
port of `report/model/emoji.js` (tokenizer, encoder, keyword boost, flag rule) with no
Android dependencies in its core; `:app` is the chat UI, scripted by
`app/fixtures/demo_conversation.json`.

## Setup

```sh
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties
```

The model files in `emojimodel/src/main/assets` come from `app/sync_model.sh`; rerun it
after exporting a new model.

## Test

```sh
./gradlew :emojimodel:test
```

Asserts parity with the JS reference for every phrase in `app/fixtures/parity.json`
(tokens, emoji, top index exact; max probability within 2e-4) and that a prediction
averages under 50 ms on the JVM.

## Build and run

```sh
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.framna.emojichat/.MainActivity
```

Type a message to see suggestions above the composer; tap or long-press a friend's
message to react with the model's suggestions for it.

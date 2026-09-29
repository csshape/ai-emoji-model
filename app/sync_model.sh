#!/bin/sh
# Copy an exported model into both apps and refresh the parity fixture.
# Usage: app/sync_model.sh [model]   (default v10; any name under report/model/)
set -eu
cd "$(dirname "$0")/.."
MODEL="${1:-v10}"
SRC=report/model
for f in "$MODEL.json" "$MODEL.bin" keywords.json; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f -- run emoji_model.export_web first" >&2; exit 1; }
done
IOS=app/ios/EmojiModel/Sources/EmojiModel/Resources
AND=app/android/emojimodel/src/main/assets
mkdir -p "$IOS" "$AND"
for dst in "$IOS" "$AND"; do
  cp "$SRC/$MODEL.json" "$dst/model.json"
  cp "$SRC/$MODEL.bin" "$dst/model.bin"
  cp "$SRC/keywords.json" "$dst/keywords.json"
done
# What the browser implementation answers is the reference both ports test against.
node "$SRC/check.mjs" "$MODEL" app/fixtures/phrases.json > app/fixtures/parity.json
echo "$MODEL -> $IOS, $AND; fixture app/fixtures/parity.json"

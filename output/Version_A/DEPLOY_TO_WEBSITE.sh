#!/usr/bin/env bash
# Deploys Version A assets directly to WeatherXM Website
set -e
TARGET="${WEBSITE_DIR:-../weatherxmcom-website}/public/assets/videos"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_DIR="$SCRIPT_DIR"
if [ ! -f "$SOURCE_DIR/station_matchcut_720p.mp4" ] && [ -d "$SCRIPT_DIR/Part_1" ]; then
    SOURCE_DIR="$SCRIPT_DIR/Part_1"
fi

if [ ! -d "$TARGET" ]; then
    echo "❌ Error: Website directory $TARGET not found."
    exit 1
fi

echo "🚀 Deploying Version A assets from $SOURCE_DIR to $TARGET..."
cp -v "$SOURCE_DIR"/station_matchcut* "$TARGET/"
if [ -f "$SOURCE_DIR/network-matchcut-poster.webp" ]; then
    cp -v "$SOURCE_DIR/network-matchcut-poster.webp" "$TARGET/"
fi
echo "✅ Successfully deployed video assets to $TARGET!"

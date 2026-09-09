#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
APP="$PWD/build/NovTokenizer Monitor.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
xcrun swiftc -parse-as-library -O -framework SwiftUI -framework AppKit Monitor.swift -o "$APP/Contents/MacOS/NovTokenizerMonitor"
cp collector.py "$APP/Contents/Resources/collector.py"
if [ -f .private/config.json ]; then cp .private/config.json "$APP/Contents/Resources/config.json"; fi
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict><key>CFBundleExecutable</key><string>NovTokenizerMonitor</string><key>CFBundleIdentifier</key><string>local.novtokenizer.monitor</string><key>CFBundleName</key><string>NovTokenizer Monitor</string><key>CFBundleVersion</key><string>1</string><key>CFBundlePackageType</key><string>APPL</string><key>LSMinimumSystemVersion</key><string>13.0</string><key>NSHighResolutionCapable</key><true/></dict></plist>
PLIST
printf '%s\n' "$APP"

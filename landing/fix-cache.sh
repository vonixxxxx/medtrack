#!/bin/bash
echo "🔄 Fixing cache and restarting..."
cd "$(dirname "$0")"
pkill -f "next dev" 2>/dev/null || true
rm -rf .next
echo "✅ Cache cleared. Starting dev server..."
npm run dev






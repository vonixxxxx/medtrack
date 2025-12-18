#!/bin/bash

# Clear Next.js cache and restart dev server
echo "🧹 Clearing Next.js cache..."
rm -rf .next

echo "🔄 Restarting dev server..."
echo ""
echo "✅ Server starting... Open http://localhost:3000 in your browser"
echo "💡 If you still see old content, do a hard refresh (Cmd+Shift+R or Ctrl+Shift+R)"
echo ""

npm run dev






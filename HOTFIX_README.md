# 🔥 HOTFIX: useEffect Import Error

## Issue
`Uncaught ReferenceError: useEffect is not defined` in GraphBuilder.tsx

## Root Cause
Vite HMR (Hot Module Replacement) cache issue - file updated but browser still has old version.

## Fix Applied
✅ Updated import to: `import React, { useState, useEffect } from 'react';`

## Solution Steps

### Option 1: Hard Refresh Browser (Fastest)
1. Press `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
2. Or open DevTools → Right-click refresh button → "Empty Cache and Hard Reload"

### Option 2: Restart Dev Server (Recommended)
```bash
# Stop the dev server (Ctrl+C)
# Then restart:
cd frontend && npm run dev
```

### Option 3: Clear Vite Cache
```bash
cd frontend
rm -rf node_modules/.vite
npm run dev
```

## Verification
After refresh, GraphBuilder should:
- ✅ Load without errors
- ✅ React to filter changes
- ✅ Show empty state when no data
- ✅ Update Y-axis when metric filter changes

---

**Status:** Fixed in code | Requires browser refresh/cache clear







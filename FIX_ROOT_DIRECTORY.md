# Fix Root Directory Error

## Problem

Vercel error: "The specified Root Directory 'app-root' does not exist."

This happens because the Vercel project was configured with a root directory that doesn't exist.

## Solution: Update Project Settings in Dashboard

### Step 1: Go to Project Settings

1. Go to: https://vercel.com/vonixs-projects/medtrack/settings

### Step 2: Update Root Directory

1. Scroll to **"General"** section
2. Find **"Root Directory"** setting
3. Either:
   - **Option A**: Clear the field (leave empty) - Recommended
   - **Option B**: Set to `.` (current directory)
4. Click **"Save"**

### Step 3: Redeploy

After saving:
1. Go to **"Deployments"** tab
2. Click **"Deploy"** button
3. Select `main` branch
4. Click **"Deploy"**

## Alternative: Use Vercel CLI

If you have Vercel CLI access:

```bash
# Remove old .vercel config
rm -rf .vercel

# Link project fresh (will prompt for settings)
vercel link

# Deploy
vercel --prod
```

When linking, make sure:
- Root Directory: Leave empty or set to `.`
- Framework Preset: Other
- Build Command: (already in vercel.json)
- Output Directory: (already in vercel.json)

## What We Did

1. ✅ Removed `.vercel` directory from repository
2. ✅ This forces Vercel to use default root directory
3. ✅ Next deployment will use project root (`.`)

## Expected Result

After updating root directory in dashboard:
- ✅ Build will start from project root
- ✅ No "app-root does not exist" error
- ✅ Build should proceed normally

---

**Quick Link**: https://vercel.com/vonixs-projects/medtrack/settings


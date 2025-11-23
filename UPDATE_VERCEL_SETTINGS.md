# Update Vercel Project Settings

## Problem

Build command is failing even though vercel.json is configured correctly. Vercel dashboard settings might be overriding vercel.json.

## Solution: Update Settings in Vercel Dashboard

### Step 1: Go to Project Settings

1. Go to: **https://vercel.com/vonixs-projects/medtrack/settings**
2. Click **"Build and Deployment"** section

### Step 2: Update Build Settings

Set these values exactly:

**Build Command:**
```
cd api && npm run prisma:generate && cd ../frontend && npm run build
```

**Output Directory:**
```
frontend/dist
```

**Install Command:**
```
npm install && cd frontend && npm install && cd ../api && npm install
```

**Root Directory:**
```
(leave empty or set to '.')
```

### Step 3: Framework Preset

- Set to: **"Other"** or **"Vite"** (if available)
- Don't use auto-detection

### Step 4: Save and Deploy

1. Click **"Save"** button
2. Go to **"Deployments"** tab
3. Click **"Deploy"** button
4. Select `main` branch
5. Click **"Deploy"**

## Alternative: Use Build Script

If chained commands don't work, we can create a build script:

1. Create `build.sh` in root:
```bash
#!/bin/bash
set -e
cd api && npm run prisma:generate
cd ../frontend && npm run build
```

2. Update vercel.json buildCommand to: `bash build.sh`

3. Make script executable: `chmod +x build.sh`

## Verify Settings Match

Make sure dashboard settings match vercel.json:
- Build Command
- Output Directory  
- Install Command
- Root Directory

---

**Quick Link**: https://vercel.com/vonixs-projects/medtrack/settings

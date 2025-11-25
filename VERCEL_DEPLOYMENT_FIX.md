# Vercel Deployment Fix Guide

## Issue: Root Directory "app-root" does not exist

### Problem
Vercel project settings have Root Directory set to "app-root" which doesn't exist.
The project root is the repository root (medtrack/).

### Solution: Update Vercel Project Settings

1. Go to: https://vercel.com/dashboard
2. Find your "medtrack" project
3. Click on "Settings"
4. Click on "General"
5. Find "Root Directory" setting
6. **Clear it** (set to empty/blank) OR set to `.` (current directory)
7. Click "Save"

### Alternative: Delete and Reconnect Project

If the above doesn't work:
1. Go to Vercel Dashboard
2. Find "medtrack" project
3. Settings → General → Delete Project
4. Click "Add New Project"
5. Import from GitHub: `vonixxxxx/medtrack`
6. **DO NOT set Root Directory** (leave blank)
7. Framework Preset: "Other" or "Vite"
8. Click "Deploy"

### Build Configuration

The project uses:
- **Root**: Repository root (no subdirectory)
- **Build Command**: `bash build.sh`
- **Output Directory**: `frontend/dist`
- **Install Command**: `npm install && cd frontend && npm install --include=dev && cd ../api && npm install`

### Project Structure

```
medtrack/                    ← This is the root
├── frontend/                ← Frontend (Vite + React)
│   └── dist/               ← Build output
├── api/                     ← Serverless functions
│   ├── auth/
│   │   └── [...].ts        ← Auth routes
│   ├── meds/
│   │   └── [...].ts        ← Medication routes
│   └── [...].ts            ← Other routes
├── vercel.json              ← Vercel config
└── build.sh                 ← Build script
```

### After Fixing Root Directory

1. The deployment should work
2. All routes should function:
   - `/api/auth/login` → `api/auth/[...].ts`
   - `/api/auth/signup` → `api/auth/[...].ts`
   - `/api/meds/user` → `api/meds/[...].ts`
   - Frontend routes → `frontend/dist/index.html`

### Verify Deployment

After fixing:
1. Check build logs - should complete successfully
2. Test login: Should see logs in function logs
3. Check all API endpoints work
4. Verify frontend loads correctly


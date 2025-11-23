# Deploy via GitHub (Recommended for Large Projects)

## Why GitHub Deployment?

Vercel's CLI has a 5,000 file limit for free tier uploads. Your project has 65,733 files.

**Solution:** Deploy via GitHub - Vercel deploys directly from your repository, avoiding the upload limit.

## Step 1: Commit and Push to GitHub

```bash
# Add all changes
git add .

# Commit
git commit -m "feat: restructure to Vercel monorepo with serverless functions"

# Push to GitHub
git push origin main
```

## Step 2: Deploy via Vercel Dashboard

1. Go to: https://vercel.com/vonixs-projects/medtrack
2. The project is already connected to: https://github.com/vonixxxxx/medtrack
3. Click "Deploy" or wait for auto-deploy
4. Vercel will build from GitHub (no file upload limit!)

## Step 3: Set Environment Variables

In Vercel Dashboard → Settings → Environment Variables:

Add these variables for **Production**:

```
DATABASE_URL=postgresql://postgres:tirpuV-sihsu7-rijjem@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres
JWT_SECRET=8a1ac4d831720f929941ac89de22dea979bbe7c5c4dee9a06ffc17e07d80a400
SUPABASE_URL=https://ydfksxcktsjhadiotlrc.supabase.co
NODE_ENV=production
FRONTEND_URL=https://medtrack.vercel.app
CORS_ORIGIN=https://medtrack.vercel.app
```

(Add SUPABASE_ANON_KEY if you have it)

## Step 4: Verify Deployment

After deployment completes:
- Frontend: `https://medtrack.vercel.app`
- API: `https://medtrack.vercel.app/api/health`

## Alternative: Use Vercel CLI with GitHub

```bash
# Deploy from GitHub (not local files)
vercel --prod --git
```

This uses GitHub as source instead of uploading files.

## Benefits of GitHub Deployment

✅ No file upload limit
✅ Automatic deployments on git push
✅ Better build caching
✅ Version control integration
✅ Rollback capabilities

## Troubleshooting

### Build Fails
- Check build logs in Vercel dashboard
- Verify `vercel.json` is correct
- Check environment variables are set

### API Routes Not Working
- Verify `api/` directory structure
- Check Prisma client is generated (add to build command if needed)
- Verify DATABASE_URL is correct

### Frontend Not Loading
- Check `frontend/dist` is being built
- Verify `outputDirectory` in vercel.json
- Check build logs

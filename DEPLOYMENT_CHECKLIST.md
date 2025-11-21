# 🚀 Final Deployment Checklist

**QUICK START:** Run `./DEPLOY_NOW.sh` for automated deployment, or follow the steps below manually.

**See `LIVE_URLS.md` for all production URLs after deployment.**

Follow these steps in order to deploy MedTrack to production.

## Step 1: Set Up Supabase Secrets

1. Install Supabase CLI (if not already installed):
   ```bash
   npm install -g supabase
   ```

2. Login to Supabase:
   ```bash
   supabase login
   ```

3. Link your project:
   ```bash
   supabase link --project-ref ydfksxcktsjhadiotlrc
   ```

4. Set all secrets (copy-paste from `supabase/secrets.txt`):
   ```bash
   # Open supabase/secrets.txt and run each command
   # Or run them all at once:
   cat supabase/secrets.txt | grep "^supabase secrets set" | bash
   ```

   **IMPORTANT:** Update these placeholders before running:
   - `JWT_SECRET` - Generate a secure random string (at least 32 characters)
   - `SMTP_USER` - Your email address
   - `SMTP_PASS` - Your email app password
   - `FRONTEND_URL` - Will be your Vercel URL (update after Step 4)
   - `CORS_ORIGIN` - Will be your Vercel URL (update after Step 4)

## Step 2: Deploy Supabase Edge Function

Run the deployment script:

```bash
cd supabase/functions/backend-express && ./deploy.sh
```

Or manually:

```bash
cd supabase/functions/backend-express
# Copy backend files
mkdir -p backend
cp -r ../../backend/* backend/
# Deploy
supabase functions deploy backend-express
```

**Your backend will be available at:**
```
https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express
```

## Step 3: Run Prisma Migrations on Supabase Database

1. Update your local `.env` with the Supabase DATABASE_URL:
   ```bash
   DATABASE_URL="postgresql://postgres.ydfksxcktsjhadiotlrc:fibbu6-foqJop-qydron@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
   ```

2. Generate Prisma Client:
   ```bash
   cd backend
   npx prisma generate
   ```

3. Run migrations:
   ```bash
   npx prisma migrate deploy
   ```

## Step 4: Deploy Frontend to Vercel

1. Install Vercel CLI (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. Navigate to frontend directory:
   ```bash
   cd frontend
   ```

3. Deploy:
   ```bash
   vercel --prod
   ```

4. **IMPORTANT:** When prompted, add environment variables:
   - `VITE_API_URL` = `https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express`
   - `VITE_SUPABASE_URL` = `https://ydfksxcktsjhadiotlrc.supabase.co`
   - `VITE_SUPABASE_ANON_KEY` = `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlkZmtzeGNrdHNqaGFkaW90bHJjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM0NDUxMDAsImV4cCI6MjA3OTAyMTEwMH0.0ijYSj2crCIa3KDrWab6uqaiwpR_q-V7vpjILyzFTfA`

   Or add them via Vercel Dashboard: Project → Settings → Environment Variables

5. **After deployment, get your Vercel URL** (e.g., `https://medtrack.vercel.app`)

## Step 5: Update CORS and Frontend URLs

1. Update Supabase secrets with your actual Vercel URL:
   ```bash
   supabase secrets set FRONTEND_URL="https://your-actual-vercel-url.vercel.app"
   supabase secrets set CORS_ORIGIN="https://your-actual-vercel-url.vercel.app"
   ```

2. Redeploy the Edge Function:
   ```bash
   cd supabase/functions/backend-express
   supabase functions deploy backend-express
   ```

## ✅ Verification

Test your deployment:

1. **Backend Health Check:**
   ```bash
   curl https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express/api/test-public
   ```

2. **Frontend:** Visit your Vercel URL and test login/registration

3. **Database:** Verify Prisma migrations were applied in Supabase Dashboard

## 🎉 You're Live!

Your MedTrack app is now deployed:
- **Backend:** `https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express`
- **Frontend:** `https://your-vercel-url.vercel.app`
- **Database:** Supabase PostgreSQL (ydfksxcktsjhadiotlrc)


# 🚀 Final Deployment Checklist - 5 Steps

## Step 1: Install & Link Supabase CLI

```bash
npm install -g supabase
supabase login
supabase link --project-ref ydfksxcktsjhadiotlrc
```

## Step 2: Set All Secrets

Open `supabase/secrets.txt` and run each command, OR:

```bash
# Review and update placeholders first, then:
cat supabase/secrets.txt | grep "^supabase secrets set" | bash
```

**⚠️ IMPORTANT:** Update these in `supabase/secrets.txt` before running:
- `JWT_SECRET` - Generate secure random string
- `SMTP_USER` - Your email
- `SMTP_PASS` - Your email app password
- `FRONTEND_URL` - Update after Vercel deploy (Step 4)
- `CORS_ORIGIN` - Update after Vercel deploy (Step 4)

## Step 3: Deploy Backend Edge Function

```bash
cd supabase/functions/backend-express && ./deploy.sh
```

**Backend URL:** `https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express`

## Step 4: Deploy Frontend to Vercel

```bash
cd frontend
vercel --prod
```

When prompted, add these environment variables (or add via Vercel Dashboard):
- `VITE_API_URL` = `https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express`
- `VITE_SUPABASE_URL` = `https://ydfksxcktsjhadiotlrc.supabase.co`
- `VITE_SUPABASE_ANON_KEY` = `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlkZmtzeGNrdHNqaGFkaW90bHJjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM0NDUxMDAsImV4cCI6MjA3OTAyMTEwMH0.0ijYSj2crCIa3KDrWab6uqaiwpR_q-V7vpjILyzFTfA`

**Get your Vercel URL** after deployment (e.g., `https://medtrack.vercel.app`)

## Step 5: Update CORS & Run Migrations

```bash
# Update CORS with your Vercel URL
supabase secrets set FRONTEND_URL="https://your-actual-vercel-url.vercel.app"
supabase secrets set CORS_ORIGIN="https://your-actual-vercel-url.vercel.app"

# Redeploy function
cd supabase/functions/backend-express
supabase functions deploy backend-express

# Run Prisma migrations
cd ../../backend
export DATABASE_URL="postgresql://postgres.ydfksxcktsjhadiotlrc:fibbu6-foqJop-qydron@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
npx prisma generate
npx prisma migrate deploy
```

## ✅ Done!

Your app is live! See `LIVE_URLS.md` for all URLs.

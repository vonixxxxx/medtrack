# Manual Deployment Steps

If you prefer to run steps manually instead of using `DEPLOY.sh`, follow these instructions:

## Step 0: Update DATABASE_URL

Edit `api/.env.local` and ensure `DATABASE_URL` contains your actual PostgreSQL connection string:

```bash
DATABASE_URL="postgresql://user:password@host:5432/database"
```

## Step 1: Start API Server

**Terminal 1:**
```bash
cd api
npx vercel dev
```

The API will run on `http://localhost:3000`

## Step 2: Start Frontend

**Terminal 2:**
```bash
cd frontend
npm run dev
```

The frontend will run on `http://localhost:5173`

## Step 3: Test Endpoints

**Terminal 3 (or use existing terminal):**

```bash
# Health check
curl http://localhost:3000/api/health

# Test public
curl http://localhost:3000/api/test-public

# Login
curl -X POST http://localhost:3000/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"test@test.com","password":"test"}'
```

**Expected Results:**
- `/api/health` → `{"status":"OK","service":"medtrack-backend","timestamp":"..."}`
- `/api/test-public` → `{"message":"Backend is running!"}`
- `/api/auth/login` → Either success with token or error (both confirm routing works)

## Step 4: Open Frontend

Open in browser: http://localhost:5173

**Test:**
- Check browser console (F12) - no errors
- Login/signup forms appear
- API calls work (check Network tab)

## Step 5: Set Production Environment Variables

### Option A: Using Vercel CLI

```bash
# Login first
vercel login

# Add variables (you'll be prompted for values)
vercel env add DATABASE_URL production
# Paste your PostgreSQL connection string when prompted

vercel env add JWT_SECRET production
# Generate one: node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
# Paste the generated secret when prompted

vercel env add NODE_ENV production
# Enter: production

vercel env add FRONTEND_URL production
# Enter: https://your-app.vercel.app (after first deploy, or use placeholder)

vercel env add CORS_ORIGIN production
# Enter: https://your-app.vercel.app

# If using Supabase:
vercel env add SUPABASE_URL production
vercel env add SUPABASE_ANON_KEY production
```

### Option B: Using Vercel Dashboard

1. Go to: https://vercel.com/dashboard
2. Select your project (or create new)
3. Settings → Environment Variables
4. Add each variable:
   - Name: `DATABASE_URL`
   - Value: Your connection string
   - Environment: Select "Production", "Preview", "Development"
5. Repeat for all variables

## Step 6: Deploy to Production

```bash
# From project root
vercel --prod
```

**What happens:**
- Vercel detects monorepo structure
- Builds frontend (Vite)
- Deploys API functions
- Configures routing
- Provides live URLs

**After deployment:**
- Frontend: `https://your-app.vercel.app`
- API: `https://your-app.vercel.app/api/*`

## Step 7: Test Production

```bash
# Test production endpoints
curl https://your-app.vercel.app/api/health
curl https://your-app.vercel.app/api/test-public
```

## Troubleshooting

### API Not Starting
- Check `api/.env.local` exists and has `DATABASE_URL`
- Check port 3000 is available: `lsof -ti:3000`
- Check logs: `tail -f /tmp/vercel-dev.log`

### Frontend Not Starting
- Check port 5173 is available: `lsof -ti:5173`
- Check logs: `tail -f /tmp/vite-dev.log`

### Deployment Fails
- Check environment variables are set: `vercel env ls`
- Check `vercel.json` syntax
- Check build logs in Vercel dashboard

### Environment Variables Not Working
- Verify variable names match exactly (case-sensitive)
- Check environment scope (production/preview/development)
- Redeploy after adding variables: `vercel --prod`

## Quick Reference

**Start both servers:**
```bash
# Terminal 1
cd api && npx vercel dev

# Terminal 2
cd frontend && npm run dev
```

**Test API:**
```bash
curl http://localhost:3000/api/health
```

**Deploy:**
```bash
vercel --prod
```

**Check environment variables:**
```bash
vercel env ls
```








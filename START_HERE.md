# 🚀 START HERE - Deployment Instructions

## Step 1: Update Environment Variables

### Update DATABASE_URL

```bash
# Edit api/.env.local
nano api/.env.local
```

**Replace the placeholder with your real PostgreSQL connection string:**

```bash
# Example content:
DATABASE_URL="postgresql://username:password@host:port/dbname"
```

**Common formats:**
- Local: `postgresql://postgres:password@localhost:5432/medtrack`
- Supabase: `postgresql://postgres.xxx:password@aws-0-xxx.pooler.supabase.com:6543/postgres`
- Railway: `postgresql://postgres:password@containers-us-west-xxx.railway.app:5432/railway`

### Generate JWT_SECRET

```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

**Copy the output** - you'll need this when the script prompts you.

**Example output:**
```
a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456
```

### Prepare Supabase Values (if using)

Have these ready:
- **SUPABASE_URL**: `https://your-project-ref.supabase.co`
- **SUPABASE_ANON_KEY**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

---

## Step 2: Run the Deployment Script

```bash
./DEPLOY.sh
```

### What the Script Does Automatically:

1. ✅ **Validates DATABASE_URL**
   - Checks if `api/.env.local` exists
   - Warns if `DATABASE_URL` looks like a placeholder
   - Asks for confirmation before proceeding

2. ✅ **Starts Servers in Background**
   - API server: `http://localhost:3000`
   - Frontend server: `http://localhost:5173`
   - Logs saved to `/tmp/vercel-dev.log` and `/tmp/vite-dev.log`

3. ✅ **Tests Endpoints Automatically**
   - `/api/health` - Health check
   - `/api/test-public` - Public test endpoint
   - `/api/auth/login` - Login endpoint
   - Reports success/failure for each

4. ✅ **Opens Frontend in Browser**
   - Automatically opens `http://localhost:5173`
   - macOS: Uses `open` command
   - Linux: Uses `xdg-open` command

5. ✅ **Prompts for Environment Variables**
   - Interactive prompts for each variable
   - You'll enter values when asked

6. ✅ **Offers Production Deployment**
   - Asks for confirmation
   - Deploys to Vercel if confirmed

7. ✅ **Clean Cleanup on Exit**
   - Press `Ctrl+C` to stop servers
   - Automatically cleans up background processes

---

## Step 3: Follow Interactive Prompts

### Prompt 1: DATABASE_URL Validation
```
⚠️  WARNING: DATABASE_URL appears to be a placeholder!
Continue anyway? (y/N):
```
- Type `y` if you've updated it
- Type `N` to exit and fix it

### Prompt 2: Environment Variable Setup
```
Continue with environment variable setup? (y/N):
```
- Type `y` to proceed
- You'll be prompted for each variable

### Prompt 3: Enter Variables

**DATABASE_URL:**
```
Adding DATABASE_URL...
Enter value: [paste your connection string]
```

**JWT_SECRET:**
```
Adding JWT_SECRET...
Enter value: [paste the generated secret]
```

**Supabase Variables (if using):**
```
Add Supabase variables? (y/N):
```
- Type `y` if using Supabase
- You'll be prompted for `SUPABASE_URL` and `SUPABASE_ANON_KEY`

**NODE_ENV:**
```
Adding NODE_ENV...
Enter value: production
```

### Prompt 4: Production Deployment
```
Deploy to Vercel now? (y/N):
```
- Type `y` to deploy to production
- Type `N` to skip (deploy later with `vercel --prod`)

---

## ✅ Expected Results

### During Testing

```
Testing /api/health...
   ✅ Health check passed
   Response: {"status":"OK","service":"medtrack-backend","timestamp":"..."}

Testing /api/test-public...
   ✅ Test public endpoint passed
   Response: {"message":"Backend is running!"}

Testing /api/auth/login...
   ✅ Login endpoint responding
   Response: {"error":"Invalid credentials"}  # (expected if no user exists)
```

### After Deployment

- ✅ Frontend: `https://your-app.vercel.app`
- ✅ API: `https://your-app.vercel.app/api/*`
- ✅ All endpoints working
- ✅ Environment variables set

---

## 🐛 Troubleshooting

### Script Won't Start
- Check `api/.env.local` exists
- Verify `DATABASE_URL` is not a placeholder
- Ensure Vercel CLI is installed: `npm i -g vercel`

### Endpoints Fail
- Check server logs: `tail -f /tmp/vercel-dev.log`
- Verify `DATABASE_URL` is correct
- Check Prisma client: `cd api && npm run prisma:generate`

### Port Already in Use
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Kill process on port 5173
lsof -ti:5173 | xargs kill -9
```

### Environment Variables Not Saving
- Verify Vercel login: `vercel whoami`
- Check variable names match exactly (case-sensitive)
- Try setting via dashboard: https://vercel.com/dashboard

---

## 📋 Quick Reference

**Before Running:**
```bash
# 1. Update DATABASE_URL
nano api/.env.local

# 2. Generate JWT_SECRET
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"

# 3. Run script
./DEPLOY.sh
```

**During Execution:**
- Follow prompts
- Enter values when asked
- Press `Ctrl+C` to stop anytime

**After Deployment:**
```bash
# Test production
curl https://your-app.vercel.app/api/health

# Check environment variables
vercel env ls
```

---

## 🎯 Summary

1. ✅ Update `api/.env.local` with real `DATABASE_URL`
2. ✅ Generate `JWT_SECRET` and have it ready
3. ✅ Run `./DEPLOY.sh`
4. ✅ Follow interactive prompts
5. ✅ Deploy to production when ready

**That's it!** Your monorepo will be fully live on Vercel, fully tested, and ready for use. 🚀

---

**Need help?** See:
- `FINAL_CHECKLIST.md` - Complete checklist
- `DEPLOY_MANUAL.md` - Manual step-by-step
- `STEP_BY_STEP_GUIDE.md` - Full guide








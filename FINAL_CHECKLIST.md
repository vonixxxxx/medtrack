# 🚀 Final Checklist - Before Running DEPLOY.sh

## ✅ Pre-Deployment Checklist

### Step 1: Update Database Connection

**⚠️ CRITICAL: Must be done before running DEPLOY.sh**

```bash
# Edit api/.env.local
nano api/.env.local   # or use your preferred editor (vim, code, etc.)

# Replace this placeholder:
DATABASE_URL="postgresql://user:password@localhost:5432/medtrack"

# With your actual PostgreSQL connection string:
DATABASE_URL="postgresql://username:password@host:5432/database_name"
```

**Example formats:**
- Local: `postgresql://postgres:password@localhost:5432/medtrack`
- Supabase: `postgresql://postgres.xxx:password@aws-0-xxx.pooler.supabase.com:6543/postgres`
- Railway: `postgresql://postgres:password@containers-us-west-xxx.railway.app:5432/railway`

**Verify:**
```bash
# Check the file was updated
grep DATABASE_URL api/.env.local
```

---

### Step 2: Install Vercel CLI (if not already installed)

```bash
npm i -g vercel
```

**Verify installation:**
```bash
vercel --version
```

Should output: `Vercel CLI 48.x.x` or similar

---

### Step 3: Prepare Environment Variable Values

Have these ready when the script prompts you:

#### Required Variables:

1. **DATABASE_URL**
   - Your PostgreSQL connection string (same as in `.env.local`)
   - Format: `postgresql://user:password@host:port/database`

2. **JWT_SECRET**
   - Generate a secure random secret:
   ```bash
   node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
   ```
   - Copy the output (64 character hex string)
   - Example: `a1b2c3d4e5f6...` (64 characters)

3. **NODE_ENV**
   - Value: `production`

#### Optional Variables (if using Supabase):

4. **SUPABASE_URL**
   - Format: `https://your-project-ref.supabase.co`
   - Example: `https://ydfksxcktsjhadiotlrc.supabase.co`

5. **SUPABASE_ANON_KEY**
   - Your Supabase anonymous/public key
   - Format: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

#### Additional Variables (may be needed):

6. **FRONTEND_URL**
   - Will be: `https://your-app.vercel.app` (after first deploy)
   - Or use placeholder for now: `https://medtrack.vercel.app`

7. **CORS_ORIGIN**
   - Same as FRONTEND_URL

---

## 🚀 Running DEPLOY.sh

### Quick Start

```bash
# From project root
./DEPLOY.sh
```

### What the Script Does

1. ✅ **Validates Setup**
   - Checks `api/.env.local` exists
   - Warns if `DATABASE_URL` looks like a placeholder
   - Asks for confirmation before proceeding

2. ✅ **Starts Servers**
   - API server on `http://localhost:3000` (background)
   - Frontend server on `http://localhost:5173` (background)
   - Logs saved to `/tmp/vercel-dev.log` and `/tmp/vite-dev.log`

3. ✅ **Tests Endpoints**
   - `/api/health` - Health check
   - `/api/test-public` - Public test endpoint
   - `/api/auth/login` - Login endpoint
   - Reports success/failure for each

4. ✅ **Opens Browser**
   - Automatically opens `http://localhost:5173`
   - macOS: Uses `open` command
   - Linux: Uses `xdg-open` command

5. ✅ **Sets Environment Variables** (Interactive)
   - Prompts to continue
   - Guides through `vercel env add` for each variable
   - You'll enter values when prompted

6. ✅ **Deploys to Production** (Optional)
   - Asks for confirmation
   - Runs `vercel --prod`
   - Shows deployment URL when complete

### During Execution

**Interactive Prompts:**

1. **DATABASE_URL Warning** (if placeholder detected)
   ```
   Continue anyway? (y/N):
   ```
   - Type `y` if you've updated it, `N` to exit and fix

2. **Environment Variables Setup**
   ```
   Continue with environment variable setup? (y/N):
   ```
   - Type `y` to proceed
   - You'll be prompted for each variable value

3. **Supabase Variables**
   ```
   Add Supabase variables? (y/N):
   ```
   - Type `y` if using Supabase, `N` to skip

4. **Production Deployment**
   ```
   Deploy to Vercel now? (y/N):
   ```
   - Type `y` to deploy, `N` to skip

### Stopping the Script

**Press `Ctrl+C`** at any time to:
- Stop both servers
- Exit the script
- Clean up background processes

---

## 📊 Expected Output

### Successful Test Results

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

### If Tests Fail

The script will show:
- ❌ Failed endpoint
- Response received
- Log file location for debugging

**Debug:**
```bash
# Check API logs
tail -f /tmp/vercel-dev.log

# Check Frontend logs
tail -f /tmp/vite-dev.log
```

---

## 🐛 Troubleshooting

### "DATABASE_URL is not set"
- Ensure `api/.env.local` exists
- Check `DATABASE_URL` is not commented out
- Restart script after fixing

### "Port 3000 already in use"
```bash
# Find and kill process
lsof -ti:3000 | xargs kill -9
```

### "Port 5173 already in use"
```bash
# Find and kill process
lsof -ti:5173 | xargs kill -9
```

### "Vercel CLI not found"
```bash
npm i -g vercel
```

### "Cannot find module '@prisma/client'"
```bash
cd api
npm run prisma:generate
```

### Environment Variables Not Saving
- Check you're entering values correctly
- Verify Vercel login: `vercel whoami`
- Try setting via dashboard instead

---

## 📝 Alternative: Manual Deployment

If you prefer step-by-step control:

```bash
# Follow the manual guide
cat DEPLOY_MANUAL.md
```

Or open: `DEPLOY_MANUAL.md`

---

## ✅ Final Verification

After deployment, verify:

1. **Production Health Check:**
   ```bash
   curl https://your-app.vercel.app/api/health
   ```

2. **Production Test Endpoint:**
   ```bash
   curl https://your-app.vercel.app/api/test-public
   ```

3. **Frontend Loads:**
   - Visit: `https://your-app.vercel.app`
   - Check browser console (F12)
   - Test login/signup

4. **Environment Variables:**
   ```bash
   vercel env ls
   ```

---

## 🎯 Quick Reference

**Before Running:**
- [ ] Updated `DATABASE_URL` in `api/.env.local`
- [ ] Vercel CLI installed: `vercel --version`
- [ ] JWT_SECRET generated and ready
- [ ] Supabase values ready (if using)

**Run:**
```bash
./DEPLOY.sh
```

**After Deployment:**
- [ ] Test production endpoints
- [ ] Verify frontend loads
- [ ] Check environment variables
- [ ] Monitor Vercel dashboard

---

**Ready?** Update `DATABASE_URL` and run `./DEPLOY.sh`! 🚀


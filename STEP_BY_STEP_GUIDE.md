# MedTrack - Step-by-Step Deployment Guide

## ✅ Completed Steps

### Step 1: Install Dependencies ✓
- API: 154 packages installed
- Frontend: 518 packages installed
- Vercel CLI: Available

### Step 2: Generate Prisma Client ✓
- Prisma Client v5.22.0 generated
- Ready for database operations

---

## 📋 Current Steps

### Step A: Test Dev Server

**Important:** Before testing, you need a `DATABASE_URL` environment variable.

#### Quick Setup for Testing:

1. **Create `api/.env.local`:**
   ```bash
   cd api
   echo 'DATABASE_URL="postgresql://user:password@host:5432/database"' > .env.local
   ```

2. **Test API Server:**
   ```bash
   cd api
   npx vercel dev
   ```
   
   This starts the API on `http://localhost:3000`
   
   **Test endpoints:**
   ```bash
   # Health check
   curl http://localhost:3000/api/health
   
   # Test public
   curl http://localhost:3000/api/test-public
   
   # Login (will fail without DB, but tests routing)
   curl -X POST http://localhost:3000/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@test.com","password":"test"}'
   ```

3. **Test Frontend (separate terminal):**
   ```bash
   cd frontend
   npm run dev
   ```
   
   Open: `http://localhost:5173`
   
   **Test:**
   - Page loads without errors
   - Login form appears
   - API calls work (check browser console)

#### Expected Results:

✅ `/api/health` → `{"status":"OK","service":"medtrack-backend","timestamp":"..."}`
✅ `/api/test-public` → `{"message":"Backend is running!"}`
✅ Frontend loads → No console errors
✅ API calls from frontend → Work (may fail auth without DB, but routing works)

---

### Step B: Set Environment Variables for Vercel

**⚠️ CRITICAL: Cannot deploy without these!**

#### Required Variables:

| Variable | Purpose | Required |
|----------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection | ✅ Yes |
| `JWT_SECRET` | JWT token secret | ✅ Yes |
| `SUPABASE_URL` | Supabase URL (if using) | ⚠️ If using |
| `SUPABASE_ANON_KEY` | Supabase key (if using) | ⚠️ If using |
| `NODE_ENV` | Environment mode | ✅ Yes |
| `FRONTEND_URL` | Frontend URL for CORS | ✅ Yes |
| `CORS_ORIGIN` | CORS origin | ✅ Yes |

#### Method 1: Vercel CLI

```bash
# Login first
vercel login

# Set variables for production
vercel env add DATABASE_URL production
# Paste your PostgreSQL connection string when prompted

vercel env add JWT_SECRET production
# Generate: node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"

vercel env add NODE_ENV production
# Enter: production

vercel env add FRONTEND_URL production
# Enter: https://your-app.vercel.app (after first deploy)

vercel env add CORS_ORIGIN production
# Enter: https://your-app.vercel.app

# If using Supabase:
vercel env add SUPABASE_URL production
vercel env add SUPABASE_ANON_KEY production
```

#### Method 2: Vercel Dashboard

1. Go to: https://vercel.com/dashboard
2. Select your project (or create new)
3. Settings → Environment Variables
4. Add each variable:
   - Name: `DATABASE_URL`
   - Value: Your connection string
   - Environment: Select "Production", "Preview", "Development"
5. Repeat for all variables

#### Generate JWT_SECRET:

```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

Copy the output and use it as `JWT_SECRET`.

#### Verify Variables:

```bash
vercel env ls
```

---

### Step C: Convert Remaining Routes (Optional but Recommended)

**Status:** 15+ routes converted, ~30+ remaining

**Why convert before deployment:**
- Prevents broken features in production
- Easier to debug locally
- Better user experience

**Routes to convert:**

From `backend/simple-server.js`:
- `/api/doctor/parse-history`
- `/api/doctor/intelligent-parse`
- `/api/auth/survey-status`
- `/api/auth/survey-data`
- `/api/auth/complete-survey`
- `/api/doctor/patients/:patientId`
- `/api/doctor/audit-logs/*`
- `/api/metrics/patient/:patientId`
- `/api/lab-results/patient/:patientId`
- `/api/vital-signs/patient/:patientId`
- `/api/ai/*` endpoints

From `backend/src/routes/`:
- All route files need individual serverless functions

**Conversion Pattern:**

1. Look at existing route: `api/auth/login.ts`
2. Copy pattern to new file: `api/{module}/{route}.ts`
3. Export default handler
4. Use shared utilities: `import { prisma } from '../lib/prisma'`
5. Test locally immediately

**Example:**

```typescript
// api/auth/survey-status.ts
import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from '../lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Your logic here
  const result = await prisma.user.findMany();
  res.json(result);
}
```

---

### Step D: Deploy to Vercel

**Prerequisites:**
- ✅ All dependencies installed
- ✅ Prisma client generated
- ✅ Environment variables set
- ✅ Local testing passed

**Deploy Command:**

```bash
vercel --prod
```

**What happens:**
1. Vercel detects monorepo structure
2. Builds frontend (Vite build)
3. Deploys API functions
4. Configures routing
5. Provides live URLs

**After deployment:**
- Frontend URL: `https://your-app.vercel.app`
- API URL: `https://your-app.vercel.app/api/*`

**Test production:**
```bash
curl https://your-app.vercel.app/api/health
curl https://your-app.vercel.app/api/test-public
```

---

## 🐛 Troubleshooting

### API Not Starting Locally
- Check `DATABASE_URL` is set in `api/.env.local`
- Check port 3000 is available
- Check Vercel CLI: `npx vercel --version`

### Prisma Errors
- Run: `cd api && npm run prisma:generate`
- Verify `DATABASE_URL` is correct
- Check `api/prisma/schema.prisma` exists

### Import Errors
- Check TypeScript: `cd api && npm run typecheck`
- Verify dependencies: `cd api && npm install`

### Vercel Deployment Fails
- Check environment variables are set
- Check `vercel.json` syntax
- Check build logs in Vercel dashboard
- Verify Prisma client is generated

---

## 📊 Progress Checklist

- [x] Step 1: Install dependencies
- [x] Step 2: Generate Prisma client
- [ ] Step A: Test dev server locally
- [ ] Step B: Set environment variables
- [ ] Step C: Convert remaining routes (optional)
- [ ] Step D: Deploy to Vercel
- [ ] Step E: Test production endpoints
- [ ] Step F: Monitor and fix issues

---

## 🎯 Recommended Workflow

1. **Test locally first** (Step A)
   - Fix any issues before deploying
   - Verify all endpoints work

2. **Set environment variables** (Step B)
   - Critical for deployment
   - Test with local `.env.local` first

3. **Convert routes incrementally** (Step C)
   - Convert 5-10 routes at a time
   - Test each batch locally
   - Commit after each batch works

4. **Deploy to production** (Step D)
   - Only after local testing passes
   - Monitor logs for errors
   - Test all critical endpoints

---

## 📚 Documentation Files

- `README.md` - Full setup guide
- `STRUCTURE.md` - Project structure
- `DEPLOYMENT_STATUS.md` - Deployment progress
- `TEST_DEV_SERVER.md` - Testing instructions
- `ENV_SETUP.md` - Environment variables guide
- `STEP_BY_STEP_GUIDE.md` - This file

---

**Last Updated:** After Steps 1-2 completion
**Next Action:** Test dev server (Step A) or set environment variables (Step B)

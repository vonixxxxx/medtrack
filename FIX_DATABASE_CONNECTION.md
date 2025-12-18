# Fix: Database Connection Error

## Problem

```
PrismaClientInitializationError: Can't reach database server at 
`db.ydfksxcktsjhadiotlrc.supabase.co:5432`
```

## Root Cause

Supabase requires **connection pooling** for serverless functions (like Vercel). The direct connection (port 5432) doesn't work with serverless environments because:
- Serverless functions have connection limits
- Direct connections can exhaust the database connection pool
- Connection pooling manages connections efficiently

## Solution

Use Supabase's **Connection Pooler** instead of direct connection.

### Step 1: Get Connection Pooler URL from Supabase

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Go to **Settings** → **Database**
4. Scroll to **Connection Pooling** section
5. Copy the **Connection string** (it uses port **6543**)

### Step 2: Update DATABASE_URL in Vercel

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project: **medtrack**
3. Go to **Settings** → **Environment Variables**
4. Find `DATABASE_URL`
5. Update it with the connection pooler URL

### Connection String Format

**Wrong (Direct Connection - Port 5432):**
```
postgresql://postgres:password@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres
```

**Correct (Connection Pooler - Port 6543):**
```
postgresql://postgres:password@db.ydfksxcktsjhadiotlrc.supabase.co:6543/postgres?pgbouncer=true
```

### Important Notes

- **Port 6543** = Connection Pooler (for serverless)
- **Port 5432** = Direct Connection (for local/dev only)
- Add `?pgbouncer=true` parameter
- Use **Transaction** mode (not Session mode) for Prisma

### Step 3: Redeploy

After updating the environment variable:
1. Vercel will automatically redeploy
2. Or trigger a new deployment manually
3. Test signup again

## Verify Connection

After updating, check Vercel function logs to confirm connection works:
- Go to Vercel Dashboard → Your Project → Functions → View Logs
- Look for successful database queries

## Alternative: Check Supabase Status

If connection pooler still doesn't work:
1. Check if Supabase project is active
2. Verify database is not paused (free tier pauses after inactivity)
3. Check Supabase status page for outages

## Quick Fix Command

If you have the connection pooler URL, update it in Vercel CLI:

```bash
vercel env rm DATABASE_URL production
vercel env add DATABASE_URL production
# Paste the connection pooler URL when prompted
```

Then redeploy:
```bash
vercel --prod
```








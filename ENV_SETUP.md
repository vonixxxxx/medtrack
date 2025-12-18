# Environment Variables Setup - Step B

## Required Environment Variables

### For Local Development

Create `api/.env.local` (or set in your shell):

```bash
# Database (REQUIRED)
DATABASE_URL="postgresql://user:password@host:5432/database"

# Supabase (if using)
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_ANON_KEY="your-anon-key"
SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"

# Authentication (REQUIRED)
JWT_SECRET="your-super-secret-jwt-key-at-least-256-bits-long"
JWT_EXPIRES_IN="7d"

# Application
NODE_ENV="development"
PORT="3000"

# CORS
FRONTEND_URL="http://localhost:5173"
CORS_ORIGIN="http://localhost:5173"
```

### For Vercel Production

**Option 1: Using Vercel CLI**

```bash
# Login to Vercel first
vercel login

# Set environment variables
vercel env add DATABASE_URL production
vercel env add JWT_SECRET production
vercel env add SUPABASE_URL production
vercel env add SUPABASE_ANON_KEY production
vercel env add NODE_ENV production
vercel env add FRONTEND_URL production
vercel env add CORS_ORIGIN production

# For preview/development environments
vercel env add DATABASE_URL preview
vercel env add JWT_SECRET preview
# ... repeat for other vars
```

**Option 2: Using Vercel Dashboard**

1. Go to https://vercel.com/dashboard
2. Select your project
3. Go to Settings → Environment Variables
4. Add each variable:
   - Name: `DATABASE_URL`
   - Value: Your PostgreSQL connection string
   - Environment: Production, Preview, Development (select all)
5. Repeat for all variables

## Variable Descriptions

| Variable | Purpose | Required | Example |
|----------|---------|----------|---------|
| `DATABASE_URL` | PostgreSQL connection for Prisma | ✅ Yes | `postgresql://user:pass@host:5432/db` |
| `JWT_SECRET` | Secret for JWT token generation | ✅ Yes | Random 256-bit string |
| `SUPABASE_URL` | Supabase project URL | ⚠️ If using | `https://xxx.supabase.co` |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | ⚠️ If using | `eyJhbGc...` |
| `NODE_ENV` | Environment mode | ✅ Yes | `production` or `development` |
| `FRONTEND_URL` | Frontend URL for CORS | ✅ Yes | `https://your-app.vercel.app` |
| `CORS_ORIGIN` | Allowed CORS origin | ✅ Yes | `https://your-app.vercel.app` |

## Generate JWT_SECRET

```bash
# Generate a secure random secret
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

Or use an online generator: https://generate-secret.vercel.app/32

## Verify Environment Variables

**In Vercel CLI:**
```bash
vercel env ls
```

**In Vercel Dashboard:**
- Settings → Environment Variables
- Check all variables are set for correct environments

## Testing with Environment Variables

After setting variables:

1. **Local testing:**
   ```bash
   # Load from .env.local
   cd api
   npx vercel dev
   ```

2. **Production testing:**
   ```bash
   # Variables are automatically loaded from Vercel
   vercel --prod
   ```

## Troubleshooting

### "DATABASE_URL is not set"
- Check `.env.local` exists in `api/` directory
- Verify variable name is exactly `DATABASE_URL`
- Restart dev server after adding variables

### "Prisma Client not found"
- Run: `cd api && npm run prisma:generate`
- Check `DATABASE_URL` is correct
- Verify Prisma schema exists: `api/prisma/schema.prisma`

### Variables not loading in Vercel
- Check variable names match exactly (case-sensitive)
- Verify environment scope (production/preview/development)
- Redeploy after adding variables: `vercel --prod`

## Next Steps

After setting environment variables:
1. ✅ Test locally with `npm run dev`
2. ✅ Verify all endpoints work
3. ✅ Deploy to Vercel: `vercel --prod`
4. ✅ Test production endpoints








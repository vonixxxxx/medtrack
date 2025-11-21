# Supabase Edge Functions - Express Backend

This directory contains the Supabase Edge Function that wraps your Express.js backend.

## Quick Deploy

**Deploy with one command (recommended):**

```bash
cd supabase/functions/backend-express && ./deploy.sh
```

This script automatically:
1. Copies all necessary backend files to the function directory
2. Deploys the function to Supabase

**Or deploy manually:**

```bash
# First, copy backend files (or use the script)
cd supabase/functions/backend-express
./deploy.sh

# Or manually:
supabase functions deploy backend-express
```

## Prerequisites

1. **Install Supabase CLI:**
   ```bash
   npm install -g supabase
   ```

2. **Login to Supabase:**
   ```bash
   supabase login
   ```

3. **Link your project:**
   ```bash
   supabase link --project-ref your-project-ref
   ```
   
   Get your project ref from: https://app.supabase.com → Your Project → Settings → General → Reference ID

## Set Environment Variables/Secrets

Set all required environment variables as Supabase secrets:

```bash
supabase secrets set DATABASE_URL="your-postgresql-connection-string"
supabase secrets set JWT_SECRET="your-super-secure-secret-key-at-least-256-bits-long"
supabase secrets set JWT_EXPIRES_IN="7d"
supabase secrets set PORT="4000"
supabase secrets set NODE_ENV="production"
supabase secrets set FRONTEND_URL="https://your-frontend-domain.com"
supabase secrets set CORS_ORIGIN="https://your-frontend-domain.com"
supabase secrets set SMTP_HOST="smtp.gmail.com"
supabase secrets set SMTP_PORT="587"
supabase secrets set SMTP_USER="your-email@gmail.com"
supabase secrets set SMTP_PASS="your-app-password"
supabase secrets set SMTP_FROM="MedTrack <noreply@medtrack.com>"
supabase secrets set TOTP_ISSUER="MedTrack"
supabase secrets set TOTP_WINDOW="2"
supabase secrets set RATE_LIMIT_WINDOW_MS="900000"
supabase secrets set RATE_LIMIT_MAX_REQUESTS="100"
supabase secrets set AUTH_RATE_LIMIT_MAX="5"
```

See `backend/env.template` for the complete list of required variables.

## Access the Function

Once deployed, your Express backend will be available at:

```
https://your-project-ref.supabase.co/functions/v1/backend-express
```

All your existing Express routes work exactly as before:

- `https://your-project-ref.supabase.co/functions/v1/backend-express/api/medications/validateMedication`
- `https://your-project-ref.supabase.co/functions/v1/backend-express/api/doctor/parse-history`
- `https://your-project-ref.supabase.co/functions/v1/backend-express/api/doctor/patients`
- ... all other routes

## How It Works

1. The Edge Function loads your Express app from `backend/simple-server.js` unchanged
2. It converts Deno Request/Response to Express req/res objects
3. All your routes, middleware, Prisma, Sharp, and other dependencies work via Deno's Node.js compatibility layer
4. Zero refactoring required - your Express code is 100% unchanged

## Local Development

To test locally:

```bash
cd supabase/functions/backend-express
deno task dev
```

## Troubleshooting

### Backend files not found

If you get errors about missing backend files:

1. Ensure the `backend` folder is accessible from the function directory
2. The deploy script creates a symlink automatically
3. If symlinks don't work in your deployment environment, copy the backend files:
   ```bash
   cp -r ../../backend ./backend
   ```

### Prisma errors

Make sure:
- `DATABASE_URL` secret is set correctly
- Prisma client is generated: `cd backend && npx prisma generate`
- Database migrations are run on your Supabase database

### Module not found errors

All Node.js modules work via Deno's npm: compatibility. If you see module errors:
- Check that the module is in `backend/package.json`
- Deno automatically resolves npm: imports

## Notes

- The Express app from `backend/simple-server.js` is loaded unchanged
- All routes, middleware, and logic work exactly as before
- Prisma, Sharp, jsonwebtoken, nodemailer, and other Node.js dependencies work via Deno's Node.js compatibility layer
- Environment variables are automatically injected from Supabase secrets

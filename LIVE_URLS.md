# 🌐 LIVE URLs AFTER DEPLOY

## Backend API (Supabase Edge Function)
**URL:** `https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express`

All API endpoints are available at this base URL:
- `https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express/api/medications/validateMedication`
- `https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express/api/doctor/parse-history`
- `https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express/api/doctor/patients`
- `https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express/api/test-public`
- ... all other routes

## Frontend (Vercel)
**URL:** `https://your-vercel-url.vercel.app` (update after Vercel deployment)

After deploying to Vercel, your frontend will be available at the URL provided by Vercel.

## Custom Domain Setup

### Vercel Custom Domain

1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add your custom domain (e.g., `medtrack.com`)
3. Follow DNS configuration instructions provided by Vercel
4. Wait for DNS propagation (usually 5-60 minutes)

### Update Environment Variables After Custom Domain

Once your custom domain is active:

1. **Update Supabase Secrets:**
   ```bash
   supabase secrets set FRONTEND_URL="https://your-custom-domain.com"
   supabase secrets set CORS_ORIGIN="https://your-custom-domain.com"
   ```

2. **Redeploy Edge Function:**
   ```bash
   cd supabase/functions/backend-express
   supabase functions deploy backend-express
   ```

3. **Update Vercel Environment Variables:**
   - Go to Vercel Dashboard → Project → Settings → Environment Variables
   - Update `VITE_API_URL` if needed (should remain the same)
   - Redeploy frontend if necessary

## Database
**Supabase PostgreSQL:** `ydfksxcktsjhadiotlrc.supabase.co`
- **Dashboard:** https://app.supabase.com/project/ydfksxcktsjhadiotlrc
- **Connection String:** `postgresql://postgres.ydfksxcktsjhadiotlrc:fibbu6-foqJop-qydron@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres`

## Health Check

Test your backend is working:
```bash
curl https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express/api/test-public
```

Expected response: `{"message":"Public test endpoint working","timestamp":"..."}`


# Vercel Environment Variables - Quick Reference

## 📋 Copy-Paste Ready Values

Use these values in the Vercel Dashboard → Settings → Environment Variables

### Required Variables (Production)

**1. DATABASE_URL**
```
postgresql://postgres:tirpuV-sihsu7-rijjem@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres
```

**2. JWT_SECRET**
```
8a1ac4d831720f929941ac89de22dea979bbe7c5c4dee9a06ffc17e07d80a400
```

**3. SUPABASE_URL**
```
https://ydfksxcktsjhadiotlrc.supabase.co
```

**4. NODE_ENV**
```
production
```

**5. FRONTEND_URL** (update after first deploy with actual URL)
```
https://medtrack.vercel.app
```

**6. CORS_ORIGIN** (update after first deploy with actual URL)
```
https://medtrack.vercel.app
```

### Optional Variables (if using Supabase Auth)

**7. SUPABASE_ANON_KEY**
```
[Your Supabase anonymous key - if you have it]
```

## 🚀 How to Add in Vercel Dashboard

1. Go to: https://vercel.com/vonixs-projects/medtrack/settings/environment-variables

2. Click **"Create new"** for each variable

3. For each variable:
   - **Key**: Enter the variable name (e.g., `DATABASE_URL`)
   - **Value**: Paste the value from above
   - **Environments**: Select **Production**, **Preview**, and **Development**
   - Click **"Save"**

4. Repeat for all 6-7 variables

5. **Important**: After adding all variables, trigger a new deployment:
   - Go to: https://vercel.com/vonixs-projects/medtrack
   - Click **"Deployments"** tab
   - Click **"Redeploy"** on the latest deployment
   - Or wait for the next git push to auto-deploy

## ✅ Quick Checklist

- [ ] DATABASE_URL added
- [ ] JWT_SECRET added
- [ ] SUPABASE_URL added
- [ ] NODE_ENV added
- [ ] FRONTEND_URL added (can update after first deploy)
- [ ] CORS_ORIGIN added (can update after first deploy)
- [ ] SUPABASE_ANON_KEY added (if using)
- [ ] All variables set for Production environment
- [ ] New deployment triggered

## 📝 Notes

- **FRONTEND_URL** and **CORS_ORIGIN**: You can use `https://medtrack.vercel.app` as placeholder, then update with the actual URL after first successful deployment
- **Sensitive**: You can enable "Sensitive" for JWT_SECRET and DATABASE_URL to hide values
- **Environments**: Set for Production, Preview, and Development to work in all environments

## 🔍 Verify Variables Are Set

After adding, you can verify:
1. Go to: https://vercel.com/vonixs-projects/medtrack/settings/environment-variables
2. You should see all variables listed
3. Values will be hidden if marked as "Sensitive"

## 🚀 After Adding Variables

1. **Trigger new deployment** (required for variables to take effect)
2. **Monitor build logs** for any errors
3. **Test endpoints** after deployment completes:
   - `https://medtrack.vercel.app/api/health`
   - `https://medtrack.vercel.app/api/test-public`

---

**Quick Link**: https://vercel.com/vonixs-projects/medtrack/settings/environment-variables


# Fix: Vercel Hobby Plan Function Limit

## Problem

✅ **Build succeeded!** But deployment failed due to function limit:
- Vercel Hobby plan: **12 serverless functions max**
- Your project has: **More than 12 functions**

## Solution Options

### Option 1: Upgrade to Pro Plan (Recommended)

1. Go to: https://vercel.com/account/billing
2. Upgrade to **Pro plan** ($20/month)
3. Pro plan has **unlimited serverless functions**
4. Redeploy after upgrading

### Option 2: Consolidate Functions (Free Solution)

Combine related routes into single files using path-based routing:

#### Current Structure (Too Many Files):
```
api/
├── auth/
│   ├── login.ts
│   ├── signup.ts
│   └── me.ts
├── meds/
│   ├── user.ts
│   ├── schedule.ts
│   └── cycles.ts
└── ...
```

#### Consolidated Structure (12 or Fewer Files):
```
api/
├── auth.ts          (handles /api/auth/login, /api/auth/signup, /api/auth/me)
├── medications.ts   (handles all medication routes)
├── metrics.ts       (handles all metrics routes)
├── health.ts
└── ...
```

### Option 3: Remove Unused Functions

If some functions aren't needed for MVP:
- Remove unused API routes
- Keep only essential functionality
- Add more routes later after upgrading

## Quick Fix: Consolidate Auth Routes

Example of consolidating auth routes:

**Create `api/auth.ts`:**
```typescript
import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from './lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const path = req.url?.split('?')[0] || '';
  
  if (path === '/api/auth/login' && req.method === 'POST') {
    // Login logic
  } else if (path === '/api/auth/signup' && req.method === 'POST') {
    // Signup logic
  } else if (path === '/api/auth/me' && req.method === 'GET') {
    // Get current user logic
  } else {
    res.status(404).json({ error: 'Not found' });
  }
}
```

## Current Function Count

Check with:
```bash
find api -name "*.ts" -type f ! -path "*/node_modules/*" ! -path "*/src-temp/*" | wc -l
```

## Recommended Action

**For immediate deployment:**
1. Upgrade to Pro plan (easiest)
2. Or consolidate functions to 12 or fewer

**For long-term:**
- Consolidate related routes
- Use path-based routing in single files
- Keep function count manageable

---

**Quick Link**: https://vercel.com/account/billing

# Testing Dev Server - Step A

## Quick Test Commands

### 1. Start Dev Server
```bash
npm run dev
```

This starts:
- Frontend: http://localhost:5173
- API: http://localhost:3000

### 2. Test API Endpoints

**Health Check:**
```bash
curl http://localhost:3000/api/health
```

**Test Public:**
```bash
curl http://localhost:3000/api/test-public
```

**Login (POST):**
```bash
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test"}'
```

**Get Patients (requires auth):**
```bash
curl http://localhost:3000/api/doctor/patients
```

### 3. Test Frontend

Open browser: http://localhost:5173

Test:
- Login page loads
- Signup works
- API calls succeed
- No console errors

## Troubleshooting

### API Not Starting
- Check if port 3000 is available
- Check `api/.env` or environment variables
- Check Vercel CLI: `npx vercel dev --help`

### Prisma Errors
- Ensure `DATABASE_URL` is set
- Run: `cd api && npm run prisma:generate`
- Check `api/prisma/schema.prisma` exists

### Import Errors
- Check TypeScript compilation: `cd api && npm run typecheck`
- Verify all dependencies installed: `cd api && npm install`

## Expected Results

✅ `/api/health` returns: `{"status":"OK","service":"medtrack-backend","timestamp":"..."}`
✅ `/api/test-public` returns: `{"message":"Backend is running!"}`
✅ `/api/auth/login` returns: `{"success":true,"token":"...","user":{...}}` or error
✅ Frontend loads without errors
✅ API calls from frontend work

## Next Steps

Once all tests pass:
1. Set environment variables in Vercel (Step B)
2. Convert remaining routes (Step C)
3. Deploy to production (Step D)

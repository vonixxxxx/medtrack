# Quick Test Guide - Step A

## ⚡ Fast Setup

### 1. Update Database URL

Edit `api/.env.local` and replace the placeholder:

```bash
DATABASE_URL="postgresql://user:password@host:5432/database"
```

### 2. Start Servers

**Option A: Use the convenience script**
```bash
./START_DEV_SERVERS.sh
```

**Option B: Manual (two terminals)**

Terminal 1 - API:
```bash
cd api
npx vercel dev
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

### 3. Test Endpoints

**Health Check:**
```bash
curl http://localhost:3000/api/health
```
Expected: `{"status":"OK","service":"medtrack-backend","timestamp":"..."}`

**Test Public:**
```bash
curl http://localhost:3000/api/test-public
```
Expected: `{"message":"Backend is running!"}`

**Login (POST):**
```bash
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test"}'
```
Expected: Either success with token or error (both confirm routing works)

**Get Patients:**
```bash
curl http://localhost:3000/api/doctor/patients
```
Expected: Array of patients or empty array `[]`

### 4. Test Frontend

1. Open: http://localhost:5173
2. Check browser console (F12) - should be no errors
3. Test login form
4. Test signup form
5. Verify API calls work (check Network tab)

## ✅ Success Criteria

- [ ] API health endpoint returns 200 OK
- [ ] API test-public endpoint works
- [ ] Frontend loads without console errors
- [ ] Login/signup forms appear
- [ ] API calls from frontend succeed (check Network tab)
- [ ] No Prisma connection errors (if DATABASE_URL is correct)

## 🐛 Troubleshooting

### "Cannot find module '@prisma/client'"
```bash
cd api
npm run prisma:generate
```

### "DATABASE_URL is not set"
- Check `api/.env.local` exists
- Verify DATABASE_URL is not commented out
- Restart dev server after creating/editing .env.local

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

### API returns 404
- Check you're using `/api/` prefix
- Verify route file exists in `api/` directory
- Check Vercel dev server is running

### Frontend can't connect to API
- Verify API is running on port 3000
- Check `frontend/src/api.js` uses `/api` (relative path)
- Check browser console for CORS errors

## 📝 Next Steps

Once all tests pass:
1. ✅ Step A: Local testing (you are here)
2. ⏭️ Step B: Set environment variables for production
3. ⏭️ Step C: Convert remaining routes (optional)
4. ⏭️ Step D: Deploy to Vercel

See `STEP_BY_STEP_GUIDE.md` for complete instructions.

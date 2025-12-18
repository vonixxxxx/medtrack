# API Connection Errors - Fix Guide

## 🔴 Errors Identified

1. **ERR_CONNECTION_REFUSED** - Backend API server not running
2. **500 Internal Server Error** on `/api/auth/me` and `/api/auth/login`

## 🔍 Root Cause

The frontend (Vite app on port 3000) is trying to call:
- `/api/auth/me` → `http://localhost:3000/api/auth/me` ❌
- `/api/auth/login` → `http://localhost:3000/api/auth/login` ❌

But the backend API should be at:
- `http://localhost:4000/api/auth/me` ✅
- `http://localhost:4000/api/auth/login` ✅

**Problem**: The backend server is not running on port 4000.

## ✅ Solution

### Option 1: Start the Backend Server (Recommended)

1. **Navigate to backend directory**:
   ```bash
   cd /Users/AlexanderSokol/medtrack/backend
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   npm install
   ```

3. **Set up environment variables** (if needed):
   ```bash
   # Copy .env template if .env doesn't exist
   cp env.template .env
   # Edit .env with your configuration
   ```

4. **Start the backend server**:
   ```bash
   npm run dev
   # or
   npm start
   ```

5. **Verify backend is running**:
   ```bash
   curl http://localhost:4000/api/health
   # Should return a response
   ```

### Option 2: Configure Frontend to Use Correct API URL

Check `/Users/AlexanderSokol/medtrack/frontend/src/api.js`:

```javascript
const getBaseURL = () => {
  return import.meta.env.VITE_API_URL || '/api';
};
```

**Fix**: Set `VITE_API_URL` environment variable:

1. Create or edit `.env` in frontend directory:
   ```bash
   cd /Users/AlexanderSokol/medtrack/frontend
   echo "VITE_API_URL=http://localhost:4000/api" > .env.local
   ```

2. Restart the frontend dev server:
   ```bash
   npm run dev
   ```

### Option 3: Add Vite Proxy (Alternative)

Edit `frontend/vite.config.js` to proxy API requests:

```javascript
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:4000',
        changeOrigin: true,
      },
    },
  },
});
```

## 🚀 Quick Start Backend

```bash
# Terminal 1: Start Backend
cd /Users/AlexanderSokol/medtrack/backend
npm run dev

# Terminal 2: Start Frontend (if not already running)
cd /Users/AlexanderSokol/medtrack/frontend
npm run dev -- --port 3000
```

## ✅ Verification

After starting the backend:

1. **Check backend is running**:
   ```bash
   lsof -i :4000
   # Should show node process
   ```

2. **Test API endpoint**:
   ```bash
   curl http://localhost:4000/api/health
   ```

3. **Check browser console** - Errors should be gone

## 📝 Environment Variables Needed

### Backend (.env)
```env
DATABASE_URL=postgresql://user:password@localhost:5432/medtrack
JWT_SECRET=your-secret-key
PORT=4000
CORS_ORIGIN=http://localhost:3000
```

### Frontend (.env.local)
```env
VITE_API_URL=http://localhost:4000/api
```

---

**Status**: Backend server needs to be started
**Priority**: High - Required for authentication to work




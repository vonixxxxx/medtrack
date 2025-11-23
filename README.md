# MedTrack - Vercel Monorepo

MedTrack is a comprehensive medication tracking and health management application, restructured as a Vercel-ready monorepo.

## 📁 Project Structure

```
medtrack/
├── frontend/          # Vite + React + TypeScript frontend
│   ├── src/          # Frontend source code
│   ├── package.json  # Frontend dependencies
│   └── vite.config.js
│
├── api/              # Vercel serverless functions (backend)
│   ├── lib/          # Shared utilities (Prisma, auth)
│   ├── auth/         # Authentication endpoints
│   ├── doctor/       # Clinician endpoints
│   ├── medications/  # Medication endpoints
│   ├── meds/         # User medication endpoints
│   ├── metrics/      # Health metrics endpoints
│   ├── prisma/       # Prisma schema and migrations
│   └── *.ts          # Individual API route handlers
│
├── package.json      # Root package.json with dev scripts
├── vercel.json       # Vercel configuration
└── README.md         # This file
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Vercel CLI (`npm i -g vercel`)
- PostgreSQL database (Supabase recommended)

### Installation

1. **Install all dependencies:**
   ```bash
   npm run install:all
   ```

2. **Set up environment variables:**
   
   Create `.env` files:
   - `frontend/.env.local` - Frontend environment variables
   - `api/.env.local` - API environment variables (or use Vercel dashboard)
   
   Required variables:
   ```bash
   # Database
   DATABASE_URL="postgresql://..."
   
   # Supabase (if using)
   SUPABASE_URL="https://..."
   SUPABASE_ANON_KEY="..."
   
   # Frontend (optional - uses relative paths by default)
   VITE_API_URL="/api"  # or full URL for local dev
   ```

3. **Generate Prisma client:**
   ```bash
   cd api
   npm run prisma:generate
   ```

4. **Run database migrations:**
   ```bash
   cd api
   npm run prisma:migrate
   ```

### Local Development

**Option 1: Run everything together (recommended)**
```bash
npm run dev
```

This runs:
- Frontend dev server (Vite) on `http://localhost:5173`
- API serverless functions (Vercel Dev) on `http://localhost:3000`

**Option 2: Run separately**

Frontend only:
```bash
npm run dev:frontend
```

API only:
```bash
npm run dev:api
```

### Building for Production

```bash
npm run build
```

This builds the frontend for static deployment on Vercel.

## 📡 API Routes

All API routes are in the `/api` directory as individual TypeScript files. Vercel automatically maps these to serverless functions.

### Current Routes

- `GET /api/health` - Health check
- `GET /api/test-public` - Public test endpoint
- `GET /api/hello` - Hello world

**Authentication:**
- `GET /api/auth/me` - Get current user
- `POST /api/auth/login` - Login
- `POST /api/auth/signup` - Signup

**Doctor/Clinician:**
- `GET /api/doctor/patients` - Get all patients

**Medications:**
- `POST /api/medications/validateMedication` - Validate medication name
- `GET /api/meds/user` - Get user medications
- `POST /api/meds/user` - Create medication

**Metrics:**
- `GET /api/metrics/user` - Get user metrics
- `GET /api/health-metrics` - Get health metrics
- `GET /api/medication-schedules` - Get medication schedules

### Adding New Routes

1. Create a new `.ts` file in the appropriate directory under `/api`
2. Export a default handler function:

```typescript
import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from '../lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Your logic here
  res.json({ message: 'Success' });
}
```

3. The route will be available at `/api/[filename]`

## 🔄 Converting Express Routes

The backend has been partially converted from Express.js to Vercel serverless functions. To convert remaining routes:

### Express Route Pattern:
```javascript
app.get('/api/route', async (req, res) => {
  // logic
  res.json(data);
});
```

### Vercel Serverless Pattern:
```typescript
// api/route.ts
import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from '../lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Same logic, but use req.body, req.query instead of req.body, req.query
  res.json(data);
}
```

### Key Differences:

1. **No Express app** - Each file is a standalone handler
2. **Method checking** - Manually check `req.method`
3. **Imports** - Use `@vercel/node` types
4. **Prisma** - Import from `../lib/prisma` (singleton pattern)
5. **File structure** - One file per route, organized in folders

### Converting Route Files

For routes in `backend/src/routes/*.js`:

1. Create corresponding folder in `/api`
2. Convert each route handler to a separate `.ts` file
3. Update imports to use shared utilities from `/api/lib`
4. Test locally with `vercel dev`

## 🗄️ Database

The project uses Prisma ORM with PostgreSQL. The Prisma schema is in `/api/prisma/schema.prisma`.

### Prisma Commands

```bash
cd api

# Generate Prisma client
npm run prisma:generate

# Run migrations
npm run prisma:migrate

# Open Prisma Studio (GUI)
npx prisma studio
```

## 🚢 Deployment

### Deploy to Vercel

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Login:**
   ```bash
   vercel login
   ```

3. **Deploy:**
   ```bash
   vercel --prod
   ```

### Environment Variables

Set these in the Vercel dashboard (Project Settings → Environment Variables):

- `DATABASE_URL` - PostgreSQL connection string
- `SUPABASE_URL` - Supabase project URL (if using)
- `SUPABASE_ANON_KEY` - Supabase anonymous key
- `JWT_SECRET` - Secret for JWT tokens
- Any other backend secrets

### Build Configuration

Vercel automatically detects:
- Frontend build from `frontend/package.json` → `@vercel/static-build`
- API functions from `api/**/*.ts` → `@vercel/node`

The `vercel.json` configures routing:
- `/api/*` → API serverless functions
- `/*` → Frontend static files

## 🔧 Development Tips

### Frontend API Calls

The frontend uses relative paths by default (`/api/*`), which works in production. For local development with separate backend:

```bash
# frontend/.env.local
VITE_API_URL=http://localhost:4000/api
```

### Testing API Routes Locally

```bash
# Start Vercel dev server
cd api
vercel dev

# Test endpoint
curl http://localhost:3000/api/health
```

### Debugging

- Use `console.log` in serverless functions - logs appear in Vercel dashboard
- Use Vercel CLI: `vercel logs [deployment-url]`
- Frontend: Use browser DevTools as usual

## 📝 Remaining Work

### Routes to Convert

Many routes from `backend/simple-server.js` still need conversion:

- `/api/doctor/parse-history` - Complex medical history parsing
- `/api/doctor/intelligent-parse` - AI-powered parsing
- `/api/auth/survey-status` - Survey completion status
- `/api/auth/survey-data` - Save survey data
- `/api/auth/complete-survey` - Mark survey complete
- All routes from `backend/src/routes/*.js` files

### Utilities to Migrate

Copy and adapt utilities from `backend/utils/`:
- `intelligentMedicalParser.js`
- `ollamaParser.js`
- `biogptClient.js`
- `medicationMatchingService.js`
- Other utilities as needed

### Route Files to Convert

Convert routes from `backend/src/routes/`:
- `medication-tracking.js` → Multiple files in `/api/medications/`
- `ai.js` → `/api/ai/` routes
- `health-metrics.js` → `/api/health-metrics/` routes
- And all other route files

## 🐛 Troubleshooting

### Prisma Client Not Found

```bash
cd api
npm run prisma:generate
```

### Module Not Found Errors

Ensure all dependencies are installed:
```bash
npm run install:all
```

### Vercel Build Fails

- Check `vercel.json` syntax
- Ensure all TypeScript files compile: `cd api && npm run typecheck`
- Check environment variables in Vercel dashboard

### API Routes Return 404

- Ensure file is in `/api` directory
- File must export default handler function
- Check Vercel function logs in dashboard

## 📚 Resources

- [Vercel Serverless Functions Docs](https://vercel.com/docs/functions)
- [Prisma Docs](https://www.prisma.io/docs)
- [Vite Docs](https://vitejs.dev)

## 📄 License

MIT

---

**Note:** This is a work-in-progress conversion. Some routes may still need to be converted from Express to Vercel serverless functions. See "Remaining Work" section above.
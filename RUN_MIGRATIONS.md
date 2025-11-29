# Run Database Migrations

## Problem

The database tables don't exist yet. You need to run Prisma migrations to create them.

## Solution Options

### Option 1: Run Migrations via Supabase SQL Editor (Easiest)

1. Go to Supabase Dashboard → SQL Editor
2. Copy the migration SQL from: `api/prisma/migrations/20251028141800_init_postgresql_production/migration.sql`
3. Paste and run in SQL Editor
4. This will create all tables

### Option 2: Run Migrations Locally (Requires Direct Connection)

For migrations, you need the **direct connection** (port 5432), not the pooler:

1. Update `api/.env.local` with direct connection:
   ```
   DATABASE_URL="postgresql://postgres.ydfksxcktsjhadiotlrc:tirpuV-sihsu7-rijjem@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
   ```

2. Run migrations:
   ```bash
   cd api
   npx prisma migrate deploy
   ```

   Or use `db push` to sync schema:
   ```bash
   cd api
   npx prisma db push
   ```

### Option 3: Use Prisma Studio (Visual)

```bash
cd api
npx prisma studio
```

Then manually create tables or run migrations.

## Important Notes

- **For Migrations**: Use direct connection (port 5432)
- **For Application**: Use connection pooler (port 6543) - already set in Vercel
- Migrations only need to run once to create tables
- After tables are created, the app will work with the pooler connection

## Verify Tables Created

After running migrations, verify in Supabase:
- Go to Database → Tables
- You should see: `users`, `patients`, `medications`, etc.

## Quick Fix

The fastest way is to use Supabase SQL Editor and run the migration SQL directly.


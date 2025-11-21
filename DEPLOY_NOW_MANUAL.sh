#!/bin/bash
# Manual deployment steps - copy and paste these commands

set -e

echo "🚀 MEDTRACK DEPLOYMENT - MANUAL STEPS"
echo "======================================"
echo ""

# Generated JWT_SECRET
JWT_SECRET="baf0cf2081523f3dc2fdaf1eb1b5fcd4b9b1ed3f3881345a64fc97365dfb8de9"
echo "✅ JWT_SECRET Generated: $JWT_SECRET"
echo ""

echo "📋 STEP 1: Install Supabase CLI (if not installed)"
echo "   Run: brew install supabase/tap/supabase"
echo "   OR: https://github.com/supabase/cli#install-the-cli"
echo ""

echo "📋 STEP 2: Link Supabase Project"
echo "   Run: supabase link --project-ref ydfksxcktsjhadiotlrc"
echo ""

echo "📋 STEP 3: Set Supabase Secrets"
echo "   Run these commands:"
echo ""
echo "   supabase secrets set DATABASE_URL=\"postgresql://postgres.ydfksxcktsjhadiotlrc:fibbu6-foqJop-qydron@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres\""
echo "   supabase secrets set JWT_SECRET=\"$JWT_SECRET\""
echo "   supabase secrets set FRONTEND_URL=\"https://medtrack.vercel.app\""
echo "   supabase secrets set CORS_ORIGIN=\"https://medtrack.vercel.app\""
echo "   supabase secrets set NODE_ENV=\"production\""
echo ""

echo "📋 STEP 4: Deploy Backend Function"
echo "   cd supabase/functions/backend-express"
echo "   supabase functions deploy backend-express --no-verify-jwt"
echo ""

echo "📋 STEP 5: Run Prisma Migrations"
echo "   cd ../../backend"
echo "   export DATABASE_URL=\"postgresql://postgres.ydfksxcktsjhadiotlrc:fibbu6-foqJop-qydron@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres\""
echo "   npx prisma generate"
echo "   npx prisma migrate deploy"
echo ""

echo "📋 STEP 6: Deploy Frontend to Vercel"
echo "   cd ../frontend"
echo "   vercel login"
echo "   vercel --prod --yes"
echo ""
echo "   Then add these environment variables in Vercel Dashboard:"
echo "   - VITE_API_URL=https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express"
echo "   - VITE_SUPABASE_URL=https://ydfksxcktsjhadiotlrc.supabase.co"
echo "   - VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlkZmtzeGNrdHNqaGFkaW90bHJjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM0NDUxMDAsImV4cCI6MjA3OTAyMTEwMH0.0ijYSj2crCIa3KDrWab6uqaiwpR_q-V7vpjILyzFTfA"
echo ""

echo "✅ All steps completed! Your app will be live at:"
echo "   Backend: https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express"
echo "   Frontend: (Your Vercel URL)"
echo "   JWT_SECRET: $JWT_SECRET"


#!/bin/bash
# Complete Deployment Script for MedTrack
# Run this script after installing Supabase CLI and logging into Vercel

set -e

echo "🚀 MEDTRACK DEPLOYMENT SCRIPT"
echo "=============================="
echo ""

# Check if Supabase CLI is installed
if ! command -v supabase &> /dev/null; then
    echo "❌ Supabase CLI not found!"
    echo "   Install it with: brew install supabase/tap/supabase"
    echo "   OR download from: https://github.com/supabase/cli/releases"
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found!"
    echo "   Install it with: npm install -g vercel"
    exit 1
fi

echo "✅ Step 1: Linking Supabase project..."
supabase link --project-ref ydfksxcktsjhadiotlrc

echo ""
echo "✅ Step 2: Setting Supabase secrets..."
supabase secrets set DATABASE_URL="postgresql://postgres.ydfksxcktsjhadiotlrc:fibbu6-foqJop-qydron@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
supabase secrets set JWT_SECRET="baf0cf2081523f3dc2fdaf1eb1b5fcd4b9b1ed3f3881345a64fc97365dfb8de9"
supabase secrets set FRONTEND_URL="https://medtrack.vercel.app"
supabase secrets set CORS_ORIGIN="https://medtrack.vercel.app"
supabase secrets set NODE_ENV="production"

echo ""
echo "✅ Step 3: Deploying backend function..."
cd supabase/functions/backend-express
supabase functions deploy backend-express --no-verify-jwt

echo ""
echo "✅ Step 4: Running Prisma migrations..."
cd ../../backend
export DATABASE_URL="postgresql://postgres.ydfksxcktsjhadiotlrc:fibbu6-foqJop-qydron@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
npx prisma migrate deploy

echo ""
echo "✅ Step 5: Deploying frontend to Vercel..."
cd ../frontend
vercel --prod --yes

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║           ✅ MEDTRACK IS LIVE!                               ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🔗 Backend: https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express"
echo "🔗 Frontend: (Check Vercel output above for your URL)"
echo "🔑 JWT_SECRET: baf0cf2081523f3dc2fdaf1eb1b5fcd4b9b1ed3f3881345a64fc97365dfb8de9"
echo ""


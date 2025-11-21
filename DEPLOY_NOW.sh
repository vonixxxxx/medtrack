#!/bin/bash
# Quick deployment script for MedTrack
# This script automates the deployment process

set -e

echo "🚀 MedTrack Deployment Script"
echo "=============================="
echo ""

# Check if Supabase CLI is installed
if ! command -v supabase &> /dev/null; then
    echo "❌ Supabase CLI not found. Installing..."
    npm install -g supabase
fi

# Step 1: Link Supabase project
echo "📌 Step 1: Linking Supabase project..."
if [ ! -f "supabase/.temp/project-ref" ]; then
    echo "   Linking to project: ydfksxcktsjhadiotlrc"
    supabase link --project-ref ydfksxcktsjhadiotlrc || {
        echo "   ⚠️  Already linked or need to login first"
        echo "   Run: supabase login"
    }
else
    echo "   ✅ Project already linked"
fi

# Step 2: Set secrets
echo ""
echo "📌 Step 2: Setting Supabase secrets..."
echo "   ⚠️  IMPORTANT: Review supabase/secrets.txt first!"
echo "   Some secrets need to be updated (JWT_SECRET, SMTP credentials, etc.)"
read -p "   Have you updated supabase/secrets.txt? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Setting secrets from supabase/secrets.txt..."
    grep "^supabase secrets set" supabase/secrets.txt | bash || {
        echo "   ⚠️  Some secrets may have failed. Check manually."
    }
else
    echo "   ⚠️  Skipping secrets. Set them manually from supabase/secrets.txt"
fi

# Step 3: Deploy Edge Function
echo ""
echo "📌 Step 3: Deploying Supabase Edge Function..."
cd supabase/functions/backend-express
./deploy.sh
cd ../../..

# Step 4: Prisma migrations
echo ""
echo "📌 Step 4: Running Prisma migrations..."
cd backend
export DATABASE_URL="postgresql://postgres.ydfksxcktsjhadiotlrc:fibbu6-foqJop-qydron@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
npx prisma generate
npx prisma migrate deploy
cd ..

echo ""
echo "✅ Backend deployment complete!"
echo ""
echo "📋 Next Steps:"
echo "   1. Deploy frontend to Vercel (see DEPLOYMENT_CHECKLIST.md)"
echo "   2. Add Vercel environment variables (see vercel-env-vars.txt)"
echo "   3. Update FRONTEND_URL and CORS_ORIGIN secrets after Vercel deploy"
echo ""
echo "🔗 Backend URL: https://ydfksxcktsjhadiotlrc.supabase.co/functions/v1/backend-express"


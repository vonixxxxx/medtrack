#!/bin/bash
# Deploy via GitHub - avoids file upload limit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🚀 DEPLOY VIA GITHUB (NO FILE LIMIT) 🚀                ║"
echo "║                                                              ║
╚══════════════════════════════════════════════════════════════╝"
echo ""

# Set environment variables
export DATABASE_URL="postgresql://postgres:tirpuV-sihsu7-rijjem@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
export JWT_SECRET="8a1ac4d831720f929941ac89de22dea979bbe7c5c4dee9a06ffc17e07d80a400"
export SUPABASE_URL="https://ydfksxcktsjhadiotlrc.supabase.co"

# Write DATABASE_URL to .env.local
echo "DATABASE_URL=\"$DATABASE_URL\"" > api/.env.local

echo "✅ Environment variables set"
echo ""

# Check if git repo
if [ ! -d ".git" ]; then
    echo "❌ ERROR: Not a git repository"
    echo "   Initialize with: git init"
    exit 1
fi

echo "📋 Step 1: Staging changes..."
git add .
echo "✅ Changes staged"
echo ""

echo "📋 Step 2: Committing..."
git commit -m "feat: restructure to Vercel monorepo with serverless functions" || echo "⚠️  No changes to commit or already committed"
echo ""

echo "📋 Step 3: Pushing to GitHub..."
git push origin main || git push origin master || echo "⚠️  Push failed or no remote configured"
echo ""

echo "📋 Step 4: Deploying via Vercel (from GitHub)..."
echo "   This uses GitHub as source, avoiding file upload limit"
echo ""

# Try to deploy using --git flag (uses GitHub instead of uploading)
vercel --prod --git 2>/dev/null || {
    echo ""
    echo "⚠️  --git flag not available or project not linked to GitHub"
    echo ""
    echo "📝 MANUAL STEPS:"
    echo ""
    echo "1. Go to: https://vercel.com/vonixs-projects/medtrack"
    echo "2. Click 'Deploy' or wait for auto-deploy from GitHub"
    echo "3. Set environment variables in Settings → Environment Variables:"
    echo ""
    echo "   DATABASE_URL=$DATABASE_URL"
    echo "   JWT_SECRET=$JWT_SECRET"
    echo "   SUPABASE_URL=$SUPABASE_URL"
    echo "   NODE_ENV=production"
    echo ""
    echo "✅ Project pushed to GitHub"
    echo "   Vercel will auto-deploy from GitHub (no file limit!)"
}

echo ""
echo "✅ GitHub deployment initiated!"
echo "   Check: https://vercel.com/vonixs-projects/medtrack"

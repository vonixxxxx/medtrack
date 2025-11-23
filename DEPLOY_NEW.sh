#!/bin/bash
# Deploy as new project or update existing with correct settings

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🚀 DEPLOY MEDTRACK MONOREPO 🚀                          ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Project directory: $SCRIPT_DIR"
echo ""

# Set environment variables
export DATABASE_URL="postgresql://postgres:tirpuV-sihsu7-rijjem@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
export JWT_SECRET="8a1ac4d831720f929941ac89de22dea979bbe7c5c4dee9a06ffc17e07d80a400"
export SUPABASE_URL="https://ydfksxcktsjhadiotlrc.supabase.co"

# Write DATABASE_URL to .env.local
echo "DATABASE_URL=\"$DATABASE_URL\"" > api/.env.local

echo "✅ Environment variables set"
echo ""

# Remove old .vercel if it exists (to start fresh)
if [ -d ".vercel" ]; then
    echo "⚠️  Removing old .vercel configuration..."
    rm -rf .vercel
    echo "✅ Old configuration removed"
    echo ""
fi

# Verify vercel.json exists
if [ ! -f "vercel.json" ]; then
    echo "❌ ERROR: vercel.json not found in $(pwd)"
    exit 1
fi

echo "✅ vercel.json found"
echo ""

# Deploy - this will prompt to create new project or link existing
echo "🚀 Deploying to Vercel..."
echo "   When prompted:"
echo "   - Choose to create NEW project (recommended)"
echo "   - Or link to existing project (if you want to update existing)"
echo "   - Project name: medtrack (or your preferred name)"
echo ""

vercel --prod

echo ""
echo "✅ Deployment complete!"
echo "   Check Vercel dashboard for your live URL"

#!/bin/bash
# Deploy using archive mode to reduce file count

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🚀 DEPLOY WITH ARCHIVE (REDUCED FILE COUNT) 🚀          ║"
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

# Verify vercel.json exists
if [ ! -f "vercel.json" ]; then
    echo "❌ ERROR: vercel.json not found"
    exit 1
fi

echo "✅ vercel.json found"
echo ""

# Deploy with archive mode to reduce file count
echo "🚀 Deploying to Vercel with --archive=tgz..."
echo "   This reduces the number of files uploaded"
echo ""

vercel --prod --archive=tgz

echo ""
echo "✅ Deployment complete!"
echo "   Check Vercel dashboard for your live URL"

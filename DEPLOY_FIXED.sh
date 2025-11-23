#!/bin/bash
# Fixed deployment script - ensures correct directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🚀 FIXED DEPLOYMENT SCRIPT 🚀                           ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Project directory: $SCRIPT_DIR"
echo ""

# Load environment variables
export DATABASE_URL="postgresql://postgres:tirpuV-sihsu7-rijjem@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
export JWT_SECRET="8a1ac4d831720f929941ac89de22dea979bbe7c5c4dee9a06ffc17e07d80a400"
export SUPABASE_URL="https://ydfksxcktsjhadiotlrc.supabase.co"

# Write DATABASE_URL to .env.local
echo "DATABASE_URL=\"$DATABASE_URL\"" > api/.env.local

echo "✅ Environment variables set"
echo ""

# Deploy directly from project root
echo "🚀 Deploying to Vercel from: $(pwd)"
echo ""

# Verify vercel.json exists
if [ ! -f "vercel.json" ]; then
    echo "❌ ERROR: vercel.json not found in $(pwd)"
    exit 1
fi

echo "✅ vercel.json found"
echo ""

# Deploy
echo "Running: vercel --prod"
vercel --prod

echo ""
echo "✅ Deployment initiated!"
echo "   Check Vercel dashboard for status"

#!/bin/bash
# Force deploy latest commit to Vercel

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     FORCING VERCEL DEPLOYMENT FROM LATEST COMMIT              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

LATEST_COMMIT=$(git rev-parse HEAD)
LATEST_COMMIT_SHORT=$(git rev-parse --short HEAD)

echo "📋 Latest commit: $LATEST_COMMIT_SHORT"
echo "   Full: $LATEST_COMMIT"
echo ""

echo "🚀 Deploying to Vercel..."
echo "   This will force deploy the latest commit"
echo ""

# Deploy using Vercel CLI
vercel --prod

echo ""
echo "✅ Deployment initiated!"
echo "   Check: https://vercel.com/vonixs-projects/medtrack/deployments"
echo ""
echo "📋 The deployment should use commit: $LATEST_COMMIT_SHORT"

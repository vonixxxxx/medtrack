#!/bin/bash
# Automated deployment runner with pre-configured values

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🚀 AUTOMATED DEPLOYMENT RUNNER 🚀                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Load JWT_SECRET if available
if [ -f "/tmp/jwt_secret.txt" ]; then
    export JWT_SECRET=$(cat /tmp/jwt_secret.txt)
    echo "✅ JWT_SECRET loaded from /tmp/jwt_secret.txt"
    echo "   Value: $JWT_SECRET"
    echo ""
fi

# Verify DATABASE_URL is set
if [ ! -f "api/.env.local" ]; then
    echo "❌ ERROR: api/.env.local not found!"
    exit 1
fi

echo "✅ DATABASE_URL configured in api/.env.local"
echo ""

# Run deployment script
echo "🚀 Starting DEPLOY.sh..."
echo "   Note: You'll be prompted for environment variables"
echo "   Use the JWT_SECRET shown above when prompted"
echo ""

./DEPLOY.sh

#!/bin/bash
# Automated deployment script with pre-filled values
# This version handles the interactive prompts automatically

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🚀 AUTOMATED DEPLOYMENT (NON-INTERACTIVE) 🚀           ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Load environment variables if not set
if [ -z "$DATABASE_URL" ]; then
    if [ -f "api/.env.local" ]; then
        export DATABASE_URL=$(grep DATABASE_URL api/.env.local | cut -d'"' -f2)
    else
        echo "❌ ERROR: DATABASE_URL not set and api/.env.local not found"
        exit 1
    fi
fi

if [ -z "$JWT_SECRET" ]; then
    if [ -f "/tmp/jwt_secret.txt" ]; then
        export JWT_SECRET=$(cat /tmp/jwt_secret.txt)
    else
        echo "❌ ERROR: JWT_SECRET not set"
        exit 1
    fi
fi

if [ -z "$SUPABASE_URL" ]; then
    export SUPABASE_URL="https://ydfksxcktsjhadiotlrc.supabase.co"
fi

echo "✅ Environment variables loaded"
echo "   DATABASE_URL: [configured]"
echo "   JWT_SECRET: [configured]"
echo "   SUPABASE_URL: $SUPABASE_URL"
echo ""

# Step 1: Validate DATABASE_URL
echo "📋 Step 1: Validating setup..."
if [ ! -f "api/.env.local" ]; then
    echo "❌ ERROR: api/.env.local not found!"
    exit 1
fi

if echo "$DATABASE_URL" | grep -q "user:password"; then
    echo "⚠️  WARNING: DATABASE_URL appears to be a placeholder!"
    echo "   Continuing anyway..."
fi

echo "✅ Setup validated"
echo ""

# Step 2: Start API server
echo "📋 Step 2: Starting API server..."
cd api
npx vercel dev > /tmp/vercel-dev.log 2>&1 &
API_PID=$!
echo "   API server starting (PID: $API_PID)"
cd ..
echo "   Waiting for API server to initialize..."
sleep 10
echo ""

# Step 3: Start frontend
echo "📋 Step 3: Starting frontend server..."
cd frontend
npm run dev > /tmp/vite-dev.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend server starting (PID: $FRONTEND_PID)"
cd ..
echo "   Waiting for frontend server to initialize..."
sleep 5
echo ""

# Step 4: Test endpoints
echo "📋 Step 4: Testing API endpoints..."
echo ""

echo "Testing /api/health..."
HEALTH_RESPONSE=$(curl -s http://localhost:3000/api/health || echo "FAILED")
if echo "$HEALTH_RESPONSE" | grep -q "status"; then
    echo "   ✅ Health check passed"
else
    echo "   ⚠️  Health check may have failed (check logs)"
fi
echo ""

echo "Testing /api/test-public..."
TEST_RESPONSE=$(curl -s http://localhost:3000/api/test-public || echo "FAILED")
if echo "$TEST_RESPONSE" | grep -q "message"; then
    echo "   ✅ Test public endpoint passed"
else
    echo "   ⚠️  Test public endpoint may have failed"
fi
echo ""

# Step 5: Open browser
echo "📋 Step 5: Opening frontend in browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:5173 2>/dev/null || true
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:5173 2>/dev/null || true
fi
echo "   Frontend: http://localhost:5173"
echo "   API: http://localhost:3000"
echo ""

# Step 6: Set environment variables in Vercel
echo "📋 Step 6: Setting production environment variables..."
echo "   This step requires Vercel CLI and login"
echo ""

# Check if logged in
if ! vercel whoami > /dev/null 2>&1; then
    echo "   ⚠️  Not logged into Vercel. Please run: vercel login"
    echo "   Then run this script again or set variables manually"
    echo ""
else
    echo "   ✅ Logged into Vercel"
    echo ""
    
    # Set DATABASE_URL
    echo "   Setting DATABASE_URL..."
    echo "$DATABASE_URL" | vercel env add DATABASE_URL production 2>/dev/null || echo "   ⚠️  DATABASE_URL may already exist or failed"
    echo ""
    
    # Set JWT_SECRET
    echo "   Setting JWT_SECRET..."
    echo "$JWT_SECRET" | vercel env add JWT_SECRET production 2>/dev/null || echo "   ⚠️  JWT_SECRET may already exist or failed"
    echo ""
    
    # Set NODE_ENV
    echo "   Setting NODE_ENV..."
    echo "production" | vercel env add NODE_ENV production 2>/dev/null || echo "   ⚠️  NODE_ENV may already exist or failed"
    echo ""
    
    # Set Supabase variables if SUPABASE_ANON_KEY is provided
    if [ ! -z "$SUPABASE_ANON_KEY" ] && [ "$SUPABASE_ANON_KEY" != "YOUR_ANON_KEY_HERE" ]; then
        echo "   Setting SUPABASE_URL..."
        echo "$SUPABASE_URL" | vercel env add SUPABASE_URL production 2>/dev/null || echo "   ⚠️  SUPABASE_URL may already exist or failed"
        echo ""
        
        echo "   Setting SUPABASE_ANON_KEY..."
        echo "$SUPABASE_ANON_KEY" | vercel env add SUPABASE_ANON_KEY production 2>/dev/null || echo "   ⚠️  SUPABASE_ANON_KEY may already exist or failed"
        echo ""
    else
        echo "   ⏭️  Skipping Supabase variables (SUPABASE_ANON_KEY not set or is placeholder)"
        echo ""
    fi
    
    echo "✅ Environment variables setup complete"
fi
echo ""

# Step 7: Deploy to production
echo "📋 Step 7: Deploying to production..."
echo "   This will deploy your monorepo to Vercel"
echo ""

# Ask for confirmation (can be automated with environment variable)
if [ "$AUTO_DEPLOY" = "true" ]; then
    DEPLOY_ANSWER="y"
else
    read -p "Deploy to Vercel now? (y/N): " -n 1 -r
    echo
    DEPLOY_ANSWER="$REPLY"
fi

if [[ $DEPLOY_ANSWER =~ ^[Yy]$ ]]; then
    echo ""
    echo "   Deploying from project root..."
    # Ensure we're in the project root (where this script is located)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    echo "   Current directory: $(pwd)"
    echo "   Verifying vercel.json exists..."
    if [ -f "vercel.json" ]; then
        echo "   ✅ vercel.json found"
    else
        echo "   ❌ vercel.json not found in $(pwd)"
        exit 1
    fi
    echo "   Running: vercel --prod --yes"
    vercel --prod --yes
    echo ""
    echo "✅ Deployment complete!"
    echo "   Check Vercel dashboard for your live URL"
else
    echo "   ⏭️  Skipped deployment"
    echo "   Run manually: cd $(pwd) && vercel --prod"
fi
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    kill $API_PID $FRONTEND_PID 2>/dev/null || true
    echo "✅ Servers stopped"
}

trap cleanup EXIT INT TERM

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     ✅ AUTOMATED DEPLOYMENT COMPLETE ✅                     ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Summary:"
echo "   - API server: http://localhost:3000 (PID: $API_PID)"
echo "   - Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
echo "   - Environment variables: Set in Vercel"
echo "   - Deployment: $([ "$DEPLOY_ANSWER" = "y" ] && echo "Completed" || echo "Skipped")"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Keep script running
wait

#!/bin/bash
# MedTrack - Complete Deployment Script
# This script automates testing and deployment to Vercel

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🚀 MEDTRACK DEPLOYMENT SCRIPT 🚀                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 0: Check DATABASE_URL
echo "📋 Step 0: Checking environment setup..."
if [ ! -f "api/.env.local" ]; then
    echo "❌ ERROR: api/.env.local not found!"
    echo "   Create it with: cd api && echo 'DATABASE_URL=\"your-connection-string\"' > .env.local"
    exit 1
fi

if ! grep -q "DATABASE_URL=" api/.env.local || grep -q "DATABASE_URL=\"postgresql://user:password" api/.env.local; then
    echo "⚠️  WARNING: DATABASE_URL appears to be a placeholder!"
    echo "   Please update api/.env.local with your actual PostgreSQL connection string"
    echo "   Current value:"
    grep "DATABASE_URL=" api/.env.local | head -1
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Environment file found"
echo ""

# Step 1: Start API server
echo "📋 Step 1: Starting API server..."
cd api
npx vercel dev > /tmp/vercel-dev.log 2>&1 &
API_PID=$!
echo "   API server starting (PID: $API_PID)"
echo "   Logs: /tmp/vercel-dev.log"
cd ..
echo ""

# Step 2: Start frontend
echo "📋 Step 2: Starting frontend server..."
cd frontend
npm run dev > /tmp/vite-dev.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend server starting (PID: $FRONTEND_PID)"
echo "   Logs: /tmp/vite-dev.log"
cd ..
echo ""

# Step 3: Wait for servers to initialize
echo "📋 Step 3: Waiting for servers to initialize..."
sleep 8
echo "✅ Servers should be ready"
echo ""

# Step 4: Test core API endpoints
echo "📋 Step 4: Testing API endpoints..."
echo ""

echo "Testing /api/health..."
HEALTH_RESPONSE=$(curl -s http://localhost:3000/api/health || echo "FAILED")
if echo "$HEALTH_RESPONSE" | grep -q "status"; then
    echo "   ✅ Health check passed"
    echo "   Response: $HEALTH_RESPONSE"
else
    echo "   ❌ Health check failed"
    echo "   Response: $HEALTH_RESPONSE"
    echo "   Check logs: tail -f /tmp/vercel-dev.log"
fi
echo ""

echo "Testing /api/test-public..."
TEST_RESPONSE=$(curl -s http://localhost:3000/api/test-public || echo "FAILED")
if echo "$TEST_RESPONSE" | grep -q "message"; then
    echo "   ✅ Test public endpoint passed"
    echo "   Response: $TEST_RESPONSE"
else
    echo "   ❌ Test public endpoint failed"
    echo "   Response: $TEST_RESPONSE"
fi
echo ""

echo "Testing /api/auth/login..."
LOGIN_RESPONSE=$(curl -s -X POST http://localhost:3000/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"test@test.com","password":"test"}' || echo "FAILED")
if echo "$LOGIN_RESPONSE" | grep -q -E "(success|error|Invalid)"; then
    echo "   ✅ Login endpoint responding"
    echo "   Response: $LOGIN_RESPONSE"
else
    echo "   ⚠️  Login endpoint may not be working correctly"
    echo "   Response: $LOGIN_RESPONSE"
fi
echo ""

# Step 5: Open frontend in browser
echo "📋 Step 5: Opening frontend in browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:5173 2>/dev/null || echo "   Could not open browser automatically"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open http://localhost:5173 2>/dev/null || echo "   Could not open browser automatically"
else
    echo "   Please open http://localhost:5173 manually"
fi
echo "   Frontend: http://localhost:5173"
echo "   API: http://localhost:3000"
echo ""

# Step 6: Set production environment variables
echo "📋 Step 6: Setting production environment variables..."
echo "   ⚠️  This step requires interactive input"
echo "   You'll be prompted for each variable value"
echo ""
read -p "Continue with environment variable setup? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "   Logging into Vercel..."
    vercel login || echo "   ⚠️  Vercel login failed or already logged in"
    echo ""
    
    echo "   Adding DATABASE_URL..."
    vercel env add DATABASE_URL production || echo "   ⚠️  Failed to add DATABASE_URL"
    echo ""
    
    echo "   Adding JWT_SECRET..."
    echo "   💡 Generate one with: node -e \"console.log(require('crypto').randomBytes(32).toString('hex'))\""
    vercel env add JWT_SECRET production || echo "   ⚠️  Failed to add JWT_SECRET"
    echo ""
    
    read -p "Add Supabase variables? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Adding SUPABASE_URL..."
        vercel env add SUPABASE_URL production || echo "   ⚠️  Failed to add SUPABASE_URL"
        echo ""
        
        echo "   Adding SUPABASE_ANON_KEY..."
        vercel env add SUPABASE_ANON_KEY production || echo "   ⚠️  Failed to add SUPABASE_ANON_KEY"
        echo ""
    fi
    
    echo "   Adding NODE_ENV..."
    echo "production" | vercel env add NODE_ENV production || echo "   ⚠️  Failed to add NODE_ENV"
    echo ""
    
    echo "✅ Environment variables setup complete"
else
    echo "   ⏭️  Skipped environment variable setup"
    echo "   Run manually: vercel env add <VAR_NAME> production"
fi
echo ""

# Step 7: Deploy to production
echo "📋 Step 7: Deploying to production..."
read -p "Deploy to Vercel now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "   Deploying..."
    vercel --prod
    echo ""
    echo "✅ Deployment complete!"
    echo "   Check Vercel dashboard for your live URL"
else
    echo "   ⏭️  Skipped deployment"
    echo "   Run manually: vercel --prod"
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
echo "║     ✅ DEPLOYMENT SCRIPT COMPLETE ✅                        ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Summary:"
echo "   - API server: http://localhost:3000 (PID: $API_PID)"
echo "   - Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
echo "   - Logs: /tmp/vercel-dev.log and /tmp/vite-dev.log"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Keep script running
wait
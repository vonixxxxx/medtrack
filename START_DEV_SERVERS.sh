#!/bin/bash
# Start both dev servers (for convenience)

echo "Starting MedTrack dev servers..."
echo ""

# Check if .env.local exists
if [ ! -f "api/.env.local" ]; then
    echo "⚠️  WARNING: api/.env.local not found!"
    echo "   Please create it with your DATABASE_URL first"
    echo "   Example: cd api && echo 'DATABASE_URL=\"postgresql://...\"' > .env.local"
    exit 1
fi

echo "Starting API server on port 3000..."
cd api
npx vercel dev &
API_PID=$!
echo "API server PID: $API_PID"
cd ..

sleep 3

echo "Starting Frontend server on port 5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend server PID: $FRONTEND_PID"
cd ..

echo ""
echo "✅ Both servers starting..."
echo "   API: http://localhost:3000"
echo "   Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for user interrupt
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

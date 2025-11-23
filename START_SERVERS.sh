#!/bin/bash

# MedTrack - Start Development Servers
# This script starts both backend and frontend servers

echo "🚀 Starting MedTrack Development Servers..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Start Backend Server
echo -e "${BLUE}📦 Starting Backend Server...${NC}"
cd backend
PORT=4000 npm run dev &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start Frontend Server
echo -e "${BLUE}🎨 Starting Frontend Server...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for servers to be ready
sleep 5

echo ""
echo -e "${GREEN}✅ Servers Started!${NC}"
echo ""
echo -e "${YELLOW}📍 Access URLs:${NC}"
echo -e "   Backend API:  ${GREEN}http://localhost:4000/api${NC}"
echo -e "   Frontend App: ${GREEN}http://localhost:5173${NC}"
echo -e "   Clinician Dashboard: ${GREEN}http://localhost:5173/dashboard/clinician${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Keep script running
wait






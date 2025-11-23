#!/bin/bash
set -e

echo "🔧 Starting build process..."

echo "📦 Step 1: Generating Prisma client..."
cd api
npm run prisma:generate
cd ..

echo "🏗️  Step 2: Building frontend..."
cd frontend
# Use local vite from node_modules/.bin
./node_modules/.bin/vite build || npx --yes vite build
cd ..

echo "✅ Build completed successfully!"

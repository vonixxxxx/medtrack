#!/bin/bash
set -e

echo "🔧 Starting build process..."

echo "📦 Step 1: Generating Prisma client..."
cd api
npm run prisma:generate
cd ..

echo "🏗️  Step 2: Building frontend..."
cd frontend
# Use local vite from node_modules/.bin to ensure vite.config.js can find vite module
if [ -f "./node_modules/.bin/vite" ]; then
  ./node_modules/.bin/vite build
else
  npm run build
fi
cd ..

echo "✅ Build completed successfully!"

#!/bin/bash
set -e

echo "🔧 Starting build process..."

echo "📦 Step 1: Generating Prisma client..."
cd api
npm run prisma:generate
cd ..

echo "🏗️  Step 2: Building frontend..."
cd frontend
# Use node to run vite directly from installed package
# This ensures vite.config.js can find vite module from same node_modules
node node_modules/vite/bin/vite.js build
cd ..

echo "✅ Build completed successfully!"

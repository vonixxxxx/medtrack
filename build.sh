#!/bin/bash
set -e

echo "🔧 Starting build process..."

echo "📦 Step 1: Generating Prisma client..."
cd api
npm run prisma:generate
cd ..

echo "🏗️  Step 2: Building frontend..."
cd frontend
# Use npm run build which uses the fixed package.json script
# The package.json build script uses: node node_modules/vite/bin/vite.js build
# This ensures proper path resolution and module finding
npm run build
cd ..

echo "✅ Build completed successfully!"

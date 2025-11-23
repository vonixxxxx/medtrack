#!/bin/bash
set -e

echo "🔧 Starting build process..."

echo "📦 Step 1: Generating Prisma client..."
cd api
npm run prisma:generate
cd ..

echo "🏗️  Step 2: Building frontend..."
cd frontend
npm run build
cd ..

echo "✅ Build completed successfully!"

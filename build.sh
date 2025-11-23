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
# Set NODE_PATH to ensure vite module resolution works
export NODE_PATH=$(pwd)/node_modules
./node_modules/.bin/vite build
cd ..

echo "✅ Build completed successfully!"

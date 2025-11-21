#!/bin/bash
# Deployment script for Supabase Edge Function
# This script ensures backend files are accessible to the function

set -e

echo "🚀 Preparing Supabase Edge Function deployment..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FUNCTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Backend dir: $BACKEND_DIR"
echo "📁 Function dir: $FUNCTION_DIR"

# Check if backend directory exists
if [ ! -d "$BACKEND_DIR" ]; then
  echo "❌ Error: Backend directory not found at $BACKEND_DIR"
  exit 1
fi

# Copy backend files to function directory (Supabase bundles everything in function dir)
# We'll copy only necessary files to keep bundle size manageable
echo "📦 Copying backend files to function directory..."

# Create backend directory in function
mkdir -p "$FUNCTION_DIR/backend"

# Copy essential backend files
echo "  - Copying simple-server.js..."
cp "$BACKEND_DIR/simple-server.js" "$FUNCTION_DIR/backend/"

# Copy src directory (all routes and controllers)
echo "  - Copying src directory..."
cp -r "$BACKEND_DIR/src" "$FUNCTION_DIR/backend/"

# Copy utils directory
echo "  - Copying utils directory..."
cp -r "$BACKEND_DIR/utils" "$FUNCTION_DIR/backend/" 2>/dev/null || true

# Copy data directory (drug dictionary, etc.)
echo "  - Copying data directory..."
cp -r "$BACKEND_DIR/data" "$FUNCTION_DIR/backend/" 2>/dev/null || true

# Copy Prisma schema (needed for Prisma Client)
echo "  - Copying Prisma schema..."
mkdir -p "$FUNCTION_DIR/backend/prisma"
cp "$BACKEND_DIR/prisma/schema.prisma" "$FUNCTION_DIR/backend/prisma/" 2>/dev/null || true

# Copy package.json for dependency reference
echo "  - Copying package.json..."
cp "$BACKEND_DIR/package.json" "$FUNCTION_DIR/backend/" 2>/dev/null || true

echo "✅ Backend files copied"

# Deploy the function
echo "📤 Deploying function to Supabase..."
supabase functions deploy backend-express

echo "✅ Deployment complete!"
echo ""
echo "🔗 Your Express backend is now available at:"
echo "   https://your-project-ref.supabase.co/functions/v1/backend-express"

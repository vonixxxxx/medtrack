#!/bin/bash
# Quick deploy with all environment variables pre-set

export DATABASE_URL="postgresql://postgres:tirpuV-sihsu7-rijjem@db.ydfksxcktsjhadiotlrc.supabase.co:5432/postgres"
export JWT_SECRET="8a1ac4d831720f929941ac89de22dea979bbe7c5c4dee9a06ffc17e07d80a400"
export SUPABASE_URL="https://ydfksxcktsjhadiotlrc.supabase.co"

# Write DATABASE_URL to .env.local
echo "DATABASE_URL=\"$DATABASE_URL\"" > api/.env.local

echo "✅ Environment variables set"
echo "🚀 Starting automated deployment..."

# Run automated deployment
./DEPLOY_AUTO.sh

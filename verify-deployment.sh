#!/bin/bash
# Verify deployment is ready - check for conflicting files

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     VERIFYING DEPLOYMENT READINESS                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if api/src-temp exists
if [ -d "api/src-temp" ]; then
    echo "❌ ERROR: api/src-temp/ still exists!"
    exit 1
else
    echo "✅ api/src-temp/ is removed"
fi

# Check .vercelignore
if grep -q "api/src-temp/" .vercelignore; then
    echo "✅ api/src-temp/ is in .vercelignore"
else
    echo "❌ ERROR: api/src-temp/ not in .vercelignore!"
    exit 1
fi

# Check .gitignore
if grep -q "api/src-temp/" .gitignore; then
    echo "✅ api/src-temp/ is in .gitignore"
else
    echo "❌ ERROR: api/src-temp/ not in .gitignore!"
    exit 1
fi

# Check for conflicting files in api/ (should be none)
CONFLICTS=$(find api/ -type f \( -name "*.js" -o -name "*.ts" \) 2>/dev/null | sed 's/\.js$//;s/\.ts$//' | sort | uniq -d)
if [ -z "$CONFLICTS" ]; then
    echo "✅ No conflicting .js/.ts files in api/"
else
    echo "❌ WARNING: Found potential conflicts:"
    echo "$CONFLICTS"
fi

# Check latest commit
LATEST=$(git log --oneline -1 | cut -d' ' -f1)
echo "✅ Latest commit: $LATEST"

echo ""
echo "✅ All checks passed! Ready for deployment."
echo ""
echo "📋 Next: Go to Vercel and trigger deployment with latest commit:"
echo "   https://vercel.com/vonixs-projects/medtrack"

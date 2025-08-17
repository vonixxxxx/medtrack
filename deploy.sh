#!/bin/bash

echo "🚀 MedTrack Deployment Script"
echo "=============================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not found. Please run 'git init' first."
    exit 1
fi

# Check if we have commits
if ! git rev-parse HEAD >/dev/null 2>&1; then
    echo "❌ No commits found. Please make at least one commit first."
    exit 1
fi

echo "✅ Git repository ready"
echo ""

# Check if we have a remote origin
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "⚠️  No remote origin found."
    echo "Please add your GitHub repository as origin:"
    echo "git remote add origin <your-github-repo-url>"
    echo ""
    read -p "Would you like to add a remote origin now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your GitHub repository URL: " repo_url
        git remote add origin "$repo_url"
        echo "✅ Remote origin added"
    else
        echo "❌ Deployment requires a remote repository. Exiting."
        exit 1
    fi
fi

echo "✅ Remote repository configured"
echo ""

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ Code pushed to GitHub successfully"
else
    echo "❌ Failed to push to GitHub. Please check your credentials."
    exit 1
fi

echo ""
echo "🎉 Ready for Render.com deployment!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://render.com and sign up/login"
echo "2. Click 'New +' → 'Web Service' for backend"
echo "3. Connect your GitHub repository"
echo "4. Configure backend service (see README.md)"
echo "5. Deploy backend and get the URL"
echo "6. Create frontend static site with backend URL"
echo "7. Update environment variables"
echo ""
echo "📖 See README.md for detailed deployment instructions"
echo ""
echo "🔗 Your repository: $(git remote get-url origin)"
echo "🌐 Render.com: https://render.com"

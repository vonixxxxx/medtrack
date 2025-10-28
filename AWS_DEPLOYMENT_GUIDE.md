# AWS Deployment Guide for MedTrack

## Prerequisites
1. AWS account with appropriate permissions
2. AWS CLI installed: `pip install awscli`
3. Elastic Beanstalk CLI: `pip install awsebcli`
4. Docker installed (for local testing)

---

## Step 1: Setup AWS

### Install AWS and EB CLI
\`\`\`bash
# Install AWS CLI
pip install awscli --upgrade

# Install Elastic Beanstalk CLI
pip install awsebcli

# Verify installations
aws --version
eb --version
\`\`\`

### Configure AWS Credentials
\`\`\`bash
# Configure AWS credentials
aws configure

# Enter your values:
# AWS Access Key ID: [YOUR_KEY]
# AWS Secret Access Key: [YOUR_SECRET]
# Default region: us-east-1 (or your preferred region)
# Default output format: json
\`\`\`

**What this does:** Sets up command-line access to AWS.

---

## Step 2: Deploy Backend to Elastic Beanstalk

### Navigate to project directory
\`\`\`bash
cd /path/to/medtrack
\`\`\`

### Initialize Elastic Beanstalk
\`\`\`bash
# Initialize EB application
eb init -p "Docker running on 64bit Amazon Linux 2" medtrack-app --region us-east-1

# Select environment type: Single instance (free tier) or Load balanced (production)
# Answer prompts:
# - Application name: medtrack-app
# - Platform: Docker
# - Platform version: Latest
# - Public IP: Yes
\`\`\`

### Create Elastic Beanstalk environment
\`\`\`bash
# Create and deploy environment
eb create medtrack-prod \
  --instance-types t3.small \
  --single \
  --envvars DATABASE_URL=postgresql://user:pass@host:5432/dbname
\`\`\`

**What this does:** Creates a Docker container that runs your backend API.

### Set environment variables
\`\`\`bash
# Set JWT secret (IMPORTANT: Use a secure random string)
eb setenv JWT_SECRET="your-super-secure-secret-key-here" NODE_ENV=production

# Set database URL (will be updated in Step 3)
eb setenv DATABASE_URL="postgresql://medtrack:password@your-rds-endpoint:5432/medtrack"
\`\`\`

### Deploy the application
\`\`\`bash
eb deploy
\`\`\`

### Get your backend URL
\`\`\`bash
# Get the URL of your backend
eb status

# Save this URL - you'll need it for frontend:
# Example: http://medtrack-app.us-east-1.elasticbeanstalk.com
\`\`\`

---

## Step 3: Deploy Database (RDS PostgreSQL)

### Create RDS PostgreSQL database
\`\`\`bash
# Create RDS instance (takes 5-10 minutes)
aws rds create-db-instance \
  --db-instance-identifier medtrack-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username medtrack \
  --master-user-password YOUR_SECURE_PASSWORD \
  --allocated-storage 20 \
  --storage-type gp2 \
  --publicly-accessible \
  --region us-east-1
\`\`\`

### Wait for database to be available
\`\`\`bash
# Check database status
aws rds describe-db-instances --db-instance-identifier medtrack-db --query 'DBInstances[0].DBInstanceStatus'

# When status is "available", get the endpoint
aws rds describe-db-instances --db-instance-identifier medtrack-db --query 'DBInstances[0].Endpoint.Address'
\`\`\`

### Create database and user
\`\`\`bash
# Connect to your RDS instance (replace ENDPOINT)
psql -h YOUR_RDS_ENDPOINT.rds.amazonaws.com -U medtrack -d postgres

# Run these SQL commands:
CREATE DATABASE medtrack;
\\c medtrack
GRANT ALL PRIVILEGES ON DATABASE medtrack TO medtrack;

# Exit psql: \\q
\`\`\`

### Update backend environment variable
\`\`\`bash
# Update DATABASE_URL with actual RDS endpoint
eb setenv DATABASE_URL="postgresql://medtrack:YOUR_SECURE_PASSWORD@YOUR_RDS_ENDPOINT.rds.amazonaws.com:5432/medtrack"
\`\`\`

### Run database migrations
\`\`\`bash
# SSH into Elastic Beanstalk instance
eb ssh

# Once inside, run migrations
cd /var/app/current
npx prisma migrate deploy

# Exit SSH
exit
\`\`\`

**What this does:** Creates your PostgreSQL database and runs migrations to create tables.

---

## Step 4: Deploy Frontend (S3 + CloudFront)

### Build the React frontend
\`\`\`bash
cd frontend

# Install dependencies
npm install

# Create .env.production file
echo "VITE_API_URL=https://YOUR_EB_URL.onrender.com/api" > .env.production

# Build for production
npm run build

# The build output is in dist/ directory
\`\`\`

### Create S3 bucket
\`\`\`bash
# Create S3 bucket for static hosting
aws s3 mb s3://medtrack-frontend --region us-east-1

# Enable static website hosting
aws s3 website s3://medtrack-frontend \
  --index-document index.html \
  --error-document index.html
\`\`\`

### Upload frontend to S3
\`\`\`bash
# Upload build files
aws s3 sync frontend/dist/ s3://medtrack-frontend --delete

# Set bucket policy for public read access
cat > bucket-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::medtrack-frontend/*"
    }
  ]
}
EOF

aws s3api put-bucket-policy --bucket medtrack-frontend --policy file://bucket-policy.json
\`\`\`

### Create CloudFront distribution
\`\`\`bash
# Create CloudFront distribution with HTTPS
aws cloudfront create-distribution \
  --origin-domain-name medtrack-frontend.s3-website-us-east-1.amazonaws.com \
  --default-root-object index.html \
  --viewer-certificate AcmCertificateArn=arn:aws:acm:us-east-1:ACCOUNT_ID:certificate/CERT_ID
\`\`\`

**What this does:** Hosts your React app with CDN and HTTPS.

---

## Step 5: Verify & Secure

### Update CORS settings
\`\`\`bash
# Update backend CORS to allow frontend
eb setenv CORS_ORIGIN="https://YOUR_CLOUDFRONT_URL.cloudfront.net"
\`\`\`

### Update frontend API URL
\`\`\`bash
cd frontend
rm .env.production
echo "VITE_API_URL=https://YOUR_EB_URL.elasticbeanstalk.com/api" > .env.production

# Rebuild and redeploy
npm run build
aws s3 sync dist/ s3://medtrack-frontend --delete
\`\`\`

### Test the deployment
\`\`\`bash
# Test backend
curl https://YOUR_EB_URL.elasticbeanstalk.com/api/test-public

# Test frontend
curl https://YOUR_CLOUDFRONT_URL.cloudfront.net
\`\`\`

### Monitor logs
\`\`\`bash
# View backend logs
eb logs

# Stream real-time logs
eb logs --stream
\`\`\`

---

## Environment Variables Template

### Backend (.env for EB)
\`\`\`bash
NODE_ENV=production
PORT=8080
DATABASE_URL=postgresql://medtrack:password@YOUR_RDS_ENDPOINT:5432/medtrack
JWT_SECRET=your-super-secure-secret-key-at-least-256-bits
JWT_EXPIRES_IN=7d
CORS_ORIGIN=https://YOUR_CLOUDFRONT_URL.cloudfront.net
\`\`\`

### Frontend (.env.production)
\`\`\`bash
VITE_API_URL=https://YOUR_EB_URL.elasticbeanstalk.com/api
\`\`\`

---

## Quick Commands Reference

\`\`\`bash
# Deploy backend
eb deploy

# View backend logs
eb logs

# SSH into backend
eb ssh

# Update environment variables
eb setenv KEY=value

# Check status
eb status

# View application health
eb health

# Update database URL after RDS is ready
eb setenv DATABASE_URL="postgresql://medtrack:password@YOUR_RDS_ENDPOINT:5432/medtrack"

# Redeploy frontend
aws s3 sync frontend/dist/ s3://medtrack-frontend --delete
\`\`\`

---

## Migration to ECS + Fargate (Future)

When you're ready to scale:

1. **ECS Task Definition**: Convert Dockerfile to ECS task
2. **Fargate Service**: Run containers without managing servers
3. **Application Load Balancer**: Replace EB load balancer
4. **Auto Scaling**: Configure target tracking policies
5. **RDS Proxy**: For connection pooling

I'll help you migrate when you're ready!

---

## Troubleshooting

### Backend not starting
\`\`\`bash
eb logs
# Look for errors in logs
eb ssh
# Check if database is reachable
\`\`\`

### Database connection errors
\`\`\`bash
# Verify RDS is accessible
aws rds describe-db-instances --db-instance-identifier medtrack-db

# Check security groups allow EB to connect to RDS
\`\`\`

### Frontend can't reach backend
- Check CORS_ORIGIN environment variable
- Verify frontend VITE_API_URL is correct
- Check Elastic Beanstalk URL is accessible

---

## Estimated Costs (USD/month)

- **Elastic Beanstalk**: $0 (single instance free tier)
- **RDS PostgreSQL**: $15-30 (db.t3.micro)
- **S3**: $0.50 (storage + requests)
- **CloudFront**: $1-5 (data transfer)
- **Total**: ~$20-40/month for production use

**Production recommendation:** Use load-balanced multi-AZ setup (~$100-200/month)

---

## You're Done! 🎉

Your app should now be live at:
- Frontend: https://YOUR_CLOUDFRONT_URL.cloudfront.net
- Backend: https://YOUR_EB_URL.elasticbeanstalk.com
- Database: PostgreSQL on RDS

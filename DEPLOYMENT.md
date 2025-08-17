# MedTrack - Production Deployment Guide

## 🚀 Overview

This guide covers deploying the MedTrack application to production using Docker, PostgreSQL, and cloud platforms like Railway, Render, or DigitalOcean.

## 📋 Prerequisites

- Docker & Docker Compose installed
- PostgreSQL database (local or hosted)
- Email service (Gmail, SendGrid, etc.) for password reset
- Domain name (optional, for production)

## 🏗️ Production Setup

### 1. Environment Configuration

Copy the environment template:
```bash
cp backend/env.template backend/.env
```

Configure your production environment variables:

```bash
# Database (Required)
DATABASE_URL="postgresql://username:password@host:5432/medtrack?schema=public"

# Authentication (Required)
JWT_SECRET="your-super-secure-256-bit-secret-key"
JWT_EXPIRES_IN="7d"

# Email Service (Required for password reset)
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USER="your-email@gmail.com"
SMTP_PASS="your-app-password"
SMTP_FROM="MedTrack <noreply@yourdomain.com>"

# App Configuration
PORT=4000
NODE_ENV="production"
FRONTEND_URL="https://your-frontend-domain.com"

# Security
CORS_ORIGIN="https://your-frontend-domain.com"

# 2FA Configuration
TOTP_ISSUER="MedTrack"
TOTP_WINDOW=2

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
AUTH_RATE_LIMIT_MAX=5
```

### 2. Database Setup

#### Option A: Using Docker Compose (Local/VPS)

```bash
# Start PostgreSQL database
docker-compose up -d postgres

# Run database migrations
cd backend
npx prisma migrate deploy
npx prisma db seed
```

#### Option B: Hosted Database (Railway, Supabase, etc.)

1. Create a PostgreSQL database on your preferred platform
2. Copy the connection string to `DATABASE_URL`
3. Run migrations:
```bash
cd backend
npx prisma migrate deploy
```

### 3. Docker Deployment

#### Full Stack with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f backend
```

#### Backend Only (if frontend is deployed separately)

```bash
# Build backend image
cd backend
docker build -t medtrack-backend .

# Run with environment file
docker run -d \
  --name medtrack-api \
  --env-file .env \
  -p 4000:4000 \
  medtrack-backend
```

## ☁️ Cloud Platform Deployment

### Railway

1. **Connect Repository**: Link your GitHub repo to Railway
2. **Add PostgreSQL**: Railway → New → Database → PostgreSQL
3. **Configure Variables**: Add all environment variables from `.env`
4. **Deploy**: Railway will auto-deploy on git push

**Railway Configuration:**
```bash
# Build Command
npm run build

# Start Command  
npm start

# Root Directory
backend
```

### Render

1. **Create Web Service**: Connect GitHub repo
2. **Add PostgreSQL**: Render → New → PostgreSQL
3. **Environment Variables**: Add from `.env` template
4. **Build & Deploy**: Automatic on git push

**Render Configuration:**
```bash
# Build Command
npm install && npx prisma generate && npx prisma migrate deploy

# Start Command
npm start

# Root Directory
backend
```

### DigitalOcean App Platform

1. **Create App**: Import from GitHub
2. **Add Database**: Managed PostgreSQL database
3. **Configure Environment**: Add variables via dashboard
4. **Deploy**: Automatic deployment

## 🔧 Frontend Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from frontend directory
cd frontend
vercel --prod
```

### Netlify

1. **Connect Repository**: Link GitHub repo
2. **Build Settings**:
   - Build command: `npm run build`
   - Publish directory: `dist`
   - Base directory: `frontend`

### Configuration

Update frontend environment:
```bash
# frontend/.env.production
VITE_API_URL=https://your-backend-domain.com/api
```

## 🔒 Security Checklist

- [ ] Strong JWT secret (256+ bits)
- [ ] Database credentials secured
- [ ] HTTPS enabled (SSL certificates)
- [ ] CORS configured for production domain
- [ ] Rate limiting enabled
- [ ] Email service configured
- [ ] Backup strategy in place
- [ ] Monitoring setup

## 📊 Monitoring & Health Checks

### Health Check Endpoint

```bash
# Check application health
curl https://your-api-domain.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "environment": "production",
  "version": "1.0.0"
}
```

### Database Monitoring

```bash
# Check database connection
curl -H "Authorization: Bearer YOUR_JWT" \
  https://your-api-domain.com/api/cycles
```

## 🛠️ Maintenance

### Database Backups

```bash
# Manual backup
pg_dump $DATABASE_URL > backup-$(date +%Y%m%d).sql

# Restore backup
psql $DATABASE_URL < backup-20240101.sql
```

### Log Monitoring

```bash
# Docker logs
docker-compose logs -f backend

# Application logs (if using PM2)
pm2 logs medtrack-backend
```

### Updates

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose up -d --build backend
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Test database connection
   npx prisma migrate status
   ```

2. **CORS Errors**
   ```bash
   # Verify CORS_ORIGIN matches frontend domain
   echo $CORS_ORIGIN
   ```

3. **Email Not Sending**
   ```bash
   # Test SMTP configuration
   npm run test:email
   ```

4. **2FA QR Code Not Loading**
   ```bash
   # Check if QRCode package is installed
   npm list qrcode
   ```

### Support

- Check logs first: `docker-compose logs backend`
- Verify environment variables are set
- Test endpoints with curl/Postman
- Check database connectivity

## 📝 Production Checklist

- [ ] Database migrations applied
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] DNS records configured
- [ ] Email service tested
- [ ] Backup strategy implemented
- [ ] Monitoring setup
- [ ] Performance testing completed
- [ ] Security audit done

---

## 🔗 Quick Deploy Commands

```bash
# 1. Clone and setup
git clone https://github.com/your-username/medtrack
cd medtrack

# 2. Configure environment
cp backend/env.template backend/.env
# Edit .env with your values

# 3. Deploy with Docker
docker-compose up -d

# 4. Run migrations
docker-compose exec backend npx prisma migrate deploy

# 5. Check health
curl http://localhost:4000/health
```

Your MedTrack application is now ready for production! 🎉

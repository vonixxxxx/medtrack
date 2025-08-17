# MedTrack - Hospital Medication Tracking System

<div align="center">

![MedTrack Logo](https://via.placeholder.com/200x100/3b82f6/ffffff?text=MedTrack)

**A secure, production-ready medication tracking application for hospitals and patients**

[![Node.js](https://img.shields.io/badge/Node.js-18.20.2-green.svg)](https://nodejs.org/)
[![React](https://img.shields.io/badge/React-18.0-blue.svg)](https://reactjs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com/)
[![Security](https://img.shields.io/badge/Security-2FA%20%2B%20JWT-green.svg)](https://en.wikipedia.org/wiki/Multi-factor_authentication)

</div>

## 🏥 Overview

MedTrack is a comprehensive medication tracking system designed for hospitals to monitor patient medication cycles, health metrics, and provide secure data export for research purposes. The application features robust authentication, real-time notifications, and production-ready security measures.

## ✨ Features

### 🔐 Authentication & Security
- **JWT-based authentication** with secure password hashing (bcrypt)
- **Two-Factor Authentication (2FA)** with QR code setup
- **Password reset** via email with secure token validation
- **Rate limiting** and security headers (Helmet.js)
- **Input validation** with Joi schemas

### 💊 Medication Management
- **Medication cycles** with start/end dates and dosage tracking
- **Dose logging** with automatic reminder generation
- **Frequency management** (daily, weekly, custom intervals)
- **Intake tracking** with real-time updates

### 📊 Health Metrics
- **Comprehensive metrics**: Weight, Height, BMI, Blood Pressure
- **Optional metrics**: Hip Circumference, Waist measurements
- **Visual analytics** with interactive charts (Recharts)
- **Historical tracking** with trend analysis

### 🔔 Notifications & Reminders
- **Automated reminders** via cron jobs
- **Email notifications** with HTML templates
- **Real-time dashboard** updates
- **Customizable notification preferences**

### 📈 Data & Analytics
- **Interactive graphs** with zoom and pan functionality
- **Data export** (CSV, PNG, PDF) for research
- **Anonymized exports** for hospital research compliance
- **Real-time metric tracking** and visualization

### 🎨 Modern UI/UX
- **Responsive design** with TailwindCSS
- **Smooth animations** and transitions
- **Color-coded indicators** for medication status
- **Intuitive navigation** with settings management

## 🏗️ Architecture

### Backend Stack
- **Node.js** + **Express.js** - REST API server
- **PostgreSQL** - Production database
- **Prisma ORM** - Type-safe database operations
- **JWT** - Stateless authentication
- **Nodemailer** - Email service integration
- **node-cron** - Scheduled reminder system

### Frontend Stack
- **React 18** - Modern UI framework
- **Vite** - Fast build tool and dev server
- **TailwindCSS** - Utility-first styling
- **React Query** - Data fetching and caching
- **React Router** - Client-side routing
- **Recharts** - Data visualization

### DevOps & Deployment
- **Docker** + **Docker Compose** - Containerization
- **PostgreSQL** - Production database
- **Helmet.js** - Security headers
- **Rate limiting** - DDoS protection
- **Health checks** - Monitoring and alerting

## 🚀 Quick Start

### Prerequisites

- Node.js 18.20.2+
- PostgreSQL 15+
- Docker & Docker Compose (optional)

### 1. Clone Repository

```bash
git clone https://github.com/your-username/medtrack.git
cd medtrack
```

### 2. Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Configure environment
cp env.template .env
# Edit .env with your database and email settings

# Setup database
npx prisma migrate dev
npx prisma db seed

# Start development server
npm run dev
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:4000
- **Health Check**: http://localhost:4000/health

### 5. Test Credentials

```
Email: testuser@example.com
Password: password123
```

## 🐳 Docker Deployment

### Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend
```

### Production

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Check health
curl http://localhost:4000/health
```

## 📋 API Documentation

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/signup` | User registration |
| POST | `/api/auth/login` | User login |
| POST | `/api/auth/forgot-password` | Request password reset |
| POST | `/api/auth/reset-password` | Reset password with token |
| GET | `/api/auth/me` | Get current user info |
| POST | `/api/auth/change-password` | Change password |

### 2FA Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/2fa/generate` | Generate 2FA QR code |
| POST | `/api/auth/2fa/verify` | Verify and enable 2FA |
| POST | `/api/auth/2fa/disable` | Disable 2FA |

### Medication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cycles` | Get medication cycles |
| POST | `/api/cycles` | Create medication cycle |
| GET | `/api/cycles/today` | Get today's reminders |
| GET | `/api/cycles/upcoming` | Get upcoming doses |
| POST | `/api/cycles/:id/dose` | Mark dose as taken |

### Metrics Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cycles/:id/metrics` | Get cycle metrics |
| POST | `/api/cycles/:id/metrics` | Add metric log |
| PUT | `/api/cycles/:id/metrics/:metricId` | Update metric |
| DELETE | `/api/cycles/:id/metrics/:metricId` | Delete metric |

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/medtrack"

# Authentication
JWT_SECRET="your-secret-key"
JWT_EXPIRES_IN="7d"

# Email
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USER="your-email@gmail.com"
SMTP_PASS="your-app-password"

# Security
CORS_ORIGIN="http://localhost:3000"
RATE_LIMIT_MAX_REQUESTS=100
AUTH_RATE_LIMIT_MAX=5
```

## 🛡️ Security Features

### Authentication
- **bcrypt** password hashing with salt rounds
- **JWT** tokens with configurable expiration
- **2FA** with TOTP (Time-based One-Time Password)
- **Password reset** with secure token validation

### Security Headers
- **Helmet.js** for security headers
- **CORS** protection with configurable origins
- **Rate limiting** for API endpoints
- **Input validation** with Joi schemas

### Data Protection
- **Password encryption** in database
- **Secure token generation** for password reset
- **SQL injection protection** via Prisma ORM
- **XSS protection** via security headers

## 📊 Features Deep Dive

### Dashboard Components

1. **Upcoming Measurements Card**
   - Color-coded priority indicators
   - One-click metric logging
   - RAG (Red-Amber-Green) status

2. **Medication Reminders Card**
   - Today's medication list
   - "Taken" button functionality
   - Real-time updates

3. **Upcoming Intake Card**
   - Next medication schedule
   - Time-based calculations
   - Color-coded medications

4. **Metric History Table**
   - Sortable and filterable data
   - Cycle-specific filtering
   - Interactive line charts

5. **Cycle Detail Card**
   - Medication progress tracking
   - Days on medication counter
   - Intake completion statistics

6. **Add Medication Cycle**
   - Comprehensive form validation
   - Frequency configuration
   - End date scheduling

7. **Add Metric Card**
   - Multiple metric types
   - Automatic date setting
   - Cycle association

### Advanced Analytics

- **Interactive Charts**: Zoom, pan, hover tooltips
- **Trend Analysis**: Long-term health metric tracking
- **Export Functionality**: CSV, PNG, PDF formats
- **Research Data**: Anonymized exports for hospitals

## 🔄 Development Workflow

### Backend Development

```bash
# Run with hot reload
npm run dev

# Database operations
npx prisma migrate dev    # Create migration
npx prisma db push       # Push schema changes
npx prisma studio        # Database GUI

# Seeding
npm run seed
```

### Frontend Development

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Testing

```bash
# Backend tests
cd backend
npm test

# Frontend tests
cd frontend
npm test
```

## 🌐 Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for comprehensive deployment instructions including:

- Docker deployment
- Cloud platform setup (Railway, Render, Vercel)
- Database migration
- Environment configuration
- SSL certificate setup
- Monitoring and health checks

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines

- Follow ESLint configuration
- Add tests for new features
- Update documentation
- Use semantic commit messages

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check database status
   npx prisma migrate status
   ```

2. **Email Not Sending**
   - Verify SMTP credentials
   - Check email service configuration
   - Test with curl/Postman

3. **2FA Setup Issues**
   - Ensure QRCode package is installed
   - Check TOTP configuration
   - Verify authenticator app compatibility

### Getting Help

- 📖 Check [Documentation](./docs/)
- 🐛 Report [Issues](https://github.com/your-username/medtrack/issues)
- 💬 Join [Discussions](https://github.com/your-username/medtrack/discussions)
- 📧 Email: support@medtrack.com

## 🎯 Roadmap

- [ ] Mobile app (React Native)
- [ ] Real-time notifications (WebSocket)
- [ ] Advanced analytics dashboard
- [ ] Integration with hospital systems (HL7 FHIR)
- [ ] Multi-language support
- [ ] Voice reminders
- [ ] Wearable device integration

## ⭐ Acknowledgments

- **Healthcare Community** for requirements and feedback
- **Open Source Contributors** for amazing packages
- **Security Researchers** for vulnerability reports
- **Hospital Partners** for testing and validation

---

<div align="center">

**Built with ❤️ for healthcare professionals and patients**

[Website](https://medtrack.com) • [Documentation](./docs/) • [Support](mailto:support@medtrack.com)

</div>

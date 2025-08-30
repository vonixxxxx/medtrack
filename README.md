# 💊 MedTrack - Comprehensive Health Tracking Application

A complete full-stack healthcare application built with **Node.js/Express** backend and **React** frontend. MedTrack provides comprehensive health assessments, medication tracking, and clinical screening tools for healthcare professionals and patients.

## 🚀 Features

- **Hospital-Grade Medication System**: Advanced medication validation and tracking
- **Clinical Screening Tools**: AUDIT, IPAQ, IIEF-5, and specialized health assessments
- **Demographics Management**: Comprehensive patient information collection
- **Health Metrics Calculation**: Automated BMI, WHR, WHtR, BRI calculations
- **Medication Library**: BNF and EMC integration for medication lookup
- **Dashboard Analytics**: Real-time health metrics and progress tracking
- **NHS Integration**: UK NHS medicine service integration
- **Secure Authentication**: JWT-based user authentication
- **Database Management**: Prisma ORM with SQLite/PostgreSQL support

## 🏗️ Architecture

```
medtrack/
├── backend/                     # Node.js/Express backend
│   ├── src/
│   │   ├── controllers/         # API controllers
│   │   ├── services/           # Business logic services
│   │   ├── routes/             # API routes
│   │   ├── middleware/         # Authentication & security
│   │   └── utils/              # Helper utilities
│   ├── prisma/                 # Database schema & migrations
│   ├── tests/                  # Unit & integration tests
│   └── docs/                   # API documentation
├── frontend/                   # React frontend application
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── pages/              # Application pages
│   │   └── api.js              # API client
│   └── public/                 # Static assets
└── docs/                       # Project documentation
```

## 🛠️ Technology Stack

### Backend
- **Node.js & Express**: RESTful API server
- **Prisma ORM**: Database management with SQLite/PostgreSQL
- **JWT Authentication**: Secure user authentication
- **Joi Validation**: Input validation and sanitization
- **Jest & Playwright**: Testing framework
- **Docker**: Containerization support

### Frontend
- **React 18**: Modern React with hooks
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Smooth animations
- **Vite**: Fast build tool and dev server
- **Responsive Design**: Mobile-first approach

### External Integrations
- **NHS Medicines Service**: UK medication database
- **BNF (British National Formulary)**: Medication information
- **EMC (Electronic Medicines Compendium)**: Prescribing information
- **OpenFDA**: FDA drug database integration

## 📋 Prerequisites

- **Node.js 16+** and npm
- **Python 3.8+** (for some utility scripts)
- **Git** for cloning the repository
- **Docker** (optional, for containerized deployment)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/vonixxxxx/medtrack.git
cd medtrack
```

### 2. Setup Backend

```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Setup environment variables
cp env.template .env
# Edit .env with your configuration

# Setup database
npx prisma migrate dev
npx prisma db seed

# Start the backend server
npm start
```

**Backend will be running at**: http://localhost:3001

### 3. Setup Frontend

```bash
# Open a new terminal and navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

**Frontend will be running at**: http://localhost:5173

### 4. Test the Application

1. Open http://localhost:5173 in your browser
2. Register a new account or login
3. Complete the comprehensive health assessment
4. Explore the dashboard features
5. Test medication search and validation

## 🔍 API Endpoints

### Authentication
- **POST** `/api/auth/login` - User login
- **POST** `/api/auth/register` - User registration
- **POST** `/api/auth/logout` - User logout

### Registration & Demographics
- **POST** `/api/registration/complete` - Complete registration
- **GET** `/api/registration/status` - Check registration status
- **PUT** `/api/registration/update` - Update registration data

### Medications
- **GET** `/api/medications/search` - Search medications
- **POST** `/api/medications/validate` - Validate medication
- **GET** `/api/medications/library` - Medication library

### Screening
- **POST** `/api/screening/audit` - AUDIT questionnaire
- **POST** `/api/screening/ipaq` - IPAQ assessment
- **POST** `/api/screening/iief5` - IIEF-5 questionnaire

### Health Metrics
- **GET** `/api/metrics/dashboard` - Dashboard metrics
- **POST** `/api/metrics/calculate` - Calculate health indices

## 📱 Key Features

### Health Assessments
- **AUDIT**: Alcohol Use Disorders Identification Test
- **IPAQ**: International Physical Activity Questionnaire
- **IIEF-5**: Erectile Dysfunction Assessment (Male)
- **Testosterone Screening**: Low T symptom assessment
- **Heart Risk Calculator**: Cardiovascular risk assessment

### Medication Management
- **Validation System**: Hospital-grade medication validation
- **Drug Interactions**: Comprehensive interaction checking
- **Dosage Monitoring**: Automated dosage validation
- **NHS Integration**: UK medicines database integration

### Dashboard Analytics
- **Health Metrics**: BMI, WHR, WHtR, BRI calculations
- **Risk Assessment**: Color-coded risk categories
- **Progress Tracking**: Historical data visualization
- **Export Functionality**: CSV export for healthcare providers

## 🧪 Testing

### Run Backend Tests

```bash
cd backend

# Unit tests
npm test

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e

# Hospital-grade system tests
npm run test:hospital-grade
```

### Test Coverage

```bash
# Generate test coverage report
npm run test:coverage
```

## 🐳 Docker Deployment

### Development Environment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment

```bash
# Build and deploy
./deploy.sh

# Or use render.yaml for Render.com deployment
```

## 🔧 Configuration

### Environment Variables

Create `.env` file in the backend directory:

```env
# Database
DATABASE_URL="file:./dev.db"

# Authentication
JWT_SECRET="your-jwt-secret-key"
JWT_EXPIRATION="24h"

# External APIs
NHS_API_KEY="your-nhs-api-key"
OPENFDA_API_KEY="your-openfda-key"

# Application
NODE_ENV="development"
PORT=3001
FRONTEND_URL="http://localhost:5173"
```

### Database Configuration

```bash
# Reset database
npx prisma migrate reset

# Deploy migrations
npx prisma migrate deploy

# Generate Prisma client
npx prisma generate

# View database
npx prisma studio
```

## 📚 Documentation

- **API Documentation**: Available at http://localhost:3001/docs
- **Features Guide**: See [MEDTRACK_FEATURES.md](./MEDTRACK_FEATURES.md)
- **Deployment Guide**: See [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Hospital System**: See [backend/HOSPITAL_GRADE_SUMMARY.md](./backend/HOSPITAL_GRADE_SUMMARY.md)
- **NHS Integration**: See [backend/docs/NHS_INTEGRATION.md](./backend/docs/NHS_INTEGRATION.md)

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in configuration files
2. **Database issues**: Run `npx prisma migrate reset`
3. **Authentication errors**: Check JWT configuration
4. **API errors**: Verify external API keys and internet connection

### Database Issues

```bash
# Clear database and start fresh
rm backend/prisma/dev.db
npx prisma migrate dev
npx prisma db seed
```

### Dependencies Issues

```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Input Validation**: Comprehensive input sanitization
- **SQL Injection Protection**: Prisma ORM protection
- **CORS Configuration**: Controlled cross-origin requests
- **Rate Limiting**: API rate limiting protection
- **Data Encryption**: Sensitive data encryption

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow ESLint configuration
- Write tests for new features
- Update documentation
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support and questions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review the documentation files
3. Open an issue on GitHub
4. Contact the development team

## 🔄 Version History

- **v2.0.0**: Hospital-grade medication system
- **v1.5.0**: NHS integration and advanced screening
- **v1.0.0**: Initial release with basic health tracking

---

**Made with ❤️ for better healthcare management**

This application provides comprehensive health tracking and medication management tools designed for healthcare professionals and patients, with enterprise-grade security and reliability.
# 🏥 MedTrack - Medication Tracking App

A full-stack Node.js + React application for tracking medications, metrics, and health data.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Git

### Local Development
```bash
# Clone the repository
git clone <your-repo-url>
cd medtrack

# Install backend dependencies
cd backend
npm install

# Install frontend dependencies
cd ../frontend
npm install

# Start backend (in backend directory)
npm run dev

# Start frontend (in frontend directory, new terminal)
npm run dev
```

## 🌐 Deployment to Render.com

### Step 1: Deploy Backend

1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `medtrack-backend`
   - **Environment**: `Node`
   - **Build Command**: `npm install`
   - **Start Command**: `npm start`
   - **Port**: `8000`

5. Add Environment Variables:
   ```
   PORT=8000
   FRONTEND_URL=https://your-frontend-url.onrender.com
   JWT_SECRET=your-secret-key-here
   ```

6. Click "Create Web Service"

### Step 2: Deploy Frontend

1. In Render, click "New +" → "Static Site"
2. Configure the site:
   - **Name**: `medtrack-frontend`
   - **Build Command**: `npm run build`
   - **Publish Directory**: `dist`

3. Add Environment Variables:
   ```
   VITE_API_URL=https://your-backend-url.onrender.com/api
   ```

4. Click "Create Static Site"

### Step 3: Update Configuration

After both services are deployed, update the frontend environment variable with your actual backend URL.

## 📁 Project Structure

```
medtrack/
├── backend/                 # Node.js API server
│   ├── src/
│   │   ├── controllers/    # API logic
│   │   ├── middleware/     # Auth & security
│   │   ├── routes/         # API endpoints
│   │   └── index.js        # Server entry point
│   ├── prisma/             # Database schema & migrations
│   └── package.json
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Application pages
│   │   └── api.js          # API configuration
│   └── package.json
└── README.md
```

## 🔧 Features

- **User Authentication**: JWT-based login/signup
- **Medication Management**: Track medications and cycles
- **Health Metrics**: Log and visualize health data
- **Reminders**: Automated medication reminders
- **Responsive Design**: Mobile-first UI with Tailwind CSS
- **Real-time Updates**: Live data synchronization

## 🛠️ Tech Stack

### Backend
- **Node.js** + **Express.js**
- **Prisma** ORM with **SQLite**
- **JWT** authentication
- **bcrypt** password hashing
- **CORS** enabled

### Frontend
- **React 18** with **Vite**
- **Tailwind CSS** for styling
- **React Query** for data fetching
- **React Router** for navigation
- **Recharts** for data visualization

## 🔒 Security Features

- Rate limiting on API endpoints
- Helmet.js security headers
- Input validation with Joi
- Secure password hashing
- CORS configuration

## 📊 API Endpoints

- `POST /api/auth/login` - User login
- `POST /api/auth/signup` - User registration
- `GET /api/cycles` - Get medication cycles
- `GET /api/metrics/logs` - Get health metrics
- `POST /api/medications` - Create medication
- `GET /api/reminders` - Get reminders

## 🚨 Environment Variables

### Backend (.env)
```bash
PORT=8000
JWT_SECRET=your-secret-key
FRONTEND_URL=http://localhost:3000
```

### Frontend (.env)
```bash
VITE_API_URL=http://localhost:8000/api
```

## 📱 Usage

1. **Sign Up**: Create a new account
2. **Add Medications**: Create medication cycles with dosages
3. **Log Metrics**: Track health measurements
4. **View Dashboard**: Monitor medications and metrics
5. **Set Reminders**: Configure medication reminders

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions, please open a GitHub issue or contact the development team.

---

**MedTrack** - Your health, tracked simply. 🏥✨

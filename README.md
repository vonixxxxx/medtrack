# 💊 Drug Information Search System

A complete full-stack application that provides drug information by searching the OpenFDA database. Built with **FastAPI** backend and **React** frontend.

## 🚀 Features

- **FastAPI Backend**: Modern, fast Python web framework with OpenFDA integration
- **React Frontend**: Clean, responsive UI with real-time search
- **OpenFDA Integration**: Access to comprehensive FDA drug database
- **Real-time Search**: Instant results with loading states and error handling
- **Professional UI**: Beautiful gradient design with smooth animations
- **Mobile Responsive**: Works perfectly on all devices

## 🏗️ Architecture

```
drug-info-system/
├── backend_fastapi/          # FastAPI backend server
│   ├── main.py              # Main FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile          # Docker containerization
│   └── README.md           # Backend documentation
├── frontend_fastapi/         # React frontend application
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── App.css         # Application styles
│   │   └── main.jsx        # React entry point
│   ├── package.json        # Node.js dependencies
│   ├── vite.config.js      # Vite configuration
│   └── README.md           # Frontend documentation
└── README.md               # This file
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **HTTPX**: Async HTTP client for OpenFDA API calls
- **Uvicorn**: ASGI server for running FastAPI
- **Python 3.8+**: Modern Python with type hints

### Frontend
- **React 18**: Modern React with hooks
- **Vite**: Fast build tool and dev server
- **CSS3**: Modern styling with animations
- **Fetch API**: Native browser HTTP client

## 📋 Prerequisites

- **Python 3.8+** and pip
- **Node.js 16+** and npm
- **Git** for cloning the repository

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd drug-info-system

# Or if you're creating from scratch, the files are already created
```

### 2. Start the Backend

```bash
# Navigate to backend directory
cd backend_fastapi

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

**Backend will be running at**: http://localhost:8000

### 3. Start the Frontend

```bash
# Open a new terminal and navigate to frontend directory
cd frontend_fastapi

# Install dependencies
npm install

# Start the development server
npm run dev
```

**Frontend will be running at**: http://localhost:3000

### 4. Test the Application

1. Open http://localhost:3000 in your browser
2. Enter a drug name (e.g., "aspirin", "ibuprofen")
3. Click Search to see drug information
4. View comprehensive details including warnings and usage

## 🔍 API Endpoints

### Backend API (FastAPI)

- **GET** `/` - Welcome message
- **GET** `/drug/{name}` - Search for drug information
- **GET** `/health` - Health check

### API Documentation

Once the backend is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📱 Frontend Features

- **Search Interface**: Clean search form with validation
- **Loading States**: Animated spinner during API calls
- **Error Handling**: User-friendly error messages
- **Responsive Design**: Works on desktop and mobile
- **Modern UI**: Gradient backgrounds and smooth animations

## 🧪 Testing Examples

### Test the Backend API

```bash
# Health check
curl http://localhost:8000/health

# Search for aspirin
curl http://localhost:8000/drug/aspirin

# Search for ibuprofen
curl http://localhost:8000/drug/ibuprofen
```

### Test the Frontend

1. Open http://localhost:3000
2. Try searching for common drugs:
   - `aspirin`
   - `ibuprofen`
   - `acetaminophen`
   - `metformin`

## 🐳 Docker Support

### Backend with Docker

```bash
cd backend_fastapi

# Build the image
docker build -t drug-api .

# Run the container
docker run -p 8000:8000 drug-api
```

## 🔧 Development

### Backend Development

```bash
cd backend_fastapi

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd frontend_fastapi

# Start dev server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📚 Learning Resources

This project demonstrates:

- **FastAPI**: Modern Python web framework
- **React Hooks**: State management with useState
- **Async Operations**: Handling API calls with async/await
- **Error Handling**: Graceful error handling in both backend and frontend
- **Modern CSS**: Flexbox, Grid, and animations
- **API Integration**: External API integration with proper error handling
- **Responsive Design**: Mobile-first approach

## 🚨 Troubleshooting

### Common Issues

1. **Backend not starting**: Check if port 8000 is available
2. **Frontend not starting**: Check if port 3000 is available
3. **CORS errors**: Ensure backend is running and CORS is configured
4. **API errors**: Check OpenFDA API status and internet connection

### Port Conflicts

If ports are busy, you can change them:

**Backend**: Modify `main.py` or use uvicorn with different port
**Frontend**: Modify `vite.config.js` port setting

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review the individual README files in each directory
3. Check the API documentation at http://localhost:8000/docs
4. Ensure both backend and frontend are running

---

**Happy coding! 🎉**

This project provides a complete example of building a modern full-stack application with FastAPI and React, perfect for learning and extending.

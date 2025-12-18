import { Routes, Route, Navigate } from 'react-router-dom';
import './utils/clearStorage';
import LandingPage from './pages/LandingPage';
import FeaturesPage from './pages/FeaturesPage';
import AboutPage from './pages/AboutPage';
import ContactPage from './pages/ContactPage';
import LoginPage from './pages/LoginPage';
import SignUpPage from './pages/SignUpPage';
import EnhancedPatientSignup from './pages/EnhancedPatientSignup';
import RegistrationPage from './pages/RegistrationPage';
import Dashboard from './pages/Dashboard';
import DoctorDashboard from './pages/DoctorDashboard';
import SettingsPage from './pages/SettingsPage';
import MedHistoryPage from './pages/MedHistoryPage';
import ForgotPasswordPage from './pages/ForgotPasswordPage';
import ResetPasswordPage from './pages/ResetPasswordPage';
import AddMedication from './pages/AddMedication';
import AddMetric from './pages/AddMetric';

const PrivateRoute = ({ children, requiredRole }) => {
  const token = localStorage.getItem('token');
  const userString = localStorage.getItem('user');
  const user = userString && userString !== 'undefined' ? JSON.parse(userString) : {};
  
  if (!token) {
    // Show message instead of redirecting
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Please log in to access this page.</p>
          <p className="text-sm text-gray-500">Authentication is required.</p>
        </div>
      </div>
    );
  }
  
  // If no required role specified, allow access
  if (!requiredRole) {
    return children;
  }
  
  // If user doesn't have a role, show message
  if (!user.role) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Invalid user session.</p>
          <p className="text-sm text-gray-500">Please log in again.</p>
        </div>
      </div>
    );
  }
  
  // If user has wrong role, redirect to appropriate dashboard within the app
  if (user.role !== requiredRole) {
    if (user.role === 'clinician') {
      return <Navigate to="/dashboard/clinician" replace />;
    } else if (user.role === 'patient') {
      return <Navigate to="/dashboard/patient" replace />;
    } else {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="text-center">
            <p className="text-gray-600 mb-4">Invalid user role.</p>
            <p className="text-sm text-gray-500">Please contact support.</p>
          </div>
        </div>
      );
    }
  }
  
  return children;
};

const RoleBasedRoute = ({ children, allowedRoles }) => {
  const userString = localStorage.getItem('user');
  const user = userString && userString !== 'undefined' ? JSON.parse(userString) : {};
  
  if (!allowedRoles.includes(user.role)) {
    // Redirect to appropriate dashboard based on user role
    if (user.role === 'clinician') {
      return <Navigate to="/dashboard/clinician" />;
    } else {
      return <Navigate to="/dashboard/patient" />;
    }
  }
  
  return children;
};

export default function App() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/features" element={<FeaturesPage />} />
      <Route path="/about" element={<AboutPage />} />
      <Route path="/contact" element={<ContactPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignUpPage />} />
      <Route path="/signup/enhanced" element={<EnhancedPatientSignup />} />
      <Route path="/register" element={<RegistrationPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      
      {/* Protected routes */}
      <Route
        path="/dashboard"
        element={
          <PrivateRoute>
            <Navigate to="/dashboard/patient" />
          </PrivateRoute>
        }
      />
      <Route
        path="/dashboard/patient"
        element={
          <PrivateRoute requiredRole="patient">
            <Dashboard />
          </PrivateRoute>
        }
      />
      <Route
        path="/dashboard/clinician"
        element={
          <PrivateRoute requiredRole="clinician">
            <DoctorDashboard />
          </PrivateRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <PrivateRoute>
            <SettingsPage />
          </PrivateRoute>
        }
      />
      <Route
        path="/med-history"
        element={
          <PrivateRoute requiredRole="patient">
            <MedHistoryPage />
          </PrivateRoute>
        }
      />
      <Route
        path="/add-medication"
        element={
          <PrivateRoute requiredRole="patient">
            <AddMedication />
          </PrivateRoute>
        }
      />
      <Route
        path="/add-metric"
        element={
          <PrivateRoute requiredRole="patient">
            <AddMetric />
          </PrivateRoute>
        }
      />
    </Routes>
  );
}

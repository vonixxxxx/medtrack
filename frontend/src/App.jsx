import { Routes, Route, Navigate } from 'react-router-dom';
import './utils/clearStorage';
import LoginPage from './pages/LoginPage';
import SignUpPage from './pages/SignUpPage';
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
    return <Navigate to="/login" />;
  }
  
  // If no required role specified, allow access
  if (!requiredRole) {
    return children;
  }
  
  // If user doesn't have a role, redirect to login
  if (!user.role) {
    return <Navigate to="/login" />;
  }
  
  // If user has wrong role, redirect to appropriate dashboard
  if (user.role !== requiredRole) {
    if (user.role === 'clinician') {
      return <Navigate to="/dashboard/clinician" />;
    } else if (user.role === 'patient') {
      return <Navigate to="/dashboard/patient" />;
    } else {
      return <Navigate to="/login" />;
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
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignUpPage />} />
      <Route path="/register" element={<RegistrationPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      
      {/* Protected routes */}
      <Route
        path="/"
        element={
          <PrivateRoute>
            <Navigate to="/dashboard" />
          </PrivateRoute>
        }
      />
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

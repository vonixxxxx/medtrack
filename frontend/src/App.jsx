import { Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import SignUpPage from './pages/SignUpPage';
import Dashboard from './pages/Dashboard';
import SettingsPage from './pages/SettingsPage';
import MedHistoryPage from './pages/MedHistoryPage';
import ForgotPasswordPage from './pages/ForgotPasswordPage';
import ResetPasswordPage from './pages/ResetPasswordPage';
import AddMedication from './pages/AddMedication';
import AddMetric from './pages/AddMetric';

const PrivateRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  return token ? children : <Navigate to="/login" />;
};

export default function App() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignUpPage />} />
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
            <Dashboard />
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
          <PrivateRoute>
            <MedHistoryPage />
          </PrivateRoute>
        }
      />
      <Route
        path="/add-medication"
        element={
          <PrivateRoute>
            <AddMedication />
          </PrivateRoute>
        }
      />
      <Route
        path="/add-metric"
        element={
          <PrivateRoute>
            <AddMetric />
          </PrivateRoute>
        }
      />
    </Routes>
  );
}

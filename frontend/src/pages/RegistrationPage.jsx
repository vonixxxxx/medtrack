import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import RegistrationForm from '../components/RegistrationForm';
import MedTrackDashboard from '../components/MedTrackDashboard';

const RegistrationPage = () => {
  const navigate = useNavigate();
  const [isRegistrationComplete, setIsRegistrationComplete] = useState(false);
  const [userData, setUserData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user has completed registration
    checkRegistrationStatus();
  }, []);

  const checkRegistrationStatus = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        navigate('/login');
        return;
      }

      const response = await fetch('/api/registration/status', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        if (data.isRegistrationComplete) {
          setIsRegistrationComplete(true);
          setUserData(data.user);
        }
      }
    } catch (error) {
      console.error('Error checking registration status:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegistrationComplete = (userData) => {
    setIsRegistrationComplete(true);
    setUserData(userData);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (isRegistrationComplete && userData) {
    return <MedTrackDashboard userData={userData} />;
  }

  return <RegistrationForm onComplete={handleRegistrationComplete} />;
};

export default RegistrationPage;

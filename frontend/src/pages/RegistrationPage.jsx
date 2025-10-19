import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import RegistrationForm from '../components/RegistrationForm';
import Dashboard from './Dashboard';

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

      // Check if user profile exists
      const userProfile = localStorage.getItem('user');
      if (userProfile) {
        const parsedProfile = JSON.parse(userProfile);
        setIsRegistrationComplete(true);
        setUserData(parsedProfile);
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
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  if (isRegistrationComplete && userData) {
    return <Dashboard userData={userData} />;
  }

  return <RegistrationForm onComplete={handleRegistrationComplete} />;
};

export default RegistrationPage;





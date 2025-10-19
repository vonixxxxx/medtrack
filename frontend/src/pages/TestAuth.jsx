import { useState, useEffect } from 'react';
import api from '../api';

const TestAuth = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          setLoading(false);
          return;
        }

        const { data } = await api.get('auth/me');
        setUser(data);
      } catch (error) {
        console.error('Auth check failed:', error);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  const handleSignup = async () => {
    try {
      const { data } = await api.post('auth/signup', {
        email: 'test-patient-debug@example.com',
        password: 'password123',
        role: 'patient',
        hospitalCode: '123456789'
      });
      
      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
      console.log('Stored user data:', data.user);
      setUser(data.user);
    } catch (error) {
      console.error('Signup failed:', error);
    }
  };

  const handleLogin = async () => {
    try {
      const { data } = await api.post('auth/login', {
        email: 'test-patient-debug@example.com',
        password: 'password123'
      });
      
      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
      console.log('Stored user data:', data.user);
      setUser(data.user);
    } catch (error) {
      console.error('Login failed:', error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
  };

  const handleClearStorage = () => {
    localStorage.clear();
    setUser(null);
    console.log('Cleared all localStorage data');
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Auth Test Page</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <h2>Current User:</h2>
        <pre>{JSON.stringify(user, null, 2)}</pre>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h2>LocalStorage:</h2>
        <p>Token: {localStorage.getItem('token') ? 'Present' : 'Missing'}</p>
        <p>User: {localStorage.getItem('user') || 'Missing'}</p>
      </div>

      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <button onClick={handleSignup}>Signup as Patient</button>
        <button onClick={handleLogin}>Login</button>
        <button onClick={handleLogout}>Logout</button>
        <button onClick={handleClearStorage}>Clear Storage</button>
        <button onClick={() => window.location.href = '/dashboard/patient'}>
          Go to Patient Dashboard
        </button>
      </div>
    </div>
  );
};

export default TestAuth;

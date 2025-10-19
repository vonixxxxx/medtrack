import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    
    try {
      const { data } = await api.post('auth/login', { email, password });
      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
      console.log('Stored user data:', data.user);
      
      // Redirect based on role
      if (data.user.role === 'clinician') {
        navigate('/dashboard/clinician');
      } else {
        navigate('/dashboard/patient');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Invalid credentials');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo/Brand */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-full bg-white flex items-center justify-center mx-auto mb-4">
            <span className="text-2xl font-bold text-black">M</span>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">
            MedTrack
          </h1>
          <p className="text-gray-400 text-sm">Your health, tracked simply</p>
        </div>

        {/* Login Form */}
        <div className="bg-gray-900 rounded-3xl shadow-2xl border border-gray-800 p-8">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-white mb-2">Welcome Back</h2>
            <p className="text-gray-400 text-sm">Sign in to your account</p>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded-xl text-red-400 text-sm text-center">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-white mb-2">
                Email Address
              </label>
              <input
                id="email"
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white transition-all duration-200"
                required
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-white mb-2">
                Password
              </label>
              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white transition-all duration-200"
                required
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-white text-black py-3 px-4 rounded-xl font-medium hover:bg-gray-200 focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-gray-900 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed mt-6"
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-black mr-2"></div>
                  Signing In...
                </div>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          <div className="mt-6 text-center space-y-3">
            <p>
              <a 
                href="/forgot-password" 
                className="text-sm text-white hover:text-gray-300 font-medium transition-colors"
              >
                Forgot your password?
              </a>
            </p>
            <p className="text-sm text-gray-400">
              Need an account?{' '}
              <a 
                href="/signup" 
                className="text-white hover:text-gray-300 font-medium transition-colors"
              >
                Sign up
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

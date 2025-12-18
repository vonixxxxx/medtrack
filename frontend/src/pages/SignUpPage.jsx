import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';

export default function SignUpPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('patient');
  const [hospitalCode, setHospitalCode] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    
    // Validate hospital code - check for empty or invalid code
    if (!hospitalCode || hospitalCode.trim() === '') {
      setError('Hospital code is required');
      setIsLoading(false);
      return;
    }

    if (hospitalCode.trim() !== '123456789') {
      setError('Invalid hospital code');
      setIsLoading(false);
      return;
    }
    
    try {
      const signupData = { email, password, role, hospitalCode: hospitalCode.trim() };
      
      const { data } = await api.post('auth/signup', signupData);
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
      setError(err.response?.data?.error || 'Signup failed');
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

        {/* Signup Form */}
        <div className="bg-gray-900 rounded-3xl shadow-2xl border border-gray-800 p-8">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-white mb-2">Create Account</h2>
            <p className="text-gray-400 text-sm">Join MedTrack to start tracking your health</p>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded-xl text-red-400 text-sm text-center">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Role Selection */}
            <div>
              <label className="block text-sm font-medium text-white mb-3">
                I am a:
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setRole('patient')}
                  className={`px-4 py-3 rounded-xl border transition-all duration-200 ${
                    role === 'patient'
                      ? 'bg-white text-black border-white'
                      : 'bg-gray-800 text-white border-gray-700 hover:border-gray-600'
                  }`}
                >
                  Patient
                </button>
                <button
                  type="button"
                  onClick={() => setRole('clinician')}
                  className={`px-4 py-3 rounded-xl border transition-all duration-200 ${
                    role === 'clinician'
                      ? 'bg-white text-black border-white'
                      : 'bg-gray-800 text-white border-gray-700 hover:border-gray-600'
                  }`}
                >
                  Clinician
                </button>
              </div>
            </div>

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

            {/* Hospital Code - Required for both patients and clinicians */}
            <div>
              <label htmlFor="hospitalCode" className="block text-sm font-medium text-white mb-2">
                Hospital Code
              </label>
              <input
                id="hospitalCode"
                type="text"
                placeholder="Enter your hospital code"
                value={hospitalCode}
                onChange={(e) => setHospitalCode(e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white transition-all duration-200"
                required
              />
              <p className="text-xs text-gray-400 mt-1">
                Required for all accounts. Contact your institution if you don't have a code.
              </p>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-white mb-2">
                Password
              </label>
              <input
                id="password"
                type="password"
                placeholder="Create a password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white transition-all duration-200"
                required
              />
              <p className="text-xs text-gray-400 mt-1">Password should be at least 6 characters</p>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-white text-black py-3 px-4 rounded-xl font-medium hover:bg-gray-200 focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-gray-900 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed mt-6"
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-black mr-2"></div>
                  Creating Account...
                </div>
              ) : (
                'Create Account'
              )}
            </button>
          </form>

          <div className="mt-4 text-center">
            <p className="text-sm text-gray-400">
              For comprehensive medical data collection, use our{' '}
              <button
                onClick={() => navigate('/signup/enhanced')}
                className="text-blue-400 hover:underline font-medium"
              >
                Enhanced Patient Registration
              </button>
            </p>
          </div>

          <div className="mt-6 text-center">
            <p className="text-sm text-gray-400">
              Already have an account?{' '}
              <a 
                href="/login" 
                className="text-white hover:text-gray-300 font-medium transition-colors"
              >
                Log in
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

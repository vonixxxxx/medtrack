import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import MobileNavigation from '../components/MobileNavigation';
import { fetcher, poster } from '../api';

export default function SettingsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  
  // Form states
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  const [emailForm, setEmailForm] = useState({ email: '' });
  const [twoFAToken, setTwoFAToken] = useState('');
  const [showQR, setShowQR] = useState(false);
  const [qrData, setQrData] = useState(null);

  // Get user info
  const { data: user, isLoading, error } = useQuery({
    queryKey: ['user'],
    queryFn: () => fetcher('/auth/me'),
    retry: 1, // Only retry once
    onError: (error) => {
      console.error('Failed to load user:', error);
      if (error.response?.status === 401) {
        navigate('/login');
      }
    }
  });

  // Change password mutation
  const changePasswordMutation = useMutation({
    mutationFn: (data) => poster('/auth/change-password', data),
    onSuccess: () => {
      alert('Password changed successfully!');
      setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
    },
    onError: (error) => {
      alert(error.response?.data?.error || 'Password change failed');
    }
  });

  // Generate 2FA mutation
  const generate2FAMutation = useMutation({
    mutationFn: () => poster('/auth/2fa/generate'),
    onSuccess: (data) => {
      setQrData(data);
      setShowQR(true);
    },
    onError: (error) => {
      alert(error.response?.data?.error || '2FA setup failed');
    }
  });

  // Verify 2FA mutation
  const verify2FAMutation = useMutation({
    mutationFn: (token) => poster('/auth/2fa/verify', { token }),
    onSuccess: () => {
      alert('2FA enabled successfully!');
      setShowQR(false);
      setTwoFAToken('');
      queryClient.invalidateQueries(['user']);
    },
    onError: (error) => {
      alert(error.response?.data?.error || '2FA verification failed');
    }
  });

  // Disable 2FA mutation
  const disable2FAMutation = useMutation({
    mutationFn: (token) => poster('/auth/2fa/disable', { token }),
    onSuccess: () => {
      alert('2FA disabled successfully!');
      setTwoFAToken('');
      queryClient.invalidateQueries(['user']);
    },
    onError: (error) => {
      alert(error.response?.data?.error || '2FA disable failed');
    }
  });

  const handlePasswordChange = (e) => {
    e.preventDefault();
    
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      alert('New passwords do not match');
      return;
    }
    
    if (passwordForm.newPassword.length < 8) {
      alert('New password must be at least 8 characters');
      return;
    }
    
    changePasswordMutation.mutate({
      currentPassword: passwordForm.currentPassword,
      newPassword: passwordForm.newPassword
    });
  };

  const handleEnable2FA = () => {
    generate2FAMutation.mutate();
  };

  const handleVerify2FA = (e) => {
    e.preventDefault();
    if (twoFAToken.length === 6) {
      verify2FAMutation.mutate(twoFAToken);
    } else {
      alert('Please enter a 6-digit code');
    }
  };

  const handleDisable2FA = (e) => {
    e.preventDefault();
    if (twoFAToken.length === 6) {
      disable2FAMutation.mutate(twoFAToken);
    } else {
      alert('Please enter a 6-digit code to disable 2FA');
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading user settings...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Unable to load settings</h3>
          <p className="text-gray-600 mb-4">Please try refreshing the page or logging in again.</p>
          <button
            onClick={() => navigate('/login')}
            className="bg-blue-600 text-white px-4 py-2 rounded-xl hover:bg-blue-700 transition-colors"
          >
            Go to Login
          </button>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">No user data found. Please log in again.</p>
          <button
            onClick={() => navigate('/login')}
            className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-xl hover:bg-blue-700 transition-colors"
          >
            Go to Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <MobileNavigation />
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {/* Header */}
        <motion.div 
          className="mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900">Account Settings</h1>
          <p className="text-gray-600 mt-2 text-sm sm:text-base">Manage your account preferences and security settings</p>
        </motion.div>

        <motion.div 
          className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.3 }}
        >
          {/* Change Password */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Change Password</h2>
            <form onSubmit={handlePasswordChange} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Current Password
                </label>
                <input
                  type="password"
                  value={passwordForm.currentPassword}
                  onChange={(e) => setPasswordForm(prev => ({ ...prev, currentPassword: e.target.value }))}
                  className="w-full rounded-xl border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  New Password
                </label>
                <input
                  type="password"
                  value={passwordForm.newPassword}
                  onChange={(e) => setPasswordForm(prev => ({ ...prev, newPassword: e.target.value }))}
                  className="w-full rounded-xl border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                  minLength="8"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confirm New Password
                </label>
                <input
                  type="password"
                  value={passwordForm.confirmPassword}
                  onChange={(e) => setPasswordForm(prev => ({ ...prev, confirmPassword: e.target.value }))}
                  className="w-full rounded-xl border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                  minLength="8"
                />
              </div>
              <button
                type="submit"
                disabled={changePasswordMutation.isLoading}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-xl hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                {changePasswordMutation.isLoading ? 'Changing...' : 'Change Password'}
              </button>
            </form>
          </div>

          {/* Two-Factor Authentication */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Two-Factor Authentication</h2>
            
            {!user.is2FAEnabled ? (
              <div className="space-y-4">
                <p className="text-gray-600 text-sm">
                  Add an extra layer of security to your account by enabling two-factor authentication.
                </p>
                
                {!showQR ? (
                  <button
                    onClick={handleEnable2FA}
                    disabled={generate2FAMutation.isLoading}
                    className="w-full bg-blue-600 text-white py-2 px-4 rounded-xl hover:bg-blue-700 disabled:opacity-50 transition-colors"
                  >
                    {generate2FAMutation.isLoading ? 'Setting up...' : 'Enable 2FA'}
                  </button>
                ) : (
                  <div className="space-y-4">
                    <div className="text-center">
                      <p className="text-sm text-gray-600 mb-4">
                        Scan this QR code with your authenticator app:
                      </p>
                      {qrData && (
                        <img 
                          src={qrData.qrCode} 
                          alt="2FA QR Code" 
                          className="mx-auto mb-4 border rounded-lg"
                        />
                      )}
                      <p className="text-xs text-gray-500 mb-4">
                        Manual entry key: <code className="bg-gray-100 px-2 py-1 rounded">{qrData?.manualEntryKey}</code>
                      </p>
                    </div>
                    
                    <form onSubmit={handleVerify2FA} className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Enter 6-digit code from your app:
                        </label>
                        <input
                          type="text"
                          value={twoFAToken}
                          onChange={(e) => setTwoFAToken(e.target.value.replace(/\D/g, '').slice(0, 6))}
                          className="w-full rounded-xl border border-gray-300 px-3 py-2 text-center text-lg tracking-widest focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder="000000"
                          maxLength="6"
                          required
                        />
                      </div>
                      <button
                        type="submit"
                        disabled={verify2FAMutation.isLoading || twoFAToken.length !== 6}
                        className="w-full bg-blue-600 text-white py-2 px-4 rounded-xl hover:bg-blue-700 disabled:opacity-50 transition-colors"
                      >
                        {verify2FAMutation.isLoading ? 'Verifying...' : 'Verify and Enable'}
                      </button>
                    </form>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="text-green-700 font-medium">2FA is enabled</span>
                </div>
                <p className="text-gray-600 text-sm">
                  Your account is protected with two-factor authentication.
                </p>
                
                <form onSubmit={handleDisable2FA} className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Enter 6-digit code to disable 2FA:
                    </label>
                    <input
                      type="text"
                      value={twoFAToken}
                      onChange={(e) => setTwoFAToken(e.target.value.replace(/\D/g, '').slice(0, 6))}
                      className="w-full rounded-xl border border-gray-300 px-3 py-2 text-center text-lg tracking-widest focus:outline-none focus:ring-2 focus:ring-red-500"
                      placeholder="000000"
                      maxLength="6"
                      required
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={disable2FAMutation.isLoading || twoFAToken.length !== 6}
                    className="w-full bg-red-600 text-white py-2 px-4 rounded-xl hover:bg-red-700 disabled:opacity-50 transition-colors"
                  >
                    {disable2FAMutation.isLoading ? 'Disabling...' : 'Disable 2FA'}
                  </button>
                </form>
              </div>
            )}
          </div>

          {/* Account Information */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Account Information</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <div className="flex space-x-2">
                  <input
                    type="email"
                    value={emailForm.email || user.email}
                    onChange={(e) => setEmailForm({ email: e.target.value })}
                    className="flex-1 rounded-xl border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button className="px-4 py-2 bg-gray-100 text-gray-700 rounded-xl hover:bg-gray-200 transition-colors">
                    Update
                  </button>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Account Created</label>
                <p className="text-gray-600">
                  {user.createdAt ? new Date(user.createdAt).toLocaleDateString() : 'Unknown'}
                </p>
              </div>
            </div>
          </div>

          {/* Danger Zone */}
          <div className="bg-white rounded-xl shadow-sm p-6 border-l-4 border-red-500">
            <h2 className="text-xl font-semibold text-red-900 mb-4">Danger Zone</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-red-800 mb-2">Delete Account</h3>
                <p className="text-red-600 text-sm mb-3">
                  Once you delete your account, there is no going back. All your data will be permanently removed.
                </p>
                <button className="bg-red-600 text-white px-4 py-2 rounded-xl hover:bg-red-700 transition-colors">
                  Delete Account
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

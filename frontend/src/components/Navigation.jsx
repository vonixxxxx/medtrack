import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Menu, X, User, Settings, LogOut, Bell, Search } from 'lucide-react';

export default function Navigation() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const navigate = useNavigate();

  const handleLogout = () => {
    // Clear any stored tokens/user data
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
  const toggleProfile = () => setIsProfileOpen(!isProfileOpen);

  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <Link to="/demographics" className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
            <span className="text-white font-bold text-xl">M</span>
          </div>
          <span className="text-2xl font-bold text-gray-900">MedTrack</span>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-8">
          <Link 
            to="/demographics" 
            className="text-black hover:text-blue-400 transition-colors font-medium"
          >
            Health Profile
          </Link>
          <Link 
            to="/dashboard" 
            className="text-black hover:text-blue-400 transition-colors font-medium"
          >
            Dashboard
          </Link>
          <Link 
            to="/medications" 
            className="text-black hover:text-blue-400 transition-colors font-medium"
          >
            Medications
          </Link>
        </div>

        {/* Right Side */}
        <div className="flex items-center space-x-4">
          {/* Notifications */}
          <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors relative">
            <Bell className="w-6 h-6" />
            <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>

          {/* Profile Dropdown */}
          <div className="relative">
            <button
              onClick={toggleProfile}
              className="flex items-center space-x-2 p-2 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center">
                <User className="w-4 h-4 text-white" />
              </div>
              <span className="text-gray-700 font-medium hidden sm:block">User</span>
            </button>

            {isProfileOpen && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-lg border border-gray-200 py-2 z-50"
              >
                <Link
                  to="/settings"
                  className="flex items-center space-x-3 px-4 py-2 text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  <Settings className="w-4 h-4" />
                  <span>Settings</span>
                </Link>
                <button
                  onClick={handleLogout}
                  className="flex items-center space-x-3 px-4 py-2 text-gray-700 hover:bg-gray-50 transition-colors w-full text-left"
                >
                  <LogOut className="w-4 h-4" />
                  <span>Logout</span>
                </button>
              </motion.div>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={toggleMenu}
            className="md:hidden p-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="md:hidden mt-4 pb-4 border-t border-gray-200"
        >
          <div className="flex flex-col space-y-2 pt-4">
            <Link
              to="/demographics"
              onClick={() => setIsMenuOpen(false)}
              className="px-4 py-2 text-black hover:text-blue-400 hover:bg-blue-50 rounded-lg transition-colors"
            >
              Health Profile
            </Link>
            <Link
              to="/dashboard"
              onClick={() => setIsMenuOpen(false)}
              className="px-4 py-2 text-black hover:text-blue-400 hover:bg-blue-50 rounded-lg transition-colors"
            >
              Dashboard
            </Link>
            <Link
              to="/medications"
              onClick={() => setIsMenuOpen(false)}
              className="px-4 py-2 text-black hover:text-blue-400 hover:bg-blue-50 rounded-lg transition-colors"
            >
              Medications
            </Link>
          </div>
        </motion.div>
      )}
    </nav>
  );
}

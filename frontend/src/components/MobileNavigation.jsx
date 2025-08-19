import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

export default function MobileNavigation() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  // Close menu on route change
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location.pathname]);

  // Close menu on outside click
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (isMenuOpen && !event.target.closest('.mobile-nav-menu') && !event.target.closest('.hamburger-button')) {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isMenuOpen]);

  // Prevent body scroll when menu is open
  useEffect(() => {
    if (isMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isMenuOpen]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  const navigationItems = [
    {
      label: 'Dashboard',
      path: '/dashboard',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    },
    {
      label: 'Med History',
      path: '/med-history',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2V7zm0 0a2 2 0 012-2h6l2 2h6a2 2 0 012 2v2H3V7z" />
        </svg>
      )
    },
    {
      label: 'Add Medication',
      path: '/add-medication',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
        </svg>
      )
    },
    {
      label: 'Add Metric',
      path: '/add-metric',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    },
    {
      label: 'Settings',
      path: '/settings',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      )
    }
  ];

  const isCurrentPath = (path) => location.pathname === path;

  return (
    <>
      {/* Top Navigation Bar */}
      <nav className="bg-white shadow-lg border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <motion.div 
              className="flex items-center"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
            >
              <h1 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                MedTrack
              </h1>
            </motion.div>

            {/* Desktop Navigation - Hidden on mobile */}
            <div className="hidden lg:flex items-center space-x-1">
              {navigationItems.map((item) => (
                <motion.button
                  key={item.path}
                  onClick={() => navigate(item.path)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-xl transition-all duration-200 ${
                    isCurrentPath(item.path)
                      ? 'bg-blue-100 text-blue-700 shadow-md'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {item.icon}
                  <span className="font-medium">{item.label}</span>
                </motion.button>
              ))}
              
              <motion.button
                onClick={handleLogout}
                className="ml-4 px-4 py-2 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-xl font-medium hover:from-red-600 hover:to-red-700 transition-all duration-200 shadow-md"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Logout
              </motion.button>
            </div>

            {/* Mobile Hamburger Button */}
            <motion.button
              className="lg:hidden hamburger-button p-2 rounded-xl text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              whileTap={{ scale: 0.95 }}
              aria-label="Toggle navigation menu"
              aria-expanded={isMenuOpen}
            >
              <div className="w-6 h-6 flex flex-col justify-center items-center">
                <motion.span
                  className="w-6 h-0.5 bg-current rounded-full"
                  animate={isMenuOpen ? { rotate: 45, y: 3 } : { rotate: 0, y: 0 }}
                  transition={{ duration: 0.2 }}
                />
                <motion.span
                  className="w-6 h-0.5 bg-current rounded-full mt-1"
                  animate={isMenuOpen ? { opacity: 0 } : { opacity: 1 }}
                  transition={{ duration: 0.2 }}
                />
                <motion.span
                  className="w-6 h-0.5 bg-current rounded-full mt-1"
                  animate={isMenuOpen ? { rotate: -45, y: -3 } : { rotate: 0, y: 0 }}
                  transition={{ duration: 0.2 }}
                />
              </div>
            </motion.button>
          </div>
        </div>
      </nav>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {isMenuOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            />

            {/* Mobile Menu */}
            <motion.div
              className="lg:hidden mobile-nav-menu fixed top-16 right-0 bottom-0 w-80 max-w-full bg-white shadow-2xl z-50 overflow-y-auto"
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            >
              <div className="p-6">
                <div className="space-y-3">
                  {navigationItems.map((item, index) => (
                    <motion.button
                      key={item.path}
                      onClick={() => navigate(item.path)}
                      className={`w-full flex items-center space-x-4 p-4 rounded-2xl transition-all duration-200 text-left ${
                        isCurrentPath(item.path)
                          ? 'bg-blue-100 text-blue-700 shadow-lg border border-blue-200'
                          : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                      }`}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1, duration: 0.3 }}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className={`p-2 rounded-xl ${
                        isCurrentPath(item.path) ? 'bg-blue-200' : 'bg-gray-200'
                      }`}>
                        {item.icon}
                      </div>
                      <span className="font-medium text-lg">{item.label}</span>
                    </motion.button>
                  ))}
                </div>

                <motion.div
                  className="mt-8 pt-6 border-t border-gray-200"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6, duration: 0.3 }}
                >
                  <motion.button
                    onClick={handleLogout}
                    className="w-full flex items-center justify-center space-x-3 p-4 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-2xl font-medium shadow-lg hover:from-red-600 hover:to-red-700 transition-all duration-200"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                    </svg>
                    <span className="text-lg">Logout</span>
                  </motion.button>
                </motion.div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}

import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Home, User, Pill, X, Menu } from 'lucide-react';

export default function MobileNavigation() {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const navigationItems = [
    { path: '/demographics', label: 'Health Profile', icon: Home },
    { path: '/dashboard', label: 'Dashboard', icon: User },
    { path: '/medications', label: 'Medications', icon: Pill },
  ];

  const isActive = (path) => location.pathname === path;

  const toggleMenu = () => setIsOpen(!isOpen);

  return (
    <>
      {/* Floating Menu Button */}
      <button
        onClick={toggleMenu}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-full shadow-lg flex items-center justify-center text-white hover:from-blue-700 hover:to-indigo-700 transition-all duration-200"
      >
        {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      {/* Navigation Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="fixed inset-0 z-40 bg-black bg-opacity-50"
            onClick={toggleMenu}
          >
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="absolute right-0 top-0 h-full w-80 bg-white shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="p-6 border-b border-gray-200">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                    <span className="text-white font-bold text-xl">M</span>
                  </div>
                  <span className="text-xl font-bold text-gray-900">MedTrack</span>
                </div>
              </div>

              {/* Navigation Items */}
              <nav className="p-6">
                <div className="space-y-2">
                  {navigationItems.map((item) => {
                    const Icon = item.icon;
                    return (
                      <Link
                        key={item.path}
                        to={item.path}
                        onClick={toggleMenu}
                        className={`flex items-center space-x-4 p-4 rounded-2xl transition-all duration-200 ${
                          isActive(item.path)
                            ? 'bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 border border-blue-200'
                            : 'text-gray-700 hover:bg-gray-50'
                        }`}
                      >
                        <Icon className={`w-6 h-6 ${
                          isActive(item.path) ? 'text-blue-600' : 'text-gray-500'
                        }`} />
                        <span className="font-medium">{item.label}</span>
                      </Link>
                    );
                  })}
                </div>

                {/* Quick Actions */}
                <div className="mt-8 pt-6 border-t border-gray-200">
                  <h3 className="text-sm font-semibold text-gray-600 mb-4 uppercase tracking-wide">
                    Quick Actions
                  </h3>
                  <div className="space-y-2">
                    <button className="w-full flex items-center space-x-4 p-4 rounded-2xl text-gray-700 hover:bg-gray-50 transition-all duration-200">
                      <div className="w-6 h-6 bg-blue-100 rounded-lg flex items-center justify-center">
                        <span className="text-blue-600 text-sm font-bold">+</span>
                      </div>
                      <span className="font-medium">Add Medication</span>
                    </button>
                    <button className="w-full flex items-center space-x-4 p-4 rounded-2xl text-gray-700 hover:bg-gray-50 transition-all duration-200">
                      <div className="w-6 h-6 bg-blue-100 rounded-lg flex items-center justify-center">
                        <span className="text-blue-600 text-sm font-bold">+</span>
                      </div>
                      <span className="font-medium">Add Metric</span>
                    </button>
                  </div>
                </div>
              </nav>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

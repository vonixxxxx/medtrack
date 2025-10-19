import React from 'react';
import { motion } from 'framer-motion';
import { Home, History, BarChart3, Settings, User } from 'lucide-react';

const MobileMobileNavigation = () => {
  return (
    <motion.div 
      className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 px-4 py-2 z-50"
      initial={{ y: 100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex justify-around items-center">
        <button className="flex flex-col items-center p-2 text-gray-600 hover:text-blue-600 transition-colors">
          <Home className="w-5 h-5" />
          <span className="text-xs mt-1">Home</span>
        </button>
        <button className="flex flex-col items-center p-2 text-gray-600 hover:text-blue-600 transition-colors">
          <History className="w-5 h-5" />
          <span className="text-xs mt-1">History</span>
        </button>
        <button className="flex flex-col items-center p-2 text-gray-600 hover:text-blue-600 transition-colors">
          <BarChart3 className="w-5 h-5" />
          <span className="text-xs mt-1">Analytics</span>
        </button>
        <button className="flex flex-col items-center p-2 text-gray-600 hover:text-blue-600 transition-colors">
          <Settings className="w-5 h-5" />
          <span className="text-xs mt-1">Settings</span>
        </button>
        <button className="flex flex-col items-center p-2 text-gray-600 hover:text-blue-600 transition-colors">
          <User className="w-5 h-5" />
          <span className="text-xs mt-1">Profile</span>
        </button>
      </div>
    </motion.div>
  );
};

export default MobileMobileNavigation;






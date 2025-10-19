import React from 'react';
import { motion } from 'framer-motion';
import { Bell, Settings, User, LogOut } from 'lucide-react';

export const Header = ({ onSettingsClick, onProfileClick, onSignOut }) => {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-card border-b border-border p-4"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">MedTrack</h1>
          <p className="text-sm text-muted-foreground">Your Health Dashboard</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={onSettingsClick}
            className="p-2 rounded-lg hover:bg-accent transition-colors"
            title="Settings"
          >
            <Settings className="w-5 h-5 text-muted-foreground" />
          </button>
          
          <button
            onClick={onProfileClick}
            className="p-2 rounded-lg hover:bg-accent transition-colors"
            title="Profile"
          >
            <User className="w-5 h-5 text-muted-foreground" />
          </button>
          
          <button className="p-2 rounded-lg hover:bg-accent transition-colors" title="Notifications">
            <Bell className="w-5 h-5 text-muted-foreground" />
          </button>
          
          <button
            onClick={onSignOut}
            className="p-2 rounded-lg hover:bg-red-500/20 transition-colors group"
            title="Sign Out"
          >
            <LogOut className="w-5 h-5 text-muted-foreground group-hover:text-red-500" />
          </button>
        </div>
      </div>
    </motion.header>
  );
};


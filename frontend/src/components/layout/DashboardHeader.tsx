"use client";

import { motion } from "framer-motion";
import { Bell, Settings, User, LogOut, Menu, X } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

interface DashboardHeaderProps {
  onSearchClick?: () => void;
  onSettingsClick?: () => void;
  onProfileClick?: () => void;
  onSignOut?: () => void;
  userRole?: "patient" | "clinician";
  userName?: string;
}

export const DashboardHeader = ({
  onSettingsClick,
  onProfileClick,
  onSignOut,
  userRole = "patient",
  userName,
}: DashboardHeaderProps) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  const handleSignOut = () => {
    if (window.confirm("Are you sure you want to sign out?")) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      window.location.href = "http://localhost:3000";
    }
  };

  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100 shadow-sm"
    >
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-20">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="flex items-center gap-3"
          >
            <a
              href="/"
              className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent"
            >
              MedTrack
            </a>
            <span className="hidden sm:inline text-sm text-gray-600 font-medium px-3 py-1 bg-blue-50 border border-blue-200 rounded-full">
              {userRole === "patient" ? "Patient Portal" : "Clinician Portal"}
            </span>
          </motion.div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-3">
            {userName && (
              <div className="px-4 py-2 text-sm text-gray-700 bg-gray-50 rounded-xl border border-gray-200">
                <span className="font-medium">{userName}</span>
              </div>
            )}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onSettingsClick}
              className="p-2.5 rounded-xl text-gray-600 hover:text-blue-600 hover:bg-blue-50 transition-all"
              aria-label="Settings"
            >
              <Settings size={20} />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onProfileClick}
              className="p-2.5 rounded-xl text-gray-600 hover:text-blue-600 hover:bg-blue-50 transition-all"
              aria-label="Profile"
            >
              <User size={20} />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="p-2.5 rounded-xl text-gray-600 hover:text-blue-600 hover:bg-blue-50 transition-all relative"
              aria-label="Notifications"
            >
              <Bell size={20} />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-blue-600 rounded-full"></span>
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSignOut}
              className="ml-2 px-4 py-2.5 text-sm font-medium text-gray-700 hover:text-red-600 hover:bg-red-50 rounded-xl transition-all border border-gray-200 hover:border-red-200"
            >
              Sign Out
            </motion.button>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2 text-neutral-700"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden pb-4 space-y-2 border-t border-gray-100 mt-2 pt-4"
          >
            {userName && (
              <div className="px-4 py-2 text-sm text-gray-700 font-medium bg-gray-50 rounded-xl border border-gray-200">
                {userName}
              </div>
            )}
            <button
              onClick={() => {
                onSettingsClick?.();
                setIsMobileMenuOpen(false);
              }}
              className="w-full text-left px-4 py-2 text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-xl transition-colors flex items-center gap-3"
            >
              <Settings size={18} />
              Settings
            </button>
            <button
              onClick={() => {
                onProfileClick?.();
                setIsMobileMenuOpen(false);
              }}
              className="w-full text-left px-4 py-2 text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-xl transition-colors flex items-center gap-3"
            >
              <User size={18} />
              Profile
            </button>
            <button
              onClick={handleSignOut}
              className="w-full text-left px-4 py-2 text-red-600 hover:bg-red-50 rounded-xl transition-colors flex items-center gap-3 border border-red-200"
            >
              <LogOut size={18} />
              Sign Out
            </button>
          </motion.div>
        )}
      </nav>
    </motion.header>
  );
};


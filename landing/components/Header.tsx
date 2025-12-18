"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";
import AuthModal from "./AuthModal";

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authType, setAuthType] = useState<"login" | "signup">("login");

  const openAuth = (type: "login" | "signup") => {
    setAuthType(type);
    setAuthModalOpen(true);
  };

  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100">
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 lg:h-20">
            {/* Logo */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              className="flex items-center"
            >
              <a href="/" className="text-2xl font-bold bg-gradient-to-r from-primary-600 to-primary-400 bg-clip-text text-transparent">
                MedTrack
              </a>
            </motion.div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-6">
              <a href="/features" className="text-gray-700 hover:text-primary-600 transition-colors font-medium">
                Features
              </a>
              <a href="/about" className="text-gray-700 hover:text-primary-600 transition-colors font-medium">
                About
              </a>
              <a href="/enterprise" className="text-gray-700 hover:text-primary-600 transition-colors font-medium">
                Enterprise
              </a>
              <a href="/contact" className="text-gray-700 hover:text-primary-600 transition-colors font-medium">
                Contact
              </a>
              <button
                onClick={() => openAuth("login")}
                className="text-gray-700 hover:text-primary-600 transition-colors font-medium"
              >
                Login
              </button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                const ctaSection = document.getElementById("cta-section");
                if (ctaSection) {
                  ctaSection.scrollIntoView({ behavior: "smooth" });
                } else {
                  openAuth("signup");
                }
              }}
              className="px-6 py-2.5 bg-primary-600 text-white rounded-full font-medium hover:bg-primary-700 transition-colors shadow-lg shadow-primary-600/25"
            >
              Sign Up
            </motion.button>
            </div>

            {/* Mobile Menu Button */}
            <button
              className="md:hidden p-2 text-gray-700"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              aria-label="Toggle menu"
            >
              {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>

          {/* Mobile Menu */}
          <AnimatePresence>
            {isMenuOpen && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="md:hidden pb-4 space-y-3"
              >
                <a
                  href="/features"
                  onClick={() => setIsMenuOpen(false)}
                  className="block w-full text-left px-4 py-2 text-gray-700 hover:text-primary-600 transition-colors"
                >
                  Features
                </a>
                <a
                  href="/about"
                  onClick={() => setIsMenuOpen(false)}
                  className="block w-full text-left px-4 py-2 text-gray-700 hover:text-primary-600 transition-colors"
                >
                  About
                </a>
                <a
                  href="/enterprise"
                  onClick={() => setIsMenuOpen(false)}
                  className="block w-full text-left px-4 py-2 text-gray-700 hover:text-primary-600 transition-colors"
                >
                  Enterprise
                </a>
                <a
                  href="/contact"
                  onClick={() => setIsMenuOpen(false)}
                  className="block w-full text-left px-4 py-2 text-gray-700 hover:text-primary-600 transition-colors"
                >
                  Contact
                </a>
                <button
                  onClick={() => {
                    openAuth("login");
                    setIsMenuOpen(false);
                  }}
                  className="block w-full text-left px-4 py-2 text-gray-700 hover:text-primary-600 transition-colors"
                >
                  Login
                </button>
                <button
                  onClick={() => {
                    openAuth("signup");
                    setIsMenuOpen(false);
                  }}
                  className="block w-full text-left px-4 py-2 bg-primary-600 text-white rounded-lg font-medium"
                >
                  Sign Up
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </nav>
      </header>

      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        initialType={authType}
      />
    </>
  );
}


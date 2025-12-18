"use client";

import { motion } from "framer-motion";
import { ArrowRight, Sparkles } from "lucide-react";
import ShaderAnimation from "@/components/ui/ShaderAnimation";
import PatientDashboardMockup from "@/components/ui/PatientDashboardMockup";

export default function HeroWithShader() {
  const scrollToCTA = () => {
    const ctaSection = document.getElementById("cta-section");
    ctaSection?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative w-full h-screen overflow-hidden flex items-center justify-center">
      {/* Shader Background */}
      <ShaderAnimation
        primaryColor="#0284c7"
        secondaryColor="#0369a1"
        accentColor="#0ea5e9"
        speed={1.0}
      />

      {/* Subtle overlay for better text readability */}
      <div className="absolute inset-0 bg-gradient-to-b from-white/20 via-transparent to-white/30 z-[1]" />

      {/* Content Layer */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
        <div className="text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-4 py-2 bg-white/90 backdrop-blur-sm border border-blue-200 rounded-full text-blue-700 text-sm font-medium mb-8 shadow-lg"
          >
            <Sparkles size={16} />
            <span>Powered by AI • HIPAA & GDPR Compliant</span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-bold tracking-tight mb-6"
          >
            <span className="block text-gray-900 drop-shadow-lg">Redefining</span>
            <span className="block bg-gradient-to-r from-blue-600 via-blue-500 to-blue-400 bg-clip-text text-transparent drop-shadow-lg">
              Connected Healthcare
            </span>
          </motion.h1>

          {/* Subtext */}
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-xl sm:text-2xl text-gray-700 max-w-3xl mx-auto mb-12 leading-relaxed drop-shadow-md"
          >
            MedTrack empowers patients, doctors, and institutions to collaborate seamlessly through intelligent health data management.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={scrollToCTA}
              className="group px-8 py-4 bg-blue-600 text-white rounded-full font-semibold text-lg shadow-2xl shadow-blue-600/40 hover:shadow-3xl hover:shadow-blue-600/50 transition-all flex items-center gap-2 backdrop-blur-sm"
            >
              Get Started
              <ArrowRight className="group-hover:translate-x-1 transition-transform" size={20} />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-8 py-4 bg-white/90 backdrop-blur-sm text-gray-900 rounded-full font-semibold text-lg border-2 border-gray-200 hover:border-blue-300 transition-all shadow-lg"
            >
              Learn More
            </motion.button>
          </motion.div>

          {/* Patient Dashboard Mockup */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="mt-20 lg:mt-32"
          >
            <div className="relative max-w-5xl mx-auto">
              <div className="aspect-video bg-white/80 backdrop-blur-md rounded-3xl shadow-2xl border border-blue-200/50 overflow-hidden">
                {/* Dashboard Mockup */}
                <div className="absolute inset-4">
                  <PatientDashboardMockup />
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2 z-10"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-6 h-10 border-2 border-gray-400 rounded-full flex justify-center bg-white/50 backdrop-blur-sm"
        >
          <motion.div
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-1 h-3 bg-gray-400 rounded-full mt-2"
          />
        </motion.div>
      </motion.div>
    </section>
  );
}


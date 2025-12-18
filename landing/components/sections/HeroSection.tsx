"use client";

import { ContainerScroll } from "@/components/ui/ContainerScroll";
import { ArrowRight, Calendar } from "lucide-react";
import { motion } from "framer-motion";

export function HeroSection() {
  return (
    <ContainerScroll
      titleComponent={
        <div className="text-center space-y-8">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-4xl md:text-6xl lg:text-7xl font-bold text-gray-900 leading-tight"
          >
            MedTrack — Enterprise-Grade
            <br />
            <span className="bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
              Medication & Clinical Intelligence
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-xl md:text-2xl text-gray-600 max-w-3xl mx-auto leading-relaxed"
          >
            Privacy-first AI that improves adherence, powers research, and streamlines clinical workflows.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4"
          >
            <a
              href="/enterprise"
              className="group px-8 py-4 bg-blue-600 text-white rounded-lg font-semibold text-lg shadow-lg hover:shadow-xl hover:bg-blue-700 transition-all flex items-center gap-2"
            >
              Request Enterprise Demo
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </a>
            <a
              href="/contact"
              className="group px-8 py-4 bg-white text-gray-900 border-2 border-gray-300 rounded-lg font-semibold text-lg hover:border-blue-600 hover:text-blue-600 transition-all flex items-center gap-2"
            >
              <Calendar className="w-5 h-5" />
              Schedule a Meeting
            </a>
          </motion.div>
        </div>
      }
      imageSrc="https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?w=1200&h=800&fit=crop"
      imageAlt="Healthcare dashboard showing patient data and analytics"
    />
  );
}




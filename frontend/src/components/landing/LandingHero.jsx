"use client";

import { motion } from "framer-motion";
import { ArrowRight, Sparkles, Pill, Activity, TrendingUp, Heart, Calendar, MessageSquare, AlertTriangle, Lock, Users, Bell } from "lucide-react";

export default function LandingHero() {
  const scrollToCTA = () => {
    const ctaSection = document.getElementById("cta-section");
    ctaSection?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative min-h-[100vh] sm:min-h-screen flex items-start sm:items-center justify-center overflow-hidden pt-16 sm:pt-20 lg:pt-32 pb-8 sm:pb-12">
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-blue-50 -z-10" />
      
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden -z-10">
        <motion.div
          className="absolute top-20 left-10 w-48 h-48 sm:w-72 sm:h-72 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-20"
          animate={{
            scale: [1, 1.2, 1],
            x: [0, 100, 0],
            y: [0, 50, 0],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute bottom-20 right-10 w-64 h-64 sm:w-96 sm:h-96 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-20"
          animate={{
            scale: [1, 1.3, 1],
            x: [0, -100, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 lg:py-20">
        <div className="text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-3 py-1.5 sm:px-4 sm:py-2 bg-blue-50 border border-blue-200 rounded-full text-blue-700 text-xs sm:text-sm font-medium mb-4 sm:mb-6 lg:mb-8"
          >
            <Sparkles className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span className="text-[10px] sm:text-xs lg:text-sm">Powered by AI • HIPAA & GDPR Compliant</span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-bold tracking-tight mb-4 sm:mb-6 px-2"
          >
            <span className="block text-gray-900">Redefining</span>
            <span className="block bg-gradient-to-r from-blue-600 via-blue-600 to-blue-400 bg-clip-text text-transparent">
              Connected Healthcare
            </span>
          </motion.h1>

          {/* Subtext */}
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-base sm:text-lg md:text-xl lg:text-2xl text-gray-600 max-w-3xl mx-auto mb-6 sm:mb-8 lg:mb-12 leading-relaxed px-4"
          >
            MedTrack empowers patients, doctors, and institutions to collaborate seamlessly through intelligent health data management.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 px-4"
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={scrollToCTA}
              className="group w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 bg-blue-600 text-white rounded-full font-semibold text-base sm:text-lg shadow-xl shadow-blue-600/25 hover:shadow-2xl hover:shadow-blue-600/40 transition-all flex items-center justify-center gap-2"
            >
              Get Started
              <ArrowRight className="group-hover:translate-x-1 transition-transform w-4 h-4 sm:w-5 sm:h-5" />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="w-full sm:w-auto px-6 sm:px-8 py-3 sm:py-4 bg-white text-gray-900 rounded-full font-semibold text-base sm:text-lg border-2 border-gray-200 hover:border-blue-300 transition-all"
            >
              Learn More
            </motion.button>
          </motion.div>

          {/* Patient Dashboard Mockup */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="mt-8 sm:mt-12 md:mt-16 lg:mt-20 xl:mt-32 px-2 sm:px-4"
          >
            <div className="relative max-w-6xl mx-auto">
              <div className="bg-white rounded-xl sm:rounded-2xl lg:rounded-3xl shadow-2xl border border-gray-200 overflow-hidden scale-90 sm:scale-95 md:scale-100">
                {/* Browser Chrome */}
                <div className="bg-gray-100 px-2 sm:px-4 py-2 sm:py-3 border-b border-gray-200 flex items-center gap-1.5 sm:gap-2">
                  <div className="flex gap-1 sm:gap-1.5">
                    <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full bg-red-400"></div>
                    <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full bg-yellow-400"></div>
                    <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full bg-green-400"></div>
                  </div>
                  <div className="flex-1 mx-2 sm:mx-4 bg-white rounded-md px-2 sm:px-3 py-1 sm:py-1.5 text-[10px] sm:text-xs text-gray-500 truncate">
                    medtrack.com/dashboard
                  </div>
                </div>

                {/* Patient Dashboard */}
                <div className="bg-gray-50 p-3 sm:p-4 md:p-6 relative">
                  {/* Header Navigation */}
                  <div className="bg-white border-b border-gray-200 px-3 sm:px-4 md:px-6 py-2 sm:py-3 md:py-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-4 mb-4 sm:mb-6 rounded-t-lg">
                    <div className="flex items-center gap-2 sm:gap-4">
                      <div className="text-base sm:text-lg md:text-xl font-bold bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
                        MedTrack
                      </div>
                      <div className="px-2 sm:px-3 py-0.5 sm:py-1 bg-gray-100 rounded-full text-[10px] sm:text-xs font-medium text-gray-700">
                        Patient Portal
                      </div>
                    </div>
                    <div className="flex items-center gap-2 sm:gap-4 w-full sm:w-auto justify-between sm:justify-end">
                      <div className="text-xs sm:text-sm text-gray-600 hidden sm:block">alexsokol@gmail.com</div>
                      <div className="flex items-center gap-1.5 sm:gap-3">
                        <div className="w-6 h-6 sm:w-7 sm:h-7 md:w-8 md:h-8 rounded-full bg-gray-200 flex items-center justify-center cursor-pointer hover:bg-gray-300 transition-colors">
                          <Lock className="text-gray-600 w-3 h-3 sm:w-3.5 sm:h-3.5 md:w-4 md:h-4" />
                        </div>
                        <div className="w-6 h-6 sm:w-7 sm:h-7 md:w-8 md:h-8 rounded-full bg-gray-200 flex items-center justify-center cursor-pointer hover:bg-gray-300 transition-colors relative">
                          <Users className="text-gray-600 w-3 h-3 sm:w-3.5 sm:h-3.5 md:w-4 md:h-4" />
                          <div className="absolute -top-0.5 -right-0.5 sm:-top-1 sm:-right-1 w-2 h-2 sm:w-3 sm:h-3 bg-blue-600 rounded-full border border-white"></div>
                        </div>
                        <div className="w-6 h-6 sm:w-7 sm:h-7 md:w-8 md:h-8 rounded-full bg-gray-200 flex items-center justify-center cursor-pointer hover:bg-gray-300 transition-colors">
                          <Bell className="text-gray-600 w-3 h-3 sm:w-3.5 sm:h-3.5 md:w-4 md:h-4" />
                        </div>
                      </div>
                      <button className="px-2 sm:px-3 md:px-4 py-1.5 sm:py-2 text-xs sm:text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition-colors">
                        Sign Out
                      </button>
                    </div>
                  </div>

                  {/* Welcome Section */}
                  <div className="mb-4 sm:mb-6">
                    <h1 className="text-lg sm:text-xl md:text-2xl font-bold text-gray-900 mb-1 sm:mb-2">Welcome back</h1>
                    <p className="text-xs sm:text-sm text-gray-600">Track your medications, health metrics, and stay on top of your wellness journey.</p>
                  </div>

                  {/* Summary Cards */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3 md:gap-4 mb-4 sm:mb-6">
                    {[
                      { label: "Today's Medications", value: '5', icon: Pill, bgColor: 'bg-blue-50', iconColor: 'text-blue-600' },
                      { label: 'Health Metrics', value: '12', icon: Activity, bgColor: 'bg-teal-50', iconColor: 'text-teal-600' },
                      { label: 'Progress', value: '87%', icon: TrendingUp, bgColor: 'bg-blue-50', iconColor: 'text-blue-600' },
                      { label: 'Wellness Score', value: '92', icon: Heart, bgColor: 'bg-green-50', iconColor: 'text-green-600' }
                    ].map((stat, i) => {
                      const Icon = stat.icon;
                      return (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.9 + i * 0.05 }}
                          className="bg-white rounded-lg sm:rounded-xl p-2 sm:p-3 md:p-4 border border-gray-200 shadow-sm"
                        >
                          <div className={`inline-flex p-1.5 sm:p-2 ${stat.bgColor} rounded-lg mb-1.5 sm:mb-2`}>
                            <Icon className={`${stat.iconColor} w-3.5 h-3.5 sm:w-4 sm:h-4 md:w-5 md:h-5`} />
                          </div>
                          <div className="text-lg sm:text-xl md:text-2xl font-bold text-gray-900 mb-0.5 sm:mb-1">{stat.value}</div>
                          <div className="text-[10px] sm:text-xs text-gray-600 leading-tight">{stat.label}</div>
                        </motion.div>
                      );
                    })}
                  </div>

                  {/* Two Column Layout */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4 md:gap-6">
                    {/* Left Column */}
                    <div className="space-y-3 sm:space-y-4">
                      {/* Today's Medications */}
                      <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 1.1 }}
                        className="bg-white rounded-lg sm:rounded-xl border border-gray-200 p-3 sm:p-4 shadow-sm"
                      >
                        <div className="flex items-center justify-between mb-2 sm:mb-3">
                          <div className="flex items-center gap-1.5 sm:gap-2">
                            <Pill className="text-blue-600 w-3.5 h-3.5 sm:w-4 sm:h-4 md:w-5 md:h-5" />
                            <h3 className="text-sm sm:text-base font-semibold text-gray-900">Today's Medications</h3>
                          </div>
                          <button className="px-2 sm:px-3 py-0.5 sm:py-1 bg-blue-600 text-white text-[10px] sm:text-xs font-medium rounded-lg hover:bg-blue-700 transition-colors">
                            + Add
                          </button>
                        </div>
                        <div className="space-y-1.5 sm:space-y-2">
                          {[
                            { name: 'Propranolol', time: '8:00 AM', dosage: '40mg', status: 'taken' },
                            { name: 'Metformin', time: '12:00 PM', dosage: '500mg', status: 'pending' },
                            { name: 'Lisinopril', time: '6:00 PM', dosage: '10mg', status: 'pending' }
                          ].map((med, i) => (
                            <div key={i} className="flex items-center justify-between p-1.5 sm:p-2 bg-gray-50 rounded-lg border border-gray-100">
                              <div className="flex items-center gap-1.5 sm:gap-2">
                                <div className={`w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full ${med.status === 'taken' ? 'bg-green-500' : 'bg-gray-300'}`}></div>
                                <div>
                                  <div className="text-[10px] sm:text-xs font-semibold text-gray-900">{med.name}</div>
                                  <div className="text-[9px] sm:text-[10px] text-gray-500">{med.dosage}</div>
                                </div>
                              </div>
                              <div className="text-[10px] sm:text-xs font-medium text-gray-700">{med.time}</div>
                            </div>
                          ))}
                        </div>
                      </motion.div>

                      {/* Adherence Calendar */}
                      <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 1.2 }}
                        className="bg-white rounded-lg sm:rounded-xl border border-gray-200 p-3 sm:p-4 shadow-sm"
                      >
                        <div className="flex items-center gap-1.5 sm:gap-2 mb-2 sm:mb-3">
                          <Calendar className="text-blue-600 w-3.5 h-3.5 sm:w-4 sm:h-4 md:w-5 md:h-5" />
                          <h3 className="text-sm sm:text-base font-semibold text-gray-900">Adherence Calendar</h3>
                        </div>
                        <div className="grid grid-cols-7 gap-1 sm:gap-1.5">
                          {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((day, i) => (
                            <div key={i} className="text-center">
                              <div className="text-[9px] sm:text-[10px] text-gray-500 mb-0.5 sm:mb-1">{day}</div>
                              <div className={`w-5 h-5 sm:w-6 sm:h-6 md:w-7 md:h-7 rounded-lg flex items-center justify-center text-[10px] sm:text-xs font-medium ${
                                i < 5 ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-400'
                              }`}>
                                {i + 1}
                              </div>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    </div>

                    {/* Right Column */}
                    <div className="space-y-3 sm:space-y-4">
                      {/* Medication Schedule */}
                      <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 1.1 }}
                        className="bg-white rounded-lg sm:rounded-xl border border-gray-200 p-3 sm:p-4 shadow-sm"
                      >
                        <div className="flex items-center gap-1.5 sm:gap-2 mb-2 sm:mb-3">
                          <Calendar className="text-blue-600 w-3.5 h-3.5 sm:w-4 sm:h-4 md:w-5 md:h-5" />
                          <h3 className="text-sm sm:text-base font-semibold text-gray-900">Medication Schedule</h3>
                        </div>
                        <div className="space-y-1.5 sm:space-y-2">
                          {[
                            { time: '8:00 AM', meds: ['Propranolol 40mg'] },
                            { time: '12:00 PM', meds: ['Metformin 500mg'] },
                            { time: '6:00 PM', meds: ['Lisinopril 10mg'] }
                          ].map((schedule, i) => (
                            <div key={i} className="flex items-start gap-1.5 sm:gap-2 p-1.5 sm:p-2 bg-gray-50 rounded-lg">
                              <div className="text-[10px] sm:text-xs font-semibold text-gray-700 w-12 sm:w-14">{schedule.time}</div>
                              <div className="flex-1">
                                {schedule.meds.map((med, j) => (
                                  <div key={j} className="text-[10px] sm:text-xs text-gray-900">{med}</div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </motion.div>

                      {/* Side Effects */}
                      <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 1.2 }}
                        className="bg-white rounded-lg sm:rounded-xl border border-gray-200 p-3 sm:p-4 shadow-sm"
                      >
                        <div className="flex items-center justify-between mb-2 sm:mb-3">
                          <div className="flex items-center gap-1.5 sm:gap-2">
                            <AlertTriangle className="text-blue-600 w-3.5 h-3.5 sm:w-4 sm:h-4 md:w-5 md:h-5" />
                            <h3 className="text-sm sm:text-base font-semibold text-gray-900">Side Effects</h3>
                          </div>
                          <button className="px-2 sm:px-3 py-0.5 sm:py-1 bg-blue-600 text-white text-[10px] sm:text-xs font-medium rounded-lg hover:bg-blue-700 transition-colors">
                            + Add
                          </button>
                        </div>
                        <div className="text-[10px] sm:text-xs text-gray-500 text-center py-2 sm:py-3">
                          No side effects reported
                        </div>
                      </motion.div>
                    </div>
                  </div>

                  {/* Floating Action Button */}
                  <div className="absolute bottom-3 sm:bottom-4 md:bottom-6 right-3 sm:right-4 md:right-6">
                    <motion.div
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      className="w-10 h-10 sm:w-12 sm:h-12 bg-blue-600 rounded-full shadow-lg flex items-center justify-center cursor-pointer hover:bg-blue-700 transition-colors"
                    >
                      <MessageSquare className="text-white w-4 h-4 sm:w-5 sm:h-5" />
                    </motion.div>
                  </div>
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
        className="absolute bottom-4 sm:bottom-6 md:bottom-8 left-1/2 transform -translate-x-1/2 hidden sm:block"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-5 h-8 sm:w-6 sm:h-10 border-2 border-gray-400 rounded-full flex justify-center"
        >
          <motion.div
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-1 h-2 sm:h-3 bg-gray-400 rounded-full mt-1.5 sm:mt-2"
          />
        </motion.div>
      </motion.div>
    </section>
  );
}

import React, { useRef, useEffect, useState } from 'react';
import { motion, useInView, useScroll, useTransform } from 'framer-motion';
import LandingHeader from '../components/landing/LandingHeader';
import LandingFooter from '../components/landing/LandingFooter';
import {
  Pill, Brain, Shield, BarChart3, Bell, Calendar,
  MessageSquare, Camera, Activity, Database, CheckCircle2,
  ArrowRight, TrendingUp, AlertTriangle, FileText, Zap,
  Lock, Users, Stethoscope, Sparkles, Target, Clock, Heart
} from 'lucide-react';

// Medical-grade color system
const colors = {
  primary: {
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9', // Primary blue
    600: '#0284c7', // Deep blue
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e',
  },
  teal: {
    50: '#f0fdfa',
    100: '#ccfbf1',
    200: '#99f6e4',
    300: '#5eead4',
    400: '#2dd4bf',
    500: '#14b8a6', // Secondary teal
    600: '#0d9488',
    700: '#0f766e',
  },
  gray: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
  }
};

const features = [
  {
    id: 'medication-validation',
    icon: Pill,
    title: 'Medication Validation',
    subtitle: 'FDA Database Integration',
    description: 'Real-time validation against 3+ million FDA drug-label records. Instant drug class identification, dosage recommendations, and safety verification.',
    value: 'Eliminates medication errors through authoritative database matching',
    visual: {
      type: 'dashboard',
      elements: [
        { label: 'Input: "propranolol"', position: 'top-left' },
        { label: 'Validation: ✓ Verified', position: 'center', status: 'success' },
        { label: 'Drug Class: Beta Blocker', position: 'center-right' },
        { label: 'Dosage: 10mg, 20mg, 40mg, 80mg', position: 'bottom' }
      ]
    },
    metrics: ['1,055+ medications', '99.9% accuracy', '<100ms response'],
    tech: ['FDA Database', 'RxNorm API', 'Fuzzy Matching']
  },
  {
    id: 'ai-insights',
    icon: Brain,
    title: 'Predictive Health Analytics',
    subtitle: 'AI-Powered Pattern Recognition',
    description: 'Machine learning models analyze health trends to identify patterns, predict outcomes, and provide evidence-based recommendations.',
    value: 'Proactive health management through predictive insights',
    visual: {
      type: 'chart',
      elements: [
        { type: 'trend-line', direction: 'up', label: 'Blood Pressure Trend' },
        { type: 'alert', position: 'top-right', message: 'Pattern detected' }
      ]
    },
    metrics: ['Real-time analysis', 'Pattern detection', 'Predictive modeling'],
    tech: ['ML Models', 'Time Series Analysis', 'Pattern Recognition']
  },
  {
    id: 'pill-recognition',
    icon: Camera,
    title: 'Computer Vision Pill Identification',
    subtitle: 'Deep Learning Recognition',
    description: 'Advanced CNN models trained on thousands of medication images enable instant pill identification from photographs.',
    value: 'Reduces identification errors and improves medication safety',
    visual: {
      type: 'camera',
      elements: [
        { type: 'camera-view', label: 'Pill Image' },
        { type: 'recognition-result', label: 'Match: Propranolol 40mg', confidence: '94%' }
      ]
    },
    metrics: ['94%+ accuracy', 'Instant recognition', 'Multi-angle support'],
    tech: ['CNN Models', 'TensorFlow', 'Image Processing']
  },
  {
    id: 'health-metrics',
    icon: BarChart3,
    title: 'Comprehensive Metrics Tracking',
    subtitle: '40+ Health Parameters',
    description: 'Track blood pressure, glucose, weight, mood, sleep, and more. Real-time visualization with trend analysis and clinical-grade accuracy.',
    value: 'Complete health picture through comprehensive data collection',
    visual: {
      type: 'metrics-dashboard',
      elements: [
        { metric: 'BP', value: '120/80', trend: 'stable' },
        { metric: 'Glucose', value: '95 mg/dL', trend: 'normal' },
        { metric: 'Weight', value: '72.5 kg', trend: 'down' }
      ]
    },
    metrics: ['40+ metrics', 'Real-time sync', 'Clinical accuracy'],
    tech: ['Chart.js', 'Real-time Updates', 'Data Visualization']
  },
  {
    id: 'adherence',
    icon: Activity,
    title: 'Medication Adherence Engine',
    subtitle: 'Intelligent Compliance Tracking',
    description: 'Advanced algorithms track adherence patterns, identify risk factors, and generate actionable insights to improve medication compliance.',
    value: 'Improves patient outcomes through data-driven adherence management',
    visual: {
      type: 'adherence-chart',
      elements: [
        { day: 'Mon', status: 'taken' },
        { day: 'Tue', status: 'taken' },
        { day: 'Wed', status: 'missed' },
        { day: 'Thu', status: 'taken' }
      ]
    },
    metrics: ['95%+ accuracy', 'Pattern analysis', 'Risk prediction'],
    tech: ['Analytics Engine', 'Pattern Detection', 'Compliance Scoring']
  },
  {
    id: 'interactions',
    icon: AlertTriangle,
    title: 'Drug Interaction Detection',
    subtitle: 'Real-Time Safety Checks',
    description: 'Automated screening against comprehensive drug interaction databases. Instant warnings for potential adverse reactions and contraindications.',
    value: 'Prevents adverse drug events through proactive safety monitoring',
    visual: {
      type: 'interaction-alert',
      elements: [
        { drug1: 'Warfarin', drug2: 'Aspirin', severity: 'high', message: 'Increased bleeding risk' }
      ]
    },
    metrics: ['Real-time checks', 'Comprehensive database', 'Severity classification'],
    tech: ['Drug Database', 'Interaction Algorithms', 'Safety Engine']
  },
  {
    id: 'clinical-dashboard',
    icon: Stethoscope,
    title: 'Clinical Intelligence Platform',
    subtitle: 'Provider Dashboard',
    description: 'Comprehensive patient views, real-time data synchronization, and advanced analytics tools for evidence-based clinical decision making.',
    value: 'Enhances clinical workflow efficiency and patient care quality',
    visual: {
      type: 'clinical-view',
      elements: [
        { section: 'Patient Overview', data: 'Real-time vitals' },
        { section: 'Medication History', data: 'Complete timeline' },
        { section: 'Analytics', data: 'Trend analysis' }
      ]
    },
    metrics: ['Real-time sync', 'Multi-patient view', 'Clinical reports'],
    tech: ['Real-time Sync', 'Analytics Engine', 'Reporting Tools']
  },
  {
    id: 'security',
    icon: Shield,
    title: 'Enterprise Security Architecture',
    subtitle: 'HIPAA & GDPR Compliant',
    description: 'End-to-end encryption, role-based access control, audit logging, and compliance with healthcare data protection regulations.',
    value: 'Ensures patient data privacy and regulatory compliance',
    visual: {
      type: 'security-layers',
      elements: [
        { layer: 'Encryption', status: 'active' },
        { layer: 'Access Control', status: 'active' },
        { layer: 'Audit Logging', status: 'active' }
      ]
    },
    metrics: ['HIPAA compliant', 'GDPR compliant', 'Bank-level encryption'],
    tech: ['End-to-End Encryption', 'RBAC', 'Audit Logging']
  }
];

const FeatureSection = ({ feature, index, isEven }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });
  const Icon = feature.icon;

  return (
    <section
      ref={ref}
      className={`py-24 lg:py-32 ${isEven ? 'bg-white' : 'bg-gray-50'}`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className={`grid lg:grid-cols-2 gap-16 lg:gap-24 items-center ${isEven ? '' : 'lg:grid-flow-dense'}`}>
          {/* Content */}
          <motion.div
            initial={{ opacity: 0, x: isEven ? -40 : 40 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className={isEven ? '' : 'lg:col-start-2'}
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-blue-50 rounded-xl">
                <Icon className="text-blue-600" size={24} />
              </div>
              <span className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                {feature.subtitle}
              </span>
            </div>

            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6 leading-tight">
              {feature.title}
            </h2>

            <p className="text-lg text-gray-600 mb-6 leading-relaxed">
              {feature.description}
            </p>

            <div className="mb-8 p-4 bg-blue-50 border-l-4 border-blue-600 rounded-r-lg">
              <p className="text-sm font-medium text-gray-900">
                {feature.value}
              </p>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-3 gap-6 mb-8">
              {feature.metrics.map((metric, i) => (
                <div key={i}>
                  <div className="text-2xl font-bold text-gray-900 mb-1">
                    {metric.split(' ')[0]}
                  </div>
                  <div className="text-xs text-gray-500 leading-relaxed">
                    {metric.split(' ').slice(1).join(' ')}
                  </div>
                </div>
              ))}
            </div>

            {/* Tech Stack */}
            <div className="flex flex-wrap gap-2">
              {feature.tech.map((tech, i) => (
                <span
                  key={i}
                  className="px-3 py-1.5 bg-gray-100 text-gray-700 text-xs font-medium rounded-md"
                >
                  {tech}
                </span>
              ))}
            </div>
          </motion.div>

          {/* Visual */}
          <motion.div
            initial={{ opacity: 0, x: isEven ? 40 : -40 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className={isEven ? '' : 'lg:col-start-1 lg:row-start-1'}
          >
            <FeatureVisual feature={feature} />
          </motion.div>
        </div>
      </div>
    </section>
  );
};

const FeatureVisual = ({ feature }) => {
  const [isHovered, setIsHovered] = useState(false);

  if (feature.visual.type === 'dashboard') {
    return (
      <div
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className="relative bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden"
      >
        {/* Browser Chrome */}
        <div className="bg-gray-100 px-4 py-3 border-b border-gray-200 flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-400"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
            <div className="w-3 h-3 rounded-full bg-green-400"></div>
          </div>
          <div className="flex-1 mx-4 bg-white rounded-md px-3 py-1.5 text-xs text-gray-500">
            medtrack.com/dashboard
          </div>
        </div>

        {/* Patient Dashboard Mockup */}
        <div className="bg-gray-50 min-h-[600px]">
          {/* Header Navigation */}
          <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-xl font-bold bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent">
                MedTrack
              </div>
              <div className="px-3 py-1 bg-gray-100 rounded-full text-xs font-medium text-gray-700">
                Patient Portal
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-sm text-gray-600">alexsokol@gmail.com</div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center cursor-pointer hover:bg-gray-300 transition-colors">
                  <Lock className="text-gray-600" size={16} />
                </div>
                <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center cursor-pointer hover:bg-gray-300 transition-colors relative">
                  <Users className="text-gray-600" size={16} />
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-blue-600 rounded-full border-2 border-white"></div>
                </div>
                <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center cursor-pointer hover:bg-gray-300 transition-colors">
                  <Bell className="text-gray-600" size={16} />
                </div>
              </div>
              <button className="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition-colors">
                Sign Out
              </button>
            </div>
          </div>

          {/* Main Content */}
          <div className="p-6">
            {/* Welcome Section */}
            <div className="mb-6">
              <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome back</h1>
              <p className="text-gray-600">Track your medications, health metrics, and stay on top of your wellness journey.</p>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-4 gap-4 mb-6">
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
                    animate={isHovered ? { opacity: 1, y: 0 } : { opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="bg-white rounded-xl p-5 border border-gray-200 shadow-sm hover:shadow-md transition-shadow"
                  >
                    <div className={`inline-flex p-3 ${stat.bgColor} rounded-lg mb-3`}>
                      <Icon className={stat.iconColor} size={20} />
                    </div>
                    <div className="text-3xl font-bold text-gray-900 mb-1">{stat.value}</div>
                    <div className="text-sm text-gray-600">{stat.label}</div>
                  </motion.div>
                );
              })}
            </div>

            {/* Two Column Layout */}
            <div className="grid grid-cols-2 gap-6">
              {/* Left Column */}
              <div className="space-y-6">
                {/* Today's Medications */}
                <motion.div
                  animate={isHovered ? { scale: 1.01 } : { scale: 1 }}
                  transition={{ duration: 0.3 }}
                  className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Pill className="text-blue-600" size={20} />
                      <h3 className="text-lg font-semibold text-gray-900">Today's Medications</h3>
                    </div>
                    <button className="px-3 py-1.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors">
                      + Add
                    </button>
                  </div>
                  <div className="space-y-3">
                    {[
                      { name: 'Propranolol', time: '8:00 AM', dosage: '40mg', status: 'taken' },
                      { name: 'Metformin', time: '12:00 PM', dosage: '500mg', status: 'pending' },
                      { name: 'Lisinopril', time: '6:00 PM', dosage: '10mg', status: 'pending' }
                    ].map((med, i) => (
                      <div key={i} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-100">
                        <div className="flex items-center gap-3">
                          <div className={`w-2.5 h-2.5 rounded-full ${med.status === 'taken' ? 'bg-green-500' : 'bg-gray-300'}`}></div>
                          <div>
                            <div className="text-sm font-semibold text-gray-900">{med.name}</div>
                            <div className="text-xs text-gray-500">{med.dosage}</div>
                          </div>
                        </div>
                        <div className="text-sm font-medium text-gray-700">{med.time}</div>
                      </div>
                    ))}
                  </div>
                </motion.div>

                {/* Adherence Calendar */}
                <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                  <div className="flex items-center gap-2 mb-4">
                    <Calendar className="text-blue-600" size={20} />
                    <h3 className="text-lg font-semibold text-gray-900">Adherence Calendar</h3>
                  </div>
                  <div className="grid grid-cols-7 gap-2">
                    {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((day, i) => (
                      <div key={i} className="text-center">
                        <div className="text-xs text-gray-500 mb-1">{day}</div>
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-medium ${
                          i < 5 ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-400'
                        }`}>
                          {i + 1}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right Column */}
              <div className="space-y-6">
                {/* Medication Schedule */}
                <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                  <div className="flex items-center gap-2 mb-4">
                    <Calendar className="text-blue-600" size={20} />
                    <h3 className="text-lg font-semibold text-gray-900">Medication Schedule</h3>
                  </div>
                  <div className="space-y-2">
                    {[
                      { time: '8:00 AM', meds: ['Propranolol 40mg'] },
                      { time: '12:00 PM', meds: ['Metformin 500mg'] },
                      { time: '6:00 PM', meds: ['Lisinopril 10mg'] }
                    ].map((schedule, i) => (
                      <div key={i} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                        <div className="text-sm font-semibold text-gray-700 w-16">{schedule.time}</div>
                        <div className="flex-1">
                          {schedule.meds.map((med, j) => (
                            <div key={j} className="text-sm text-gray-900">{med}</div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Side Effects */}
                <motion.div
                  animate={isHovered ? { scale: 1.01 } : { scale: 1 }}
                  transition={{ duration: 0.3 }}
                  className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="text-blue-600" size={20} />
                      <h3 className="text-lg font-semibold text-gray-900">Side Effects</h3>
                    </div>
                    <button className="px-3 py-1.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors">
                      + Add
                    </button>
                  </div>
                  <div className="text-sm text-gray-500 text-center py-4">
                    No side effects reported
                  </div>
                </motion.div>
              </div>
            </div>
          </div>

          {/* Floating Action Button */}
          <div className="absolute bottom-6 right-6">
            <motion.div
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              className="w-14 h-14 bg-blue-600 rounded-full shadow-lg flex items-center justify-center cursor-pointer hover:bg-blue-700 transition-colors"
            >
              <MessageSquare className="text-white" size={24} />
            </motion.div>
          </div>
        </div>
      </div>
    );
  }

  if (feature.visual.type === 'chart') {
    return (
      <div className="relative bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden">
        <div className="p-8 bg-gradient-to-br from-gray-50 to-white">
          <div className="mb-6">
            <div className="h-6 bg-gray-200 rounded w-1/3 mb-2"></div>
            <div className="h-4 bg-gray-100 rounded w-1/4"></div>
          </div>
          
          {/* Chart Visualization */}
          <div className="h-64 bg-white rounded-lg border border-gray-200 p-6 relative">
            {/* Chart Grid */}
            <div className="absolute inset-6 flex flex-col justify-between">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="border-t border-gray-100"></div>
              ))}
            </div>
            
            {/* Trend Line */}
            <svg className="absolute inset-6 w-[calc(100%-3rem)] h-[calc(100%-3rem)]">
              <motion.path
                d="M 0 200 Q 100 150, 200 100 T 400 50"
                fill="none"
                stroke={colors.primary[600]}
                strokeWidth="3"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 1.5, ease: "easeInOut" }}
              />
            </svg>

            {/* Data Points */}
            <div className="absolute bottom-6 left-6 right-6 flex justify-between items-end">
              {['Mon', 'Tue', 'Wed', 'Thu', 'Fri'].map((day, i) => (
                <motion.div
                  key={day}
                  initial={{ scale: 0, y: 20 }}
                  animate={{ scale: 1, y: 0 }}
                  transition={{ delay: i * 0.1, duration: 0.3 }}
                  className="w-8 bg-blue-600 rounded-t"
                  style={{ height: `${60 + i * 15}px` }}
                />
              ))}
            </div>
          </div>

          {/* Alert Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-4 flex items-center gap-2 px-4 py-2 bg-amber-50 border border-amber-200 rounded-lg"
          >
            <AlertTriangle className="text-amber-600" size={16} />
            <span className="text-sm font-medium text-amber-900">Pattern detected: Upward trend</span>
          </motion.div>
        </div>
      </div>
    );
  }

  if (feature.visual.type === 'camera') {
    return (
      <div className="relative bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden">
        <div className="p-8 bg-gradient-to-br from-gray-50 to-white">
          {/* Camera View */}
          <div className="relative bg-gray-900 rounded-xl overflow-hidden mb-4 aspect-square">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-32 h-32 bg-white rounded-full flex items-center justify-center">
                <Pill className="text-gray-400" size={48} />
              </div>
            </div>
            {/* Camera Overlay */}
            <div className="absolute inset-0 border-4 border-white/20 rounded-xl" style={{
              clipPath: 'polygon(0% 0%, 0% 100%, 25% 100%, 25% 25%, 75% 25%, 75% 75%, 25% 75%, 25% 100%, 100% 100%, 100% 0%)'
            }}></div>
          </div>

          {/* Recognition Result */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-lg border-2 border-blue-200 p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-900">Match Found</span>
              <span className="text-xs font-semibold text-blue-600">94%</span>
            </div>
            <div className="text-lg font-bold text-gray-900">Propranolol 40mg</div>
            <div className="text-xs text-gray-500 mt-1">Beta Blocker • Tablet</div>
          </motion.div>
        </div>
      </div>
    );
  }

  if (feature.visual.type === 'metrics-dashboard') {
    return (
      <div className="relative bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden">
        <div className="p-8 bg-gradient-to-br from-gray-50 to-white">
          <div className="grid grid-cols-3 gap-4">
            {feature.visual.elements.map((metric, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="bg-white rounded-xl border border-gray-200 p-4"
              >
                <div className="text-xs text-gray-500 mb-2">{metric.metric}</div>
                <div className="text-2xl font-bold text-gray-900 mb-1">{metric.value}</div>
                <div className="flex items-center gap-1 text-xs">
                  <TrendingUp className="text-green-600" size={12} />
                  <span className="text-green-600 font-medium">{metric.trend}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (feature.visual.type === 'adherence-chart') {
    return (
      <div className="relative bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden">
        <div className="p-8 bg-gradient-to-br from-gray-50 to-white">
          <div className="mb-6">
            <div className="h-6 bg-gray-200 rounded w-1/2 mb-2"></div>
            <div className="h-4 bg-gray-100 rounded w-1/3"></div>
          </div>
          
          <div className="space-y-3">
            {feature.visual.elements.map((day, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
                className="flex items-center gap-4"
              >
                <div className="w-16 text-sm font-medium text-gray-700">{day.day}</div>
                <div className="flex-1 h-12 bg-gray-100 rounded-lg flex items-center justify-center">
                  {day.status === 'taken' ? (
                    <CheckCircle2 className="text-green-600" size={24} />
                  ) : (
                    <div className="w-6 h-6 border-2 border-red-300 rounded-full"></div>
                  )}
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <div className="text-sm font-semibold text-gray-900 mb-1">Adherence Rate</div>
            <div className="text-3xl font-bold text-blue-600">87%</div>
          </div>
        </div>
      </div>
    );
  }

  if (feature.visual.type === 'interaction-alert') {
    return (
      <div className="relative bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden">
        <div className="p-8 bg-gradient-to-br from-gray-50 to-white">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <Pill className="text-gray-400" size={20} />
              <div>
                <div className="font-semibold text-gray-900">Warfarin</div>
                <div className="text-xs text-gray-500">Anticoagulant</div>
              </div>
            </div>
            
            <div className="flex items-center justify-center">
              <div className="w-8 h-0.5 bg-gray-300"></div>
              <div className="px-3 text-gray-400">+</div>
              <div className="w-8 h-0.5 bg-gray-300"></div>
            </div>

            <div className="flex items-center gap-3">
              <Pill className="text-gray-400" size={20} />
              <div>
                <div className="font-semibold text-gray-900">Aspirin</div>
                <div className="text-xs text-gray-500">NSAID</div>
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="mt-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg"
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className="text-red-600 flex-shrink-0 mt-0.5" size={20} />
                <div>
                  <div className="font-semibold text-red-900 mb-1">High Severity Interaction</div>
                  <div className="text-sm text-red-700">Increased risk of bleeding. Monitor closely.</div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    );
  }

  if (feature.visual.type === 'clinical-view') {
    return (
      <div className="relative bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden">
        <div className="p-8 bg-gradient-to-br from-gray-50 to-white">
          <div className="space-y-6">
            {feature.visual.elements.map((section, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="bg-white rounded-xl border border-gray-200 p-6"
              >
                <div className="text-sm font-semibold text-gray-900 mb-3">{section.section}</div>
                <div className="text-xs text-gray-500">{section.data}</div>
                <div className="mt-4 h-2 bg-gray-100 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${60 + i * 10}%` }}
                    transition={{ delay: i * 0.1 + 0.3, duration: 0.8 }}
                    className="h-full bg-blue-600 rounded-full"
                  />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (feature.visual.type === 'security-layers') {
    return (
      <div className="relative bg-white rounded-2xl border border-gray-200 shadow-2xl overflow-hidden">
        <div className="p-8 bg-gradient-to-br from-gray-50 to-white">
          <div className="space-y-4">
            {feature.visual.elements.map((layer, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
                className="flex items-center justify-between p-4 bg-white rounded-lg border-2 border-gray-200"
              >
                <div className="flex items-center gap-3">
                  <Shield className="text-blue-600" size={20} />
                  <span className="font-medium text-gray-900">{layer.layer}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-sm text-gray-600">{layer.status}</span>
                </div>
              </motion.div>
            ))}
          </div>
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg text-center"
          >
            <div className="text-sm font-semibold text-green-900">All Security Layers Active</div>
            <div className="text-xs text-green-700 mt-1">HIPAA & GDPR Compliant</div>
          </motion.div>
        </div>
      </div>
    );
  }

  return null;
};

export default function FeaturesPage() {
  const { scrollYProgress } = useScroll();
  const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);

  return (
    <div className="min-h-screen bg-white">
      <LandingHeader />
      
      {/* Hero Section */}
      <section className="relative pt-32 pb-24 lg:pt-40 lg:pb-32 overflow-hidden">
        <motion.div
          style={{ opacity }}
          className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-teal-50"
        />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
            className="text-center max-w-4xl mx-auto"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-full mb-8">
              <Sparkles className="text-blue-600" size={16} />
              <span className="text-sm font-medium text-blue-900">Enterprise-Grade Platform</span>
            </div>
            
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 mb-6 leading-tight tracking-tight">
              Clinical-Grade
              <br />
              <span className="bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">
                Healthcare Technology
              </span>
            </h1>
            
            <p className="text-xl lg:text-2xl text-gray-600 mb-12 leading-relaxed max-w-3xl mx-auto">
              Built for hospitals, trusted by clinicians, designed for precision. 
              Every feature engineered to meet the highest standards of medical practice.
            </p>

            <div className="flex flex-wrap justify-center gap-6 text-sm">
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle2 className="text-green-600" size={18} />
                <span>HIPAA Compliant</span>
              </div>
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle2 className="text-green-600" size={18} />
                <span>FDA Database Integrated</span>
              </div>
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle2 className="text-green-600" size={18} />
                <span>Clinical Validation</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      {features.map((feature, index) => (
        <FeatureSection
          key={feature.id}
          feature={feature}
          index={index}
          isEven={index % 2 === 0}
        />
      ))}

      {/* Technology Stack */}
      <section className="py-24 lg:py-32 bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-4">Technology Foundation</h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Built on industry-leading technologies for reliability, security, and performance
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              { category: 'Frontend', tech: ['React', 'TypeScript', 'Tailwind CSS', 'Framer Motion'] },
              { category: 'Backend', tech: ['Node.js', 'Express', 'Prisma', 'PostgreSQL'] },
              { category: 'AI & ML', tech: ['OpenAI API', 'BioGPT', 'Custom Models', 'TensorFlow'] },
              { category: 'Infrastructure', tech: ['Docker', 'Supabase', 'AWS', 'Vercel'] }
            ].map((stack, i) => (
              <motion.div
                key={stack.category}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
              >
                <h3 className="text-lg font-semibold mb-4 text-gray-200">{stack.category}</h3>
                <ul className="space-y-2">
                  {stack.tech.map((tech, j) => (
                    <li key={j} className="text-gray-400 text-sm flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-teal-500 rounded-full"></div>
                      {tech}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 lg:py-32 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
              Ready to Transform Healthcare?
            </h2>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              Join leading healthcare institutions using MedTrack to improve patient outcomes and streamline clinical workflows.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="px-8 py-4 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
              >
                Schedule a Demo
                <ArrowRight size={20} />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="px-8 py-4 bg-white text-gray-900 border-2 border-gray-300 rounded-xl font-semibold hover:border-gray-400 transition-colors"
              >
                Contact Sales
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>

      <LandingFooter />
    </div>
  );
}

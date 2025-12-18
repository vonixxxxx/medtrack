import React, { useRef } from 'react';
import { motion, useInView, useScroll, useTransform } from 'framer-motion';
import LandingHeader from '../components/landing/LandingHeader';
import LandingFooter from '../components/landing/LandingFooter';
import {
  Target, Eye, Shield, TrendingUp, Heart, CheckCircle2,
  ArrowRight, Globe, Award, Database, Users, Stethoscope,
  Zap, FileText, Lock, BarChart3
} from 'lucide-react';

// Medical-grade color system
const colors = {
  primary: {
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9',
    600: '#0284c7', // Primary blue
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

const values = [
  {
    icon: Heart,
    title: 'Patient-Centered Design',
    description: 'Every feature is validated through clinical workflows and user research. We prioritize evidence-based design that improves real-world health outcomes.',
    metric: '95% user satisfaction',
    color: colors.primary[600]
  },
  {
    icon: Shield,
    title: 'Privacy & Security First',
    description: 'HIPAA and GDPR compliant architecture with end-to-end encryption. Patient data sovereignty is non-negotiable.',
    metric: 'Zero data breaches',
    color: colors.teal[600]
  },
  {
    icon: TrendingUp,
    title: 'Evidence-Based',
    description: 'All recommendations validated against authoritative medical databases. Clinical accuracy is our foundation.',
    metric: '3M+ FDA records',
    color: colors.primary[600]
  },
  {
    icon: Zap,
    title: 'Clinical Innovation',
    description: 'Advanced AI and machine learning applied to solve real healthcare challenges. Technology serves medicine, not the reverse.',
    metric: '<100ms response',
    color: colors.teal[600]
  }
];

const milestones = [
  {
    year: '2024 Q1',
    title: 'Platform Launch',
    description: 'MedTrack launched with comprehensive medication management, FDA database integration, and real-time health tracking capabilities.',
    metrics: ['1,000+ medications', 'FDA integration', 'HIPAA compliant']
  },
  {
    year: '2024 Q2',
    title: 'AI Integration',
    description: 'Integrated advanced AI for medication validation, predictive health analytics, and personalized clinical recommendations.',
    metrics: ['BioGPT integration', 'Pattern recognition', 'Predictive models']
  },
  {
    year: '2024 Q3',
    title: 'FDA Database Expansion',
    description: 'Integrated complete FDA drug-label database with 3+ million records. Achieved 99.9% medication validation accuracy.',
    metrics: ['3M+ records', '99.9% accuracy', 'Real-time validation']
  },
  {
    year: '2024 Q4',
    title: 'Clinical Adoption',
    description: 'Pilot deployments in healthcare institutions. Clinical validation and workflow integration with EMR systems.',
    metrics: ['5+ institutions', 'Clinical validation', 'EMR integration']
  },
  {
    year: '2025',
    title: 'Scale & Expansion',
    description: 'Expanding to research organizations and healthcare networks worldwide. Building partnerships with leading medical institutions.',
    metrics: ['Global expansion', 'Research partnerships', 'Network growth']
  }
];

const impactMetrics = [
  {
    icon: Database,
    value: '3M+',
    label: 'FDA Records',
    description: 'Comprehensive medication database'
  },
  {
    icon: CheckCircle2,
    value: '99.9%',
    label: 'Validation Accuracy',
    description: 'Clinical-grade precision'
  },
  {
    icon: Users,
    value: '1,055+',
    label: 'Medications Validated',
    description: 'Real-time verification'
  },
  {
    icon: Shield,
    value: 'HIPAA',
    label: 'Compliant',
    description: 'Enterprise security'
  },
  {
    icon: Globe,
    value: '24/7',
    label: 'Availability',
    description: 'Global infrastructure'
  },
  {
    icon: Zap,
    value: '<100ms',
    label: 'Response Time',
    description: 'Real-time processing'
  }
];

const ValueCard = ({ value, index }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });
  const Icon = value.icon;

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      className="bg-white rounded-2xl p-8 border border-gray-200 hover:border-blue-300 transition-all shadow-sm hover:shadow-lg"
    >
      <div className="flex items-start gap-4 mb-6">
        <div className="p-3 rounded-xl" style={{ backgroundColor: `${value.color}15` }}>
          <Icon className="text-gray-900" size={24} style={{ color: value.color }} />
        </div>
        <div className="flex-1">
          <h3 className="text-xl font-bold text-gray-900 mb-2">{value.title}</h3>
          <p className="text-gray-600 leading-relaxed mb-4">{value.description}</p>
          <div className="text-sm font-semibold" style={{ color: value.color }}>
            {value.metric}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

const MilestoneCard = ({ milestone, index, isLast }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, x: -30 }}
      animate={isInView ? { opacity: 1, x: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      className="relative flex gap-8"
    >
      {/* Timeline Line */}
      {!isLast && (
        <div className="absolute left-6 top-16 bottom-0 w-0.5 bg-gray-200 hidden lg:block" />
      )}

      {/* Timeline Dot */}
      <div className="flex-shrink-0 relative z-10">
        <div className="w-12 h-12 bg-blue-600 rounded-full border-4 border-white shadow-lg flex items-center justify-center">
          <CheckCircle2 className="text-white" size={20} />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 pb-16">
        <div className="bg-white rounded-2xl p-8 border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center gap-4 mb-4">
            <span className="text-sm font-semibold text-blue-600 px-3 py-1 bg-blue-50 rounded-full">
              {milestone.year}
            </span>
            <h3 className="text-2xl font-bold text-gray-900">{milestone.title}</h3>
          </div>
          <p className="text-gray-600 leading-relaxed mb-6">{milestone.description}</p>
          
          {/* Metrics */}
          <div className="flex flex-wrap gap-2">
            {milestone.metrics.map((metric, i) => (
              <span
                key={i}
                className="px-3 py-1.5 bg-gray-50 text-gray-700 text-xs font-medium rounded-md"
              >
                {metric}
              </span>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default function AboutPage() {
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
              <Target className="text-blue-600" size={16} />
              <span className="text-sm font-medium text-blue-900">Our Mission</span>
            </div>
            
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 mb-6 leading-tight tracking-tight">
              Advancing Healthcare Through
              <br />
              <span className="bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">
                Clinical Technology
              </span>
            </h1>
            
            <p className="text-xl lg:text-2xl text-gray-600 mb-12 leading-relaxed max-w-3xl mx-auto">
              We build evidence-based healthcare technology that empowers clinicians, 
              improves patient outcomes, and transforms medication management through 
              precision, security, and innovation.
            </p>

            <div className="flex flex-wrap justify-center gap-6 text-sm">
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle2 className="text-green-600" size={18} />
                <span>Evidence-Based</span>
              </div>
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle2 className="text-green-600" size={18} />
                <span>Clinical Validation</span>
              </div>
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle2 className="text-green-600" size={18} />
                <span>HIPAA Compliant</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Mission Statement */}
      <section className="py-24 lg:py-32 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="prose prose-lg max-w-none"
          >
            <div className="text-center mb-12">
              <Target className="w-12 h-12 text-blue-600 mx-auto mb-6" />
              <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
                Our Mission
              </h2>
            </div>

            <div className="space-y-6 text-gray-700 leading-relaxed">
              <p className="text-xl font-medium text-gray-900">
                MedTrack exists to bridge the gap between clinical practice and technology innovation. 
                We believe that healthcare technology must be built on three foundational principles:
              </p>
              
              <div className="space-y-4 pl-6 border-l-4 border-blue-600">
                <p className="text-lg">
                  <strong className="text-gray-900">Clinical Accuracy:</strong> Every feature is validated 
                  against authoritative medical databases. We integrate with FDA records, RxNorm, and clinical 
                  guidelines to ensure precision in medication validation and health recommendations.
                </p>
                
                <p className="text-lg">
                  <strong className="text-gray-900">Patient Safety:</strong> Security and privacy are not 
                  features—they are requirements. Our architecture is designed from the ground up to meet 
                  HIPAA and GDPR standards, ensuring patient data is protected at every level.
                </p>
                
                <p className="text-lg">
                  <strong className="text-gray-900">Evidence-Based Design:</strong> We apply advanced AI and 
                  machine learning to solve real healthcare challenges. Our technology serves medicine, 
                  providing clinicians and patients with tools that improve outcomes through data-driven insights.
                </p>
              </div>

              <p className="text-lg pt-4">
                Whether you're a patient managing complex medication regimens, a caregiver supporting 
                loved ones, or a healthcare provider delivering clinical care, MedTrack provides the 
                precision, security, and intelligence needed to improve health outcomes.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Core Values */}
      <section className="py-24 lg:py-32 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
              Core Principles
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              The foundational values that guide every decision and feature we build
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {values.map((value, index) => (
              <ValueCard key={value.title} value={value} index={index} />
            ))}
          </div>
        </div>
      </section>

      {/* Vision */}
      <section className="py-24 lg:py-32 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="text-center"
          >
            <Eye className="w-16 h-16 text-blue-600 mx-auto mb-8" />
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
              Our Vision
            </h2>
            <p className="text-xl text-gray-700 leading-relaxed mb-8 max-w-3xl mx-auto">
              We envision a healthcare ecosystem where medication management is seamless, 
              health data is actionable, and every clinical decision is supported by 
              evidence-based technology. Through innovation, collaboration, and unwavering 
              commitment to patient care, we're building the infrastructure for the future 
              of precision medicine.
            </p>
            <div className="flex items-center justify-center gap-2 text-blue-600 font-semibold">
              <span>Building the future of healthcare</span>
              <ArrowRight size={20} />
            </div>
          </motion.div>
        </div>
      </section>

      {/* Timeline */}
      <section className="py-24 lg:py-32 bg-gray-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
              Our Journey
            </h2>
            <p className="text-xl text-gray-600">
              Key milestones in building clinical-grade healthcare technology
            </p>
          </motion.div>

          <div className="space-y-0">
            {milestones.map((milestone, index) => (
              <MilestoneCard
                key={index}
                milestone={milestone}
                index={index}
                isLast={index === milestones.length - 1}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Impact Metrics */}
      <section className="py-24 lg:py-32 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
              Platform Metrics
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Measurable indicators of our platform's scale, accuracy, and reliability
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {impactMetrics.map((metric, index) => {
              const Icon = metric.icon;
              return (
                <motion.div
                  key={metric.label}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                  className="bg-gray-50 rounded-2xl p-8 border border-gray-200 text-center"
                >
                  <div className="inline-flex p-4 bg-blue-50 rounded-xl mb-4">
                    <Icon className="text-blue-600" size={24} />
                  </div>
                  <div className="text-4xl font-bold text-gray-900 mb-2">{metric.value}</div>
                  <div className="text-lg font-semibold text-gray-900 mb-1">{metric.label}</div>
                  <div className="text-sm text-gray-500">{metric.description}</div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 lg:py-32 bg-gray-900 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-6">
              Join Us in Transforming Healthcare
            </h2>
            <p className="text-xl text-gray-300 mb-8 leading-relaxed">
              Partner with us to build the future of clinical technology. 
              Whether you're a healthcare institution, researcher, or clinician, 
              we're here to support your mission.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="px-8 py-4 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
              >
                Get in Touch
                <ArrowRight size={20} />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="px-8 py-4 bg-white/10 text-white border-2 border-white/20 rounded-xl font-semibold hover:bg-white/20 transition-colors"
              >
                View Features
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>

      <LandingFooter />
    </div>
  );
}

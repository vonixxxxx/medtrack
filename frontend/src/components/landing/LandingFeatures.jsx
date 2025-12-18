"use client";

import { motion } from "framer-motion";
import { Users, Stethoscope, Building2, Brain, Shield, BarChart3 } from "lucide-react";

const features = [
  {
    icon: Users,
    title: "For Patients",
    description: "Track medications, appointments, and health metrics with ease. Get personalized insights and reminders to stay on top of your health journey.",
    color: "from-blue-500 to-cyan-500",
  },
  {
    icon: Stethoscope,
    title: "For Clinicians",
    description: "Access real-time patient data, secure communication channels, and accurate record-keeping tools. Make informed decisions faster.",
    color: "from-blue-500 to-blue-600",
  },
  {
    icon: Building2,
    title: "For Institutions",
    description: "AI-processed, research-ready data aggregation and analytics. Transform healthcare delivery with comprehensive insights and reporting.",
    color: "from-purple-500 to-pink-500",
  },
];

const additionalFeatures = [
  {
    icon: Brain,
    title: "AI-Powered Insights",
    description: "Advanced machine learning algorithms analyze health patterns and provide actionable recommendations.",
  },
  {
    icon: Shield,
    title: "Enterprise Security",
    description: "Bank-level encryption, HIPAA & GDPR compliant, ensuring your data is protected at every step.",
  },
  {
    icon: BarChart3,
    title: "Advanced Analytics",
    description: "Comprehensive dashboards and reports to track outcomes and improve care quality.",
  },
];

export default function LandingFeatures() {
  return (
    <section id="features" className="py-24 lg:py-32 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-4">
            Built for Everyone
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Comprehensive healthcare management solutions tailored to your needs
          </p>
        </motion.div>

        {/* Main Features */}
        <div className="grid md:grid-cols-3 gap-8 mb-20">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ y: -8 }}
                className="group relative p-8 bg-gradient-to-br from-gray-50 to-white rounded-3xl border border-gray-100 hover:border-blue-200 transition-all shadow-sm hover:shadow-xl"
              >
                <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-br ${feature.color} mb-6 shadow-lg`}>
                  <Icon className="text-white" size={28} />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
                <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-blue-500/0 to-blue-500/0 group-hover:from-blue-500/5 group-hover:to-blue-500/5 transition-all -z-10" />
              </motion.div>
            );
          })}
        </div>

        {/* Additional Features Grid */}
        <div className="grid sm:grid-cols-3 gap-6">
          {additionalFeatures.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.3 + index * 0.1 }}
                className="p-6 bg-gray-50 rounded-2xl border border-gray-100 hover:border-blue-200 transition-all"
              >
                <Icon className="text-blue-600 mb-4" size={24} />
                <h4 className="text-lg font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h4>
                <p className="text-sm text-gray-600">
                  {feature.description}
                </p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

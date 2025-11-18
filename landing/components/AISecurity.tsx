"use client";

import { motion } from "framer-motion";
import { Shield, Lock, Brain, Database, Eye, CheckCircle2 } from "lucide-react";

const securityFeatures = [
  {
    icon: Shield,
    title: "HIPAA Compliant",
    description: "Fully compliant with Health Insurance Portability and Accountability Act standards",
  },
  {
    icon: Lock,
    title: "GDPR Compliant",
    description: "Meets all General Data Protection Regulation requirements for EU data protection",
  },
  {
    icon: Database,
    title: "End-to-End Encryption",
    description: "Bank-level encryption ensures your data is protected at rest and in transit",
  },
  {
    icon: Eye,
    title: "Data Anonymization",
    description: "Advanced AI automatically anonymizes sensitive data for research purposes",
  },
];

const aiFeatures = [
  {
    title: "Intelligent Data Structuring",
    description: "AI automatically organizes and structures health data for easy access and analysis",
  },
  {
    title: "Pattern Recognition",
    description: "Machine learning algorithms identify health patterns and trends in patient data",
  },
  {
    title: "Predictive Analytics",
    description: "Advanced models predict potential health outcomes and recommend interventions",
  },
];

export default function AISecurity() {
  return (
    <section id="security" className="py-24 lg:py-32 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* AI Section */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-50 border border-purple-200 rounded-full text-purple-700 text-sm font-medium mb-6">
              <Brain size={16} />
              <span>AI-Powered</span>
            </div>
            <h2 className="text-4xl sm:text-5xl font-bold text-gray-900 mb-6">
              Intelligent Data Processing
            </h2>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              Our advanced AI systems automatically structure, analyze, and anonymize health data, making it research-ready while maintaining the highest standards of privacy and security.
            </p>

            <div className="space-y-4">
              {aiFeatures.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="flex items-start gap-4"
                >
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-100 flex items-center justify-center mt-1">
                    <CheckCircle2 className="text-primary-600" size={16} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-1">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600">
                      {feature.description}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Security Section */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-50 border border-green-200 rounded-full text-green-700 text-sm font-medium mb-6">
              <Shield size={16} />
              <span>Enterprise Security</span>
            </div>
            <h2 className="text-4xl sm:text-5xl font-bold text-gray-900 mb-6">
              Your Data, Protected
            </h2>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              We take security seriously. Every aspect of our platform is designed with privacy and compliance in mind, ensuring your sensitive health information is always protected.
            </p>

            <div className="grid sm:grid-cols-2 gap-4">
              {securityFeatures.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6, delay: index * 0.1 }}
                    whileHover={{ y: -4 }}
                    className="p-6 bg-gradient-to-br from-gray-50 to-white rounded-2xl border border-gray-100 hover:border-primary-200 transition-all"
                  >
                    <Icon className="text-primary-600 mb-3" size={24} />
                    <h3 className="font-semibold text-gray-900 mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-sm text-gray-600">
                      {feature.description}
                    </p>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        </div>

        {/* Visual Data Flow */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-20"
        >
          <div className="bg-gradient-to-br from-primary-50 to-blue-50 rounded-3xl p-12 border border-primary-100">
            <div className="flex flex-col md:flex-row items-center justify-between gap-8">
              {[
                { label: "Data Input", color: "bg-blue-500" },
                { label: "AI Processing", color: "bg-purple-500" },
                { label: "Anonymization", color: "bg-green-500" },
                { label: "Secure Storage", color: "bg-primary-500" },
              ].map((step, index) => (
                <div key={step.label} className="flex items-center gap-4">
                  <div className={`w-16 h-16 ${step.color} rounded-2xl flex items-center justify-center text-white font-bold shadow-lg`}>
                    {index + 1}
                  </div>
                  <div className="text-gray-700 font-medium">{step.label}</div>
                  {index < 3 && (
                    <div className="hidden md:block w-12 h-0.5 bg-gray-300 mx-4" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}


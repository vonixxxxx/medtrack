"use client";

import { motion } from "framer-motion";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Shield, Heart, Users, Target, CheckCircle2 } from "lucide-react";

export default function AboutPage() {
  return (
    <main className="min-h-screen">
      <Header />
      <section className="pt-32 pb-16 bg-gradient-to-b from-white to-gray-50">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16 max-w-3xl mx-auto"
          >
            <h1 className="text-4xl lg:text-5xl font-bold mb-6">About MedTrack</h1>
            <p className="text-2xl text-gray-600 leading-relaxed">
              Empower clinicians and researchers with privacy-first, actionable patient intelligence.
            </p>
          </motion.div>

          {/* Mission */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-24"
          >
            <div className="bg-gradient-to-br from-primary-50 to-blue-50 rounded-2xl p-8 lg:p-12 border border-primary-200">
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-primary-100 rounded-xl">
                  <Target className="w-8 h-8 text-primary-600" />
                </div>
                <h2 className="text-3xl font-bold">Our Mission</h2>
              </div>
              <p className="text-lg text-gray-700 leading-relaxed">
                MedTrack is built on the principle that healthcare data should be both powerful and private. 
                We provide enterprise-grade medication and clinical intelligence tools that enable better patient 
                outcomes while maintaining the highest standards of privacy, security, and compliance. Our platform 
                empowers healthcare providers, researchers, and institutions to make data-driven decisions without 
                compromising patient privacy.
              </p>
            </div>
          </motion.div>

          {/* Compliance */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-24"
          >
            <h2 className="text-3xl font-bold mb-8 text-center">Compliance & Privacy</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="p-6 bg-white rounded-xl border border-gray-200 shadow-sm">
                <Shield className="w-8 h-8 text-primary-600 mb-4" />
                <h3 className="text-xl font-semibold mb-3">HIPAA Compliant</h3>
                <p className="text-gray-600">
                  Fully compliant with Health Insurance Portability and Accountability Act standards. 
                  All patient data is encrypted, access-controlled, and audited.
                </p>
              </div>
              <div className="p-6 bg-white rounded-xl border border-gray-200 shadow-sm">
                <Shield className="w-8 h-8 text-primary-600 mb-4" />
                <h3 className="text-xl font-semibold mb-3">GDPR Ready</h3>
                <p className="text-gray-600">
                  Meets all General Data Protection Regulation requirements for EU data protection. 
                  Patients have full control over their data with right to access, rectification, and deletion.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Privacy-First Approach */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-24"
          >
            <h2 className="text-3xl font-bold mb-8 text-center">Privacy-First Approach</h2>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                {
                  icon: Shield,
                  title: "End-to-End Encryption",
                  description: "All data encrypted at rest and in transit using industry-standard protocols.",
                },
                {
                  icon: Users,
                  title: "Role-Based Access",
                  description: "Granular permissions ensure only authorized personnel access sensitive data.",
                },
                {
                  icon: Heart,
                  title: "Data Minimization",
                  description: "We collect only what's necessary and anonymize data for research purposes.",
                },
              ].map((item, index) => {
                const Icon = item.icon;
                return (
                  <div key={index} className="p-6 bg-white rounded-xl border border-gray-200 shadow-sm">
                    <Icon className="w-8 h-8 text-primary-600 mb-4" />
                    <h3 className="text-xl font-semibold mb-3">{item.title}</h3>
                    <p className="text-gray-600">{item.description}</p>
                  </div>
                );
              })}
            </div>
          </motion.div>

          {/* Partnerships Roadmap */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mb-24"
          >
            <h2 className="text-3xl font-bold mb-8 text-center">Partnerships & Roadmap</h2>
            <div className="bg-white rounded-xl border border-gray-200 p-8 lg:p-12 shadow-sm">
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <CheckCircle2 className="w-6 h-6 text-green-600 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-semibold text-lg mb-2">Current Partnerships</h3>
                    <p className="text-gray-600">
                      Collaborating with leading healthcare institutions and research organizations to advance 
                      medical research and improve patient care outcomes.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <CheckCircle2 className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-semibold text-lg mb-2">Integration Roadmap</h3>
                    <p className="text-gray-600">
                      Expanding EMR integrations, developing advanced AI models, and building comprehensive 
                      analytics dashboards for population health management.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <CheckCircle2 className="w-6 h-6 text-primary-600 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-semibold text-lg mb-2">Research Initiatives</h3>
                    <p className="text-gray-600">
                      Partnering with academic institutions to enable groundbreaking research while maintaining 
                      strict privacy and ethical standards.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      <Footer />
    </main>
  );
}






"use client";

import { motion } from "framer-motion";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import FeaturesGrid from "@/components/FeaturesGrid";
import { Pill, TrendingUp, UserCheck, Brain, Database, Plug, ArrowRight } from "lucide-react";
import Link from "next/link";

const detailedFeatures = [
  {
    id: "med",
    title: "Medication Management",
    icon: Pill,
    color: "from-blue-500 to-cyan-500",
    description: "Comprehensive medication tracking and management system",
    features: [
      "Interactive medication timeline with visual history",
      "Quick mark-as-taken with one-tap interface",
      "AI-powered pill recognition from photos",
      "Drug interaction checking and alerts",
      "Refill reminders and prescription management",
      "Medication adherence scoring",
    ],
  },
  {
    id: "adhr",
    title: "Adherence Engine",
    icon: TrendingUp,
    color: "from-green-500 to-emerald-500",
    description: "Advanced analytics for medication adherence patterns",
    features: [
      "Streak tracking and visualization",
      "Pattern analysis and insights",
      "Predictive adherence modeling",
      "Customizable penalty systems",
      "Patient engagement metrics",
      "Clinician adherence reports",
    ],
  },
  {
    id: "clin",
    title: "Clinician Workspace",
    icon: UserCheck,
    color: "from-purple-500 to-pink-500",
    description: "Streamlined workspace for healthcare providers",
    features: [
      "SOAP note templates and automation",
      "Seamless patient switching",
      "Advanced filtering and search",
      "Real-time patient data sync",
      "Clinical decision support",
      "Billing and documentation tools",
    ],
  },
  {
    id: "ai",
    title: "AI Insights",
    icon: Brain,
    color: "from-orange-500 to-red-500",
    description: "Intelligent health analysis and recommendations",
    features: [
      "Personalized health reports",
      "Predictive health alerts",
      "Offline AI option (LLaMA)",
      "Natural language health queries",
      "Pattern recognition and trends",
      "Research-grade analytics",
    ],
  },
  {
    id: "data",
    title: "Data & Research",
    icon: Database,
    color: "from-indigo-500 to-blue-500",
    description: "Research-ready data export and analytics",
    features: [
      "Automated data anonymization",
      "K-anonymity compliance",
      "Structured export formats (CSV, JSON)",
      "Population-level analytics",
      "Clinical study data preparation",
      "HIPAA-compliant data sharing",
    ],
  },
  {
    id: "deploy",
    title: "Integrations & Deployment",
    icon: Plug,
    color: "from-primary-500 to-blue-500",
    description: "Enterprise integrations and flexible deployment",
    features: [
      "RESTful API and SDKs",
      "SSO and SAML support",
      "EMR integration (HL7, FHIR)",
      "On-premise deployment option",
      "Private cloud hosting",
      "Custom integration support",
    ],
  },
];

export default function FeaturesPage() {
  return (
    <main className="min-h-screen">
      <Header />
      <section className="pt-32 pb-16 bg-gradient-to-b from-white to-gray-50">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h1 className="text-4xl lg:text-5xl font-bold mb-4">Features</h1>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Comprehensive healthcare management solutions designed for patients, clinicians, and institutions
            </p>
          </motion.div>

          <div className="space-y-24">
            {detailedFeatures.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.id}
                  id={feature.id}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="grid lg:grid-cols-2 gap-12 items-center"
                >
                  <div className={`order-2 lg:order-${index % 2 === 0 ? "1" : "2"}`}>
                    <div className={`inline-flex p-4 rounded-xl bg-gradient-to-br ${feature.color} mb-6 shadow-lg`}>
                      <Icon className="text-white" size={32} />
                    </div>
                    <h2 className="text-3xl font-bold mb-4">{feature.title}</h2>
                    <p className="text-lg text-gray-600 mb-6">{feature.description}</p>
                    <ul className="space-y-3">
                      {feature.features.map((item, i) => (
                        <li key={i} className="flex items-start gap-3">
                          <div className="flex-shrink-0 w-5 h-5 rounded-full bg-primary-100 flex items-center justify-center mt-0.5">
                            <div className="w-2 h-2 rounded-full bg-primary-600" />
                          </div>
                          <span className="text-gray-700">{item}</span>
                        </li>
                      ))}
                    </ul>
                    <Link
                      href="/contact"
                      className="mt-6 inline-flex items-center gap-2 px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-semibold"
                    >
                      Request demo <ArrowRight className="w-5 h-5" />
                    </Link>
                  </div>
                  <div className={`order-1 lg:order-${index % 2 === 0 ? "2" : "1"}`}>
                    <div className="relative aspect-video bg-gradient-to-br from-gray-100 to-gray-200 rounded-xl overflow-hidden shadow-xl">
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className={`w-24 h-24 mx-auto mb-4 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center`}>
                            <Icon className="text-white" size={48} />
                          </div>
                          <p className="text-gray-600">Interactive demo coming soon</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>
      <FeaturesGrid />
      <Footer />
    </main>
  );
}






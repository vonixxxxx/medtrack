"use client";

import { InfoCard } from "@/components/ui/InfoCard";
import { Shield, Database, Cloud, Brain } from "lucide-react";
import { motion } from "framer-motion";
import { useInView } from "framer-motion";
import { useRef } from "react";

const benefits = [
  {
    title: "Enterprise-Grade Security",
    description: "HIPAA & GDPR compliant with end-to-end encryption. Bank-level security for sensitive healthcare data with comprehensive audit trails.",
    icon: <Shield className="w-8 h-8" />,
    delay: 0,
  },
  {
    title: "Research-Ready Datasets",
    description: "Anonymized, structured data exports with k-anonymity pipelines. Ready for clinical trials, research studies, and academic publications.",
    icon: <Database className="w-8 h-8" />,
    delay: 0.2,
  },
  {
    title: "Scalable Deployments",
    description: "Cloud, private cloud, or on-premise deployment options. SSO/SAML integration, role-based access control, and enterprise SSO support.",
    icon: <Cloud className="w-8 h-8" />,
    delay: 0.4,
  },
  {
    title: "AI-Powered Insights",
    description: "Advanced machine learning models provide predictive analytics, medication validation, and intelligent health pattern recognition for better patient care.",
    icon: <Brain className="w-8 h-8" />,
    delay: 0.6,
  },
];

export function HospitalsSection() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section className="py-20 md:py-32 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-4">
            Why Hospitals & Institutions Choose MedTrack
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Enterprise-grade infrastructure designed for healthcare organizations that demand security, scalability, and research capabilities.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {benefits.map((benefit, index) => (
            <InfoCard
              key={index}
              title={benefit.title}
              description={benefit.description}
              icon={benefit.icon}
              delay={benefit.delay}
              variant={index === 0 ? "highlight" : "default"}
            />
          ))}
        </div>
      </div>
    </section>
  );
}




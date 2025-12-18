"use client";

import { motion } from "framer-motion";
import { ArrowRight, Pill, TrendingUp, UserCheck, Brain, Database, Plug } from "lucide-react";
import { useState } from "react";
import MockupModal from "./MockupModal";

const featureList = [
  {
    id: "med",
    title: "Medication Management",
    desc: "Timeline, quick mark-as-taken, pill recognition",
    icon: Pill,
    color: "from-blue-500 to-cyan-500",
  },
  {
    id: "adhr",
    title: "Adherence Engine",
    desc: "Streak detection, pattern analysis",
    icon: TrendingUp,
    color: "from-green-500 to-emerald-500",
  },
  {
    id: "clin",
    title: "Clinician Workspace",
    desc: "SOAP notes, patient switching",
    icon: UserCheck,
    color: "from-purple-500 to-pink-500",
  },
  {
    id: "ai",
    title: "AI Insights",
    desc: "Personalized reports, offline AI option",
    icon: Brain,
    color: "from-orange-500 to-red-500",
  },
  {
    id: "data",
    title: "Data & Research",
    desc: "Anonymize, export, clinical-ready datasets",
    icon: Database,
    color: "from-indigo-500 to-blue-500",
  },
  {
    id: "deploy",
    title: "Integrations",
    desc: "APIs, SSO, EMR connectors",
    icon: Plug,
    color: "from-primary-500 to-blue-500",
  },
];

export default function FeaturesGrid() {
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  return (
    <>
      <section id="features" className="py-16 lg:py-24 bg-gradient-to-b from-white to-gray-50">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl lg:text-4xl font-semibold mb-4">Core capabilities</h2>
            <p className="text-gray-600 text-lg max-w-2xl mx-auto">
              Comprehensive features designed for modern healthcare workflows
            </p>
          </motion.div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
            {featureList.map((f, i) => {
              const Icon = f.icon;
              return (
                <motion.div
                  key={f.id}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1, duration: 0.6 }}
                  whileHover={{ y: -8, scale: 1.02 }}
                  className="group relative p-6 lg:p-8 rounded-xl bg-white border border-gray-200 hover:border-primary-300 transition-all shadow-sm hover:shadow-lg cursor-pointer"
                  onClick={() => setSelectedFeature(f.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelectedFeature(f.id);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-label={`Explore ${f.title}`}
                >
                  <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${f.color} mb-4 shadow-md`}>
                    <Icon className="text-white" size={24} />
                  </div>
                  <h3 className="font-bold text-xl mb-2">{f.title}</h3>
                  <p className="mt-2 text-gray-600 text-sm leading-relaxed">{f.desc}</p>
                  <div className="mt-4 flex items-center gap-2 text-sm text-primary-600 font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                    Explore <ArrowRight className="w-4 h-4" />
                  </div>
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-primary-500/0 to-blue-500/0 group-hover:from-primary-500/5 group-hover:to-blue-500/5 transition-all -z-10" />
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {selectedFeature && (
        <MockupModal
          featureId={selectedFeature}
          feature={featureList.find((f) => f.id === selectedFeature)}
          onClose={() => setSelectedFeature(null)}
        />
      )}
    </>
  );
}






"use client";

import { motion } from "framer-motion";
import { Download, Brain, Shield, Lock, ArrowRight } from "lucide-react";
import Link from "next/link";

const steps = [
  {
    id: 1,
    title: "Data Input",
    text: "Flexible ingestion: EMR, wearables, CSV, API",
    icon: Download,
    color: "from-blue-500 to-cyan-500",
  },
  {
    id: 2,
    title: "AI Processing",
    text: "Auto-structuring, validation & enrichment",
    icon: Brain,
    color: "from-purple-500 to-pink-500",
  },
  {
    id: 3,
    title: "Anonymization",
    text: "Synthetic IDs & k-anonymity pipelines",
    icon: Shield,
    color: "from-green-500 to-emerald-500",
  },
  {
    id: 4,
    title: "Secure Storage",
    text: "Encrypted, auditable stores with role-based access",
    icon: Lock,
    color: "from-primary-500 to-blue-500",
  },
];

export default function FlowSteps() {
  return (
    <section id="how-it-works" className="py-16 lg:py-24 bg-white">
      <div className="container mx-auto px-6 text-center">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-3xl lg:text-4xl font-semibold mb-4"
        >
          How it works — from input to research-ready data
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="text-gray-600 mt-2 text-lg max-w-2xl mx-auto"
        >
          Four simple stages that power enterprise deployments.
        </motion.p>

        <div className="mt-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
          {steps.map((s, i) => {
            const Icon = s.icon;
            return (
              <motion.div
                key={s.id}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.12, duration: 0.6 }}
                whileHover={{ y: -8, scale: 1.02 }}
                className="group relative p-6 lg:p-8 bg-gradient-to-br from-gray-50 to-white rounded-xl border border-gray-200 hover:border-primary-300 transition-all shadow-sm hover:shadow-lg"
              >
                <div className={`inline-flex p-4 rounded-xl bg-gradient-to-br ${s.color} mb-4 shadow-lg`}>
                  <Icon className="text-white" size={28} />
                </div>
                <h3 className="mt-4 font-bold text-xl mb-2">{s.title}</h3>
                <p className="mt-2 text-sm text-gray-600 leading-relaxed">{s.text}</p>
                <Link
                  href={`/docs#step-${s.id}`}
                  className="mt-4 inline-flex items-center gap-2 text-sm text-primary-600 hover:text-primary-700 font-medium opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  Learn more <ArrowRight className="w-4 h-4" />
                </Link>
              </motion.div>
            );
          })}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="mt-12"
        >
          <Link
            href="/features#demo"
            className="inline-flex items-center gap-2 px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-semibold shadow-lg"
          >
            See demo <ArrowRight className="w-5 h-5" />
          </Link>
        </motion.div>
      </div>
    </section>
  );
}






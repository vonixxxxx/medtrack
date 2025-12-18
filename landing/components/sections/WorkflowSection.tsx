"use client";

import { WorkflowStep } from "@/components/ui/WorkflowStep";
import { Download, Brain, Shield, Lock } from "lucide-react";

export function WorkflowSection() {
  return (
    <section className="py-20 md:py-32 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-4">
            How MedTrack Fixes It
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            A streamlined workflow from data input to research-ready insights.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-4 relative">
          <WorkflowStep
            step={1}
            title="Data Input"
            description="EMR, wearables, CSV, API — flexible ingestion from any source"
            icon={<Download className="w-8 h-8" />}
            delay={0}
            isLast={false}
          />
          <WorkflowStep
            step={2}
            title="AI Processing"
            description="Structuring, validation, enrichment — intelligent data transformation"
            icon={<Brain className="w-8 h-8" />}
            delay={0.2}
            isLast={false}
          />
          <WorkflowStep
            step={3}
            title="Anonymization"
            description="K-anonymity pipelines, synthetic IDs — research-ready privacy"
            icon={<Shield className="w-8 h-8" />}
            delay={0.4}
            isLast={false}
          />
          <WorkflowStep
            step={4}
            title="Secure Storage"
            description="Encrypted, auditable, role-based access — enterprise-grade security"
            icon={<Lock className="w-8 h-8" />}
            delay={0.6}
            isLast={true}
          />
        </div>
      </div>
    </section>
  );
}




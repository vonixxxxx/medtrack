"use client";

import { InfoCard } from "@/components/ui/InfoCard";
import { Target, Heart, Zap } from "lucide-react";

export function PurposeMissionSection() {
  return (
    <section className="py-20 md:py-32 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-4">
            Purpose & Mission
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Bridging patient self-management, clinical care, and medical research through intelligent data management.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          <InfoCard
            title="Enterprise-Grade Intelligence"
            description="MedTrack provides enterprise-grade medication and clinical intelligence that transforms how healthcare institutions manage patient data, monitor adherence, and conduct research."
            icon={<Target className="w-8 h-8" />}
            delay={0}
            variant="highlight"
          />
          <InfoCard
            title="Bridge Three Worlds"
            description="We connect patient self-management tools, clinical care workflows, and medical research pipelines into one unified, privacy-first platform."
            icon={<Heart className="w-8 h-8" />}
            delay={0.2}
          />
          <InfoCard
            title="Solve Real Problems"
            description="Address fragmented adherence data, poor patient compliance rates, and research bottlenecks that slow medical innovation."
            icon={<Zap className="w-8 h-8" />}
            delay={0.4}
          />
        </div>
      </div>
    </section>
  );
}




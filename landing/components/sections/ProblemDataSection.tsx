"use client";

import { StatsCounter } from "@/components/ui/StatsCounter";
import { TrendingDown, Hospital, Clock, FileX } from "lucide-react";

export function ProblemDataSection() {
  return (
    <section className="py-20 md:py-32 bg-gradient-to-b from-gray-50 to-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-4">
            The Healthcare Data Problem
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Real-world statistics showing why MedTrack is needed.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12">
          <StatsCounter
            value={50}
            suffix="%"
            label="Average medication adherence rate"
            icon={<TrendingDown className="w-8 h-8" />}
            delay={0}
          />
          <StatsCounter
            value={20}
            suffix="%"
            label="Hospital readmissions due to non-adherence"
            icon={<Hospital className="w-8 h-8" />}
            delay={0.2}
          />
          <StatsCounter
            value={6}
            suffix=" months"
            label="Average time to prepare research datasets"
            icon={<Clock className="w-8 h-8" />}
            delay={0.4}
          />
          <StatsCounter
            value={80}
            suffix="%"
            label="Healthcare data that remains unstructured"
            icon={<FileX className="w-8 h-8" />}
            delay={0.6}
          />
        </div>

        <div className="mt-16 text-center">
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            These challenges cost healthcare systems billions annually and delay critical research. 
            MedTrack addresses them with AI-powered intelligence and enterprise-grade infrastructure.
          </p>
        </div>
      </div>
    </section>
  );
}




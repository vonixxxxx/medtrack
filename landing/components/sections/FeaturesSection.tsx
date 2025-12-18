"use client";

import { FeatureCard } from "@/components/ui/FeatureCard";
import { 
  TrendingUp, 
  Brain, 
  Camera, 
  Users, 
  BarChart3, 
  Database 
} from "lucide-react";

const features = [
  {
    title: "Advanced Adherence Engine",
    description: "AI-powered medication adherence tracking with predictive analytics and personalized reminders. Real-time scoring and pattern recognition help improve patient compliance.",
    icon: <TrendingUp className="w-10 h-10" />,
    imageSrc: "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=800&h=600&fit=crop",
    imageAlt: "Medication adherence dashboard",
    delay: 0,
  },
  {
    title: "AI Insights & Predictive Analytics",
    description: "Machine learning models analyze health patterns, predict outcomes, and provide actionable recommendations for clinicians and researchers.",
    icon: <Brain className="w-10 h-10" />,
    imageSrc: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=600&fit=crop",
    imageAlt: "AI analytics visualization",
    delay: 0.1,
  },
  {
    title: "Pill Recognition System",
    description: "Computer vision technology identifies medications from photos, reducing errors and improving patient safety. Supports thousands of medication types.",
    icon: <Camera className="w-10 h-10" />,
    imageSrc: "https://images.unsplash.com/photo-1587854692152-cbe660dbde88?w=800&h=600&fit=crop",
    imageAlt: "Pill recognition interface",
    delay: 0.2,
  },
  {
    title: "Clinician Dashboard",
    description: "Comprehensive patient management interface with real-time data, AI-assisted note parsing, MES calculator, and population health analytics.",
    icon: <Users className="w-10 h-10" />,
    imageSrc: "https://images.unsplash.com/photo-1576091160550-2173dba999e8?w=800&h=600&fit=crop",
    imageAlt: "Clinician dashboard",
    delay: 0.3,
  },
  {
    title: "Population Health Analytics",
    description: "Institution-level insights and dashboards for healthcare administrators. Track outcomes, identify trends, and optimize care delivery at scale.",
    icon: <BarChart3 className="w-10 h-10" />,
    imageSrc: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=600&fit=crop",
    imageAlt: "Population health dashboard",
    delay: 0.4,
  },
  {
    title: "Anonymized Data Exports",
    description: "Research-ready datasets with k-anonymity compliance. Export to CSV, JSON, or research-standard formats. Perfect for clinical trials and studies.",
    icon: <Database className="w-10 h-10" />,
    imageSrc: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=600&fit=crop",
    imageAlt: "Data export interface",
    delay: 0.5,
  },
];

export function FeaturesSection() {
  return (
    <section id="features" className="py-20 md:py-32 bg-gradient-to-b from-white to-gray-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-4">
            Powerful Features
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Enterprise-grade capabilities designed for healthcare institutions, clinicians, and researchers.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <FeatureCard
              key={index}
              title={feature.title}
              description={feature.description}
              imageSrc={feature.imageSrc}
              imageAlt={feature.imageAlt}
              icon={feature.icon}
              delay={feature.delay}
            />
          ))}
        </div>
      </div>
    </section>
  );
}




import Section from "../ui/Section";
import SectionHeader from "../ui/SectionHeader";
import FadeIn from "../ui/FadeIn";
import { Download, Brain, Shield, Lock, ArrowRight } from "lucide-react";

const steps = [
  {
    id: 1,
    title: "Data Input",
    description: "EMR, wearables, CSV, API",
    icon: Download,
    color: "blue"
  },
  {
    id: 2,
    title: "AI Processing",
    description: "Structuring, validation, enrichment",
    icon: Brain,
    color: "purple"
  },
  {
    id: 3,
    title: "Anonymization",
    description: "K-anonymity pipelines, synthetic IDs",
    icon: Shield,
    color: "green"
  },
  {
    id: 4,
    title: "Secure Storage",
    description: "Encrypted, auditable, role-based access",
    icon: Lock,
    color: "blue"
  },
];

const colorClasses = {
  blue: "bg-blue-600 text-white",
  purple: "bg-purple-600 text-white",
  green: "bg-green-600 text-white"
};

export default function LandingFlowSteps() {
  return (
    <Section 
      id="how-it-works" 
      background="gray" 
      padding="py-20 lg:py-28"
      divider={true}
      dividerText="Workflow"
    >
      <SectionHeader
        title="How It Works"
        subtitle="From data input to research-ready insights"
        align="center"
      />

      {/* Visual Timeline */}
      <div className="relative max-w-5xl mx-auto">
        {/* Connection Line (Desktop) */}
        <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-0.5 bg-gray-300 transform -translate-y-1/2" />
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 relative">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <FadeIn key={step.id} delay={index * 0.15}>
                <div className="relative">
                  {/* Step Number Circle */}
                  <div className={`w-16 h-16 ${colorClasses[step.color]} rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg relative z-10`}>
                    <Icon className="w-8 h-8" />
                  </div>
                  
                  {/* Step Content */}
                  <div className="text-center">
                    <h3 className="text-xl font-bold text-gray-900 mb-2">
                      {step.title}
                    </h3>
                    <p className="text-gray-600 text-sm">
                      {step.description}
                    </p>
                  </div>

                  {/* Arrow (Mobile/Tablet) */}
                  {index < steps.length - 1 && (
                    <div className="lg:hidden absolute top-8 left-full w-8 h-0.5 bg-gray-300 transform translate-x-4">
                      <ArrowRight className="w-4 h-4 text-gray-400 absolute -right-2 -top-1.5" />
                    </div>
                  )}
                </div>
              </FadeIn>
            );
          })}
        </div>
      </div>
    </Section>
  );
}

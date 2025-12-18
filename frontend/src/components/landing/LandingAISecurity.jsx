import Section from "../ui/Section";
import SectionHeader from "../ui/SectionHeader";
import Card from "../ui/card";
import FadeIn from "../ui/FadeIn";
import { Shield, Lock, Brain, Eye } from "lucide-react";

const features = [
  {
    icon: Shield,
    title: "HIPAA & GDPR Compliant",
    description: "Full regulatory compliance with healthcare data protection standards"
  },
  {
    icon: Lock,
    title: "End-to-End Encryption",
    description: "Bank-level encryption for data at rest and in transit"
  },
  {
    icon: Brain,
    title: "Local AI Option",
    description: "On-premise LLaMA inference for complete data residency"
  },
  {
    icon: Eye,
    title: "Data Anonymization",
    description: "K-anonymity pipelines and synthetic ID generation for research"
  }
];

export default function LandingAISecurity() {
  return (
    <Section 
      id="security" 
      background="gray" 
      padding="py-20 lg:py-28"
      divider={true}
      dividerText="Security"
    >
      <SectionHeader
        title="Intelligent AI. Ironclad Security."
        subtitle="Enterprise-grade security meets cutting-edge AI"
        align="center"
      />

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
        {features.map((feature, index) => {
          const Icon = feature.icon;
          return (
            <FadeIn key={index} delay={index * 0.1}>
              <Card hover={false} className="text-center">
                <div className="inline-flex p-4 bg-blue-100 rounded-xl mb-4">
                  <Icon className="w-8 h-8 text-blue-600" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  {feature.description}
                </p>
              </Card>
            </FadeIn>
          );
        })}
      </div>
    </Section>
  );
}

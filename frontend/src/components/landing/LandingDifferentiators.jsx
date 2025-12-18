import Section from "../ui/Section";
import SectionHeader from "../ui/SectionHeader";
import Card from "../ui/card";
import FadeIn from "../ui/FadeIn";
import { Shield, Brain, Database, Building2 } from "lucide-react";

const differentiators = [
  {
    icon: Shield,
    title: "Privacy-First Architecture",
    description: "HIPAA & GDPR compliant with local AI option for complete data residency",
    color: "text-blue-600",
    bgColor: "bg-blue-100"
  },
  {
    icon: Brain,
    title: "AI-Powered Intelligence",
    description: "Medication validation, health insights, and intelligent medical text parsing",
    color: "text-purple-600",
    bgColor: "bg-purple-100"
  },
  {
    icon: Database,
    title: "Research-Ready",
    description: "Anonymized, structured data exports with k-anonymity pipelines",
    color: "text-green-600",
    bgColor: "bg-green-100"
  },
  {
    icon: Building2,
    title: "Enterprise-Grade",
    description: "SSO/SAML, cloud & on-premise deployment, role-based access control",
    color: "text-orange-600",
    bgColor: "bg-orange-100"
  }
];

export default function LandingDifferentiators() {
  return (
    <Section 
      background="white" 
      padding="py-20 lg:py-28"
      divider={true}
      dividerText="Core Differentiators"
    >
      <SectionHeader
        title="Why MedTrack"
        subtitle="Enterprise-grade capabilities that set us apart"
        align="center"
      />

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
        {differentiators.map((item, index) => {
          const Icon = item.icon;
          return (
            <FadeIn key={index} delay={index * 0.1}>
              <Card hover={false} className="text-center">
                <div className={`inline-flex p-4 ${item.bgColor} rounded-xl mb-4`}>
                  <Icon className={`w-8 h-8 ${item.color}`} />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">
                  {item.title}
                </h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  {item.description}
                </p>
              </Card>
            </FadeIn>
          );
        })}
      </div>
    </Section>
  );
}




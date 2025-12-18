import Section from "../ui/Section";
import SectionHeader from "../ui/SectionHeader";
import Card from "../ui/card";
import AnimatedCounter from "../ui/AnimatedCounter";
import { Users, Stethoscope, Cpu } from "lucide-react";

const items = [
  {
    label: "Patient features",
    value: 25,
    suffix: "+",
    icon: Users,
    description: "Continuously expanding",
  },
  {
    label: "Clinician features",
    value: 17,
    suffix: "+",
    icon: Stethoscope,
    description: "Enterprise-ready",
  },
  {
    label: "Core engines",
    value: 5,
    suffix: "",
    icon: Cpu,
    description: "AI-powered",
  },
];

export default function LandingPlatformHighlights() {
  return (
    <Section 
      background="gray-to-white" 
      padding="py-12 lg:py-16"
      divider={true}
      dividerText="Platform Value"
    >
      <SectionHeader
        title="Why MedTrack is Different"
        subtitle="Quantifiable depth across users and workflows"
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
        {items.map((it, idx) => {
          const Icon = it.icon;
          return (
            <Card key={idx} delay={idx * 0.1} hover={false}>
              <div className="flex items-center gap-4 mb-4">
                <div className="p-3 bg-blue-100 rounded-lg">
                  <Icon className="w-6 h-6 text-blue-600" />
                </div>
                <div className="flex-1">
                  <div className="text-3xl lg:text-4xl font-bold text-gray-900 mb-1">
                    <AnimatedCounter value={it.value} suffix={it.suffix} />
                  </div>
                  <div className="text-sm font-medium text-gray-700">{it.label}</div>
                </div>
              </div>
              {it.description && (
                <p className="text-xs text-gray-500 mt-2">{it.description}</p>
              )}
            </Card>
          );
        })}
      </div>
    </Section>
  );
}

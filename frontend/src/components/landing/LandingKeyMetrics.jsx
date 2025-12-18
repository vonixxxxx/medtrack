import Section from "../ui/Section";
import FadeIn from "../ui/FadeIn";
import { FlaskConical, Users, Rocket, Globe } from "lucide-react";

const metrics = [
  {
    icon: FlaskConical,
    value: "12+",
    label: "Clinical trials",
    color: "text-blue-600"
  },
  {
    icon: Users,
    value: "8+",
    label: "Research groups",
    color: "text-green-600"
  },
  {
    icon: Rocket,
    value: "15+",
    label: "Pilot deployments",
    color: "text-purple-600"
  },
  {
    icon: Globe,
    value: "Worldwide",
    label: "Trusted by healthcare professionals",
    color: "text-orange-600"
  }
];

export default function LandingKeyMetrics() {
  return (
    <Section 
      background="gray" 
      padding="py-16 lg:py-20"
    >
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <FadeIn key={index} delay={index * 0.1}>
              <div className="text-center">
                <div className="inline-flex p-4 bg-white rounded-xl shadow-sm mb-4">
                  <Icon className={`w-8 h-8 ${metric.color}`} />
                </div>
                <div className="text-4xl lg:text-5xl font-bold text-gray-900 mb-2">
                  {metric.value}
                </div>
                <div className="text-sm lg:text-base text-gray-600 font-medium">
                  {metric.label}
                </div>
              </div>
            </FadeIn>
          );
        })}
      </div>
    </Section>
  );
}




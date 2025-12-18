import { useState } from "react";
import Section from "../ui/Section";
import SectionHeader from "../ui/SectionHeader";
import Card from "../ui/card";
import FadeIn from "../ui/FadeIn";
import { Pill, TrendingUp, UserCheck, Brain, Database, Building2, FlaskConical, Shield, ArrowRight } from "lucide-react";

const segments = [
  {
    id: "patients",
    label: "Patients",
    icon: UserCheck,
    features: [
      {
        title: "Medication Timelines",
        description: "Visual history and quick mark-as-taken",
        icon: Pill
      },
      {
        title: "Pill Recognition",
        description: "AI-powered identification from photos",
        icon: Brain
      },
      {
        title: "Adherence Scoring",
        description: "Track streaks and patterns",
        icon: TrendingUp
      }
    ]
  },
  {
    id: "clinicians",
    label: "Clinicians",
    icon: UserCheck,
    features: [
      {
        title: "AI-Assisted Notes Parsing",
        description: "Extract structured data from medical notes",
        icon: Brain
      },
      {
        title: "MES Calculator",
        description: "HbA1c adjustment with medication effects",
        icon: TrendingUp
      },
      {
        title: "Patient Analytics",
        description: "Population health insights and visualizations",
        icon: Database
      }
    ]
  },
  {
    id: "enterprises",
    label: "Enterprises",
    icon: Building2,
    features: [
      {
        title: "Population Health Dashboards",
        description: "Institution-level analytics and insights",
        icon: Database
      },
      {
        title: "Secure Data Exports",
        description: "Compliance-ready data export pipelines",
        icon: Shield
      },
      {
        title: "Enterprise Deployment",
        description: "Cloud, private cloud, or on-premise",
        icon: Building2
      }
    ]
  },
  {
    id: "researchers",
    label: "Researchers",
    icon: FlaskConical,
    features: [
      {
        title: "Anonymized Datasets",
        description: "K-anonymity compliant research data",
        icon: Database
      },
      {
        title: "Predictive Modeling",
        description: "AI-powered health outcome predictions",
        icon: Brain
      },
      {
        title: "Export-Ready Formats",
        description: "CSV, JSON, and research-standard formats",
        icon: Database
      }
    ]
  }
];

export default function LandingFeaturesBySegment() {
  const [activeSegment, setActiveSegment] = useState("patients");

  const currentSegment = segments.find(s => s.id === activeSegment) || segments[0];
  const SegmentIcon = currentSegment.icon;

  return (
    <Section 
      id="features" 
      background="white-to-gray" 
      padding="py-20 lg:py-28"
      divider={true}
      dividerText="Features"
    >
      <SectionHeader
        title="Built for Every Healthcare Role"
        subtitle="Tailored features for patients, clinicians, enterprises, and researchers"
        align="center"
      />

      {/* Segment Tabs */}
      <div className="flex flex-wrap justify-center gap-3 mb-12">
        {segments.map((segment) => {
          const Icon = segment.icon;
          return (
            <button
              key={segment.id}
              onClick={() => setActiveSegment(segment.id)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all flex items-center gap-2 ${
                activeSegment === segment.id
                  ? "bg-blue-600 text-white shadow-lg"
                  : "bg-white text-gray-700 border-2 border-gray-200 hover:border-blue-300"
              }`}
            >
              <Icon className="w-5 h-5" />
              {segment.label}
            </button>
          );
        })}
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
        {currentSegment.features.map((feature, index) => {
          const FeatureIcon = feature.icon;
          return (
            <FadeIn key={index} delay={index * 0.1}>
              <Card className="h-full">
                <div className="inline-flex p-3 bg-blue-100 rounded-lg mb-4">
                  <FeatureIcon className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 text-sm mb-4 leading-relaxed">
                  {feature.description}
                </p>
                <a
                  href={`/features#${currentSegment.id}-${feature.title.toLowerCase().replace(/\s+/g, '-')}`}
                  className="inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  Learn More <ArrowRight className="w-4 h-4" />
                </a>
              </Card>
            </FadeIn>
          );
        })}
      </div>
    </Section>
  );
}


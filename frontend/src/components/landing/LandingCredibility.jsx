import Section from "../ui/Section";
import SectionHeader from "../ui/SectionHeader";
import Card from "../ui/card";
import FadeIn from "../ui/FadeIn";
import { Quote, CheckCircle2 } from "lucide-react";

const testimonials = [
  {
    quote: "MedTrack's adherence engine has transformed how we monitor patient compliance. The insights are actionable and the interface is intuitive.",
    author: "Dr. Sarah Chen",
    role: "Clinical Director, Metro Health",
    type: "clinician"
  },
  {
    quote: "As a patient managing multiple medications, MedTrack keeps me organized and gives me peace of mind. The reminders are never intrusive.",
    author: "Michael Rodriguez",
    role: "Patient, 2+ years",
    type: "patient"
  },
  {
    quote: "The anonymization pipeline makes our research data collection seamless. We can focus on analysis instead of data preparation.",
    author: "Dr. James Park",
    role: "Research Lead, University Medical Center",
    type: "researcher"
  }
];

const metrics = [
  { label: "Clinical trials", value: "12+" },
  { label: "Research groups", value: "8+" },
  { label: "Pilot deployments", value: "15+" }
];

export default function LandingCredibility() {
  return (
    <Section 
      id="credibility" 
      background="gray" 
      padding="py-16 lg:py-20"
      divider={true}
      dividerText="Trusted by healthcare professionals"
    >
      {/* Metrics Row */}
      <FadeIn className="mb-12">
        <div className="flex flex-wrap justify-center gap-8 lg:gap-12">
          {metrics.map((metric, index) => (
            <div key={index} className="text-center">
              <div className="text-2xl lg:text-3xl font-bold text-gray-900 mb-1">
                {metric.value}
              </div>
              <div className="text-sm text-gray-600">{metric.label}</div>
            </div>
          ))}
        </div>
      </FadeIn>

      {/* Testimonials */}
      <SectionHeader
        title="What healthcare professionals say"
        subtitle="Real feedback from clinicians, patients, and researchers using MedTrack"
      />

      <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
        {testimonials.map((testimonial, index) => (
          <FadeIn key={index} delay={index * 0.1}>
            <Card className="flex flex-col h-full">
              <Quote className="w-8 h-8 text-blue-600 mb-4" />
              <p className="text-gray-700 mb-6 flex-grow leading-relaxed">
                "{testimonial.quote}"
              </p>
              <div className="border-t border-gray-200 pt-4">
                <div className="font-semibold text-gray-900">{testimonial.author}</div>
                <div className="text-sm text-gray-600">{testimonial.role}</div>
              </div>
            </Card>
          </FadeIn>
        ))}
      </div>

      {/* Case Snippet */}
      <FadeIn delay={0.3} className="mt-12">
        <Card className="bg-gradient-to-br from-blue-50 to-white border-blue-200">
          <div className="flex items-start gap-4">
            <CheckCircle2 className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">
                Used in clinical research and pilot programs
              </h3>
              <p className="text-sm text-gray-600">
                MedTrack is actively deployed in clinical trials, research studies, and healthcare 
                institution pilot programs. Our platform enables data-driven healthcare decisions 
                while maintaining the highest standards of privacy and compliance.
              </p>
            </div>
          </div>
        </Card>
      </FadeIn>
    </Section>
  );
}


import Section from "../ui/Section";
import SectionHeader from "../ui/SectionHeader";
import Card from "../ui/card";
import FadeIn from "../ui/FadeIn";
import { Quote } from "lucide-react";

const testimonials = [
  {
    quote: "MedTrack transformed how we monitor patient adherence. Insights are actionable, interface intuitive.",
    author: "Dr. Sarah Chen",
    title: "Clinical Director",
    organization: "Metro Health",
    photo: null // Placeholder for photo
  },
  {
    quote: "The anonymization pipeline makes our research data collection seamless. We can focus on analysis instead of data preparation.",
    author: "Dr. James Park",
    title: "Research Lead",
    organization: "University Medical Center",
    photo: null
  }
];

export default function LandingTestimonials() {
  return (
    <Section 
      background="white" 
      padding="py-20 lg:py-28"
      divider={true}
      dividerText="Testimonials"
    >
      <SectionHeader
        title="Trusted by Healthcare Leaders"
        subtitle="See what professionals say about MedTrack"
        align="center"
      />

      <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
        {testimonials.map((testimonial, index) => (
          <FadeIn key={index} delay={index * 0.15}>
            <Card className="h-full">
              <Quote className="w-8 h-8 text-blue-600 mb-4" />
              <p className="text-lg text-gray-700 mb-6 leading-relaxed italic">
                "{testimonial.quote}"
              </p>
              <div className="flex items-center gap-4 pt-4 border-t border-gray-200">
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-blue-600 font-semibold text-lg">
                    {testimonial.author.split(' ').map(n => n[0]).join('')}
                  </span>
                </div>
                <div>
                  <div className="font-semibold text-gray-900">
                    {testimonial.author}
                  </div>
                  <div className="text-sm text-gray-600">
                    {testimonial.title}, {testimonial.organization}
                  </div>
                </div>
              </div>
            </Card>
          </FadeIn>
        ))}
      </div>
    </Section>
  );
}




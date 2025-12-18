"use client";

import Header from "@/components/Header";
import HeroWithShader from "@/components/HeroWithShader";
import Features from "@/components/Features";
import AISecurity from "@/components/AISecurity";
import FlowSteps from "@/components/FlowSteps";
import CTA from "@/components/CTA";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen">
      <Header />
      <HeroWithShader />
      <Features />
      <AISecurity />
      <FlowSteps />
      <CTA />
      <Footer />
    </main>
  );
}

"use client";

import Header from "@/components/Header";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import Collaboration from "@/components/Collaboration";
import AISecurity from "@/components/AISecurity";
import CTA from "@/components/CTA";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen">
      <Header />
      <Hero />
      <Features />
      <Collaboration />
      <AISecurity />
      <CTA />
      <Footer />
    </main>
  );
}


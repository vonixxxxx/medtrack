import React from 'react';
import LandingHeader from '../components/landing/LandingHeader.jsx';
import LandingHero from '../components/landing/LandingHero.jsx';
import LandingFeatures from '../components/landing/LandingFeatures.jsx';
import LandingCollaboration from '../components/landing/LandingCollaboration.jsx';
import LandingAISecurity from '../components/landing/LandingAISecurity.jsx';
import LandingCTA from '../components/landing/LandingCTA.jsx';
import LandingFooter from '../components/landing/LandingFooter.jsx';

export default function LandingPage() {
  return (
    <main className="min-h-screen">
      <LandingHeader />
      <LandingHero />
      <LandingFeatures />
      <LandingCollaboration />
      <LandingAISecurity />
      <LandingCTA />
      <LandingFooter />
    </main>
  );
}



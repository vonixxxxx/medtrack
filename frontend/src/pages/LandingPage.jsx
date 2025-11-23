import React from 'react';
import LandingHeader from '../components/landing/LandingHeader';
import LandingFeatures from '../components/landing/LandingFeatures';
import LandingCollaboration from '../components/landing/LandingCollaboration';
import LandingAISecurity from '../components/landing/LandingAISecurity';
import LandingCTA from '../components/landing/LandingCTA';
import LandingFooter from '../components/landing/LandingFooter';

export default function LandingPage() {
  return (
    <main className="min-h-screen">
      <LandingHeader />
      <LandingFeatures />
      <LandingCollaboration />
      <LandingAISecurity />
      <LandingCTA />
      <LandingFooter />
    </main>
  );
}



#!/usr/bin/env node

/**
 * Generate SVG placeholder images for mockups
 * Usage: node scripts/generate-mockup-placeholder.js
 */

const fs = require('fs');
const path = require('path');

const mockupsDir = path.join(__dirname, '../public/mockups');

// Ensure directory exists
if (!fs.existsSync(mockupsDir)) {
  fs.mkdirSync(mockupsDir, { recursive: true });
}

const mockups = [
  { name: 'dashboard-1', title: 'Patient Timeline', width: 1920, height: 1080 },
  { name: 'dashboard-2', title: 'Adherence Analytics', width: 1920, height: 1080 },
  { name: 'dashboard-3', title: 'Clinician Workspace', width: 1920, height: 1080 },
  { name: 'dashboard-4', title: 'AI Insights', width: 1920, height: 1080 },
  { name: 'feature-med', title: 'Medication Management', width: 1920, height: 1080 },
  { name: 'feature-adhr', title: 'Adherence Engine', width: 1920, height: 1080 },
  { name: 'feature-clin', title: 'Clinician Workspace', width: 1920, height: 1080 },
  { name: 'feature-ai', title: 'AI Insights', width: 1920, height: 1080 },
  { name: 'feature-data', title: 'Data & Research', width: 1920, height: 1080 },
  { name: 'feature-deploy', title: 'Integrations', width: 1920, height: 1080 },
];

function generateSVG(name, title, width, height) {
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1e293b;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0f172a;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)"/>
  <rect x="40" y="40" width="${width - 80}" height="${height - 80}" rx="16" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.1)" stroke-width="2"/>
  
  <!-- Header bar -->
  <rect x="60" y="60" width="${width - 120}" height="60" rx="8" fill="rgba(255,255,255,0.08)"/>
  <circle cx="${width - 100}" cy="90" r="8" fill="rgba(255,255,255,0.2)"/>
  <circle cx="${width - 70}" cy="90" r="8" fill="rgba(255,255,255,0.2)"/>
  <circle cx="${width - 40}" cy="90" r="8" fill="rgba(255,255,255,0.2)"/>
  
  <!-- Content area -->
  <rect x="80" y="140" width="${(width - 200) / 3}" height="${height - 200}" rx="8" fill="rgba(255,255,255,0.03)"/>
  <rect x="${80 + (width - 200) / 3 + 20}" y="140" width="${(width - 200) / 3}" height="${height - 200}" rx="8" fill="rgba(255,255,255,0.03)"/>
  <rect x="${80 + 2 * ((width - 200) / 3 + 20)}" y="140" width="${(width - 200) / 3}" height="${height - 200}" rx="8" fill="rgba(255,255,255,0.03)"/>
  
  <!-- Title text -->
  <text x="${width / 2}" y="${height / 2}" font-family="system-ui, sans-serif" font-size="32" font-weight="600" fill="rgba(255,255,255,0.3)" text-anchor="middle" dominant-baseline="middle">${title}</text>
  <text x="${width / 2}" y="${height / 2 + 40}" font-family="system-ui, sans-serif" font-size="18" fill="rgba(255,255,255,0.2)" text-anchor="middle" dominant-baseline="middle">Mockup placeholder - Replace with actual screenshot</text>
</svg>`;

  return svg;
}

console.log('Generating mockup placeholders...');

mockups.forEach((mockup) => {
  const svg = generateSVG(mockup.name, mockup.title, mockup.width, mockup.height);
  const filePath = path.join(mockupsDir, `${mockup.name}.svg`);
  fs.writeFileSync(filePath, svg);
  console.log(`✓ Generated ${mockup.name}.svg`);
});

console.log(`\n✓ Generated ${mockups.length} placeholder images in ${mockupsDir}`);
console.log('\nNext steps:');
console.log('1. Replace SVG placeholders with actual dashboard screenshots (PNG/WebP)');
console.log('2. Recommended image sizes: 1920x1080px or higher');
console.log('3. Optimize images for web (use tools like ImageOptim or Squoosh)');
console.log('4. Consider creating WebP versions for better performance');






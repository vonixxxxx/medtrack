#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('🔍 Verifying Landing Page Update...\n');

// Check page.tsx
const pagePath = path.join(__dirname, 'app', 'page.tsx');
const pageContent = fs.readFileSync(pagePath, 'utf8');

console.log('✅ app/page.tsx:');
console.log('   - Uses HeroWithMockups:', pageContent.includes('HeroWithMockups') ? '✅ YES' : '❌ NO');
console.log('   - Uses PlatformHighlights:', pageContent.includes('PlatformHighlights') ? '✅ YES' : '❌ NO');
console.log('   - Uses FeaturesGrid:', pageContent.includes('FeaturesGrid') ? '✅ YES' : '❌ NO');
console.log('   - Uses FlowSteps:', pageContent.includes('FlowSteps') ? '✅ YES' : '❌ NO');
console.log('   - Old Hero component:', pageContent.includes('from "@/components/Hero"') ? '❌ FOUND (BAD)' : '✅ NOT FOUND (GOOD)');
console.log('   - Old Collaboration:', pageContent.includes('Collaboration') ? '❌ FOUND (BAD)' : '✅ NOT FOUND (GOOD)');

// Check HeroWithMockups component
const heroPath = path.join(__dirname, 'components', 'HeroWithMockups.tsx');
if (fs.existsSync(heroPath)) {
  const heroContent = fs.readFileSync(heroPath, 'utf8');
  console.log('\n✅ components/HeroWithMockups.tsx:');
  console.log('   - Has new headline:', heroContent.includes('Enterprise-grade medication') ? '✅ YES' : '❌ NO');
  console.log('   - Has carousel:', heroContent.includes('AnimatePresence') ? '✅ YES' : '❌ NO');
}

// Check if old components still exist (they can exist, just not be used)
const oldHeroPath = path.join(__dirname, 'components', 'Hero.tsx');
const oldCollabPath = path.join(__dirname, 'components', 'Collaboration.tsx');
console.log('\n📁 Old components (can exist, just not used):');
console.log('   - Hero.tsx exists:', fs.existsSync(oldHeroPath) ? '⚠️  YES (not used)' : '✅ NO');
console.log('   - Collaboration.tsx exists:', fs.existsSync(oldCollabPath) ? '⚠️  YES (not used)' : '✅ NO');

// Check mockups
const mockupsDir = path.join(__dirname, 'public', 'mockups');
console.log('\n🖼️  Mockups:');
if (fs.existsSync(mockupsDir)) {
  const mockups = fs.readdirSync(mockupsDir).filter(f => f.endsWith('.svg') || f.endsWith('.png'));
  console.log(`   - Found ${mockups.length} mockup files`);
  mockups.slice(0, 5).forEach(m => console.log(`     • ${m}`));
} else {
  console.log('   ❌ Mockups directory not found');
}

console.log('\n✨ Summary:');
if (pageContent.includes('HeroWithMockups') && !pageContent.includes('from "@/components/Hero"')) {
  console.log('✅ Landing page is correctly updated!');
  console.log('\n💡 If you still see old content:');
  console.log('   1. Stop the dev server (Ctrl+C)');
  console.log('   2. Run: rm -rf .next');
  console.log('   3. Run: npm run dev');
  console.log('   4. Hard refresh browser (Cmd+Shift+R / Ctrl+Shift+R)');
} else {
  console.log('❌ Landing page needs updating!');
}






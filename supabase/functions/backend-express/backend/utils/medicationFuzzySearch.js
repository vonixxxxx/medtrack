const Fuse = require('fuse.js');
const path = require('path');
const fs = require('fs');

// Load medication list
const medsFile = path.join(__dirname, 'medications.json');
const medications = JSON.parse(fs.readFileSync(medsFile, 'utf8'));

// Configure Fuse
const fuse = new Fuse(medications, {
  keys: ['name', 'generic', 'brand_names', 'synonyms'],
  threshold: 0.38, // tolerant, good for typos
  includeScore: true,
});

function fuzzyMedicationSearch(query, limit = 5) {
  if (!query || query.length < 2) return { matches: [], best: null };
  const results = fuse.search(query, { limit });
  if (!results.length) return { matches: [], best: null };
  // Best match always first
  return {
    matches: results.map(r => ({ ...r.item, _score: r.score })),
    best: { ...results[0].item, _score: results[0].score }
  };
}

module.exports = {
  fuzzyMedicationSearch,
  medications
};

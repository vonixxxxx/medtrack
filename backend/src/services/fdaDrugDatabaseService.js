const Fuse = require('fuse.js');
const path = require('path');
const fs = require('fs');
const { Transform } = require('stream');

// Path to the pre-processed medication index
const INDEX_PATH = path.join(__dirname, '../../data/fda_medication_index.json');

// Cache for processed drug database
let drugDatabaseCache = null;
let drugDatabaseIndex = null;
let isLoading = false;
let loadPromise = null;

// Common inactive ingredients to exclude
const INACTIVE_INGREDIENTS = new Set([
  'ANHYDROUS', 'LACTOSE', 'CROSCARMELLOSE', 'SODIUM', 'MAGNESIUM', 'STEARATE',
  'FERRIC', 'OXIDE', 'RED', 'YELLOW', 'WHITE', 'OFF', 'CELLULOSE', 'TALC',
  'TITANIUM', 'DIOXIDE', 'GELATIN', 'SHELLAC', 'WAX', 'POLYETHYLENE', 'GLYCOL',
  'HYDROCHLORIDE', 'HYDROCHLOROTHIAZIDE', 'SULFATE', 'PHOSPHATE', 'CITRATE'
]);

// Extract primary drug name from FDA label data (improved)
function extractDrugName(record) {
  // Method 1: Extract from spl_product_data_elements (most reliable)
  if (record.spl_product_data_elements && record.spl_product_data_elements.length > 0) {
    const text = record.spl_product_data_elements[0];
    
    // Clean and normalize
    const upperText = text.toUpperCase();
    
    // Pattern 1: "DRUGNAME and DRUGNAME Tablets" - extract first drug
    const comboMatch = upperText.match(/^([A-Z][A-Z\s]{2,30}?)(?:\s+AND\s+[A-Z][A-Z\s]+?)?(?:\s+TABLETS?|\s+CAPSULES?|\s+INJECTION|\s+SOLUTION|\s+ORAL|\s+TOPICAL)/);
    if (comboMatch) {
      let drugName = comboMatch[1].trim();
      const words = drugName.split(/\s+/).filter(w => {
        const upper = w.toUpperCase();
        return w.length >= 3 && w.length <= 25 && 
               !INACTIVE_INGREDIENTS.has(upper) &&
               /^[A-Z]+$/.test(upper);
      });
      if (words.length > 0) {
        const firstWord = words[0];
        return firstWord.charAt(0) + firstWord.slice(1).toLowerCase();
      }
    }
    
    // Pattern 2: First significant capitalized word sequence (up to 3 words)
    const words = upperText.split(/\s+/);
    for (let i = 0; i < Math.min(words.length, 5); i++) {
      const word = words[i].trim();
      if (word.length >= 4 && word.length <= 25 && 
          /^[A-Z]+$/.test(word) && 
          !INACTIVE_INGREDIENTS.has(word)) {
        return word.charAt(0) + word.slice(1).toLowerCase();
      }
    }
  }
  
  // Method 2: Extract from description field
  if (record.description && record.description.length > 0) {
    const desc = record.description[0];
    
    // Pattern: "DRUGNAME Tablets" or "DRUGNAME and DRUGNAME Tablets"
    const match = desc.match(/([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:and\s+[A-Z][a-z]+)?\s*(?:Tablets?|Capsules?|Injection|Solution|Oral|Topical)/);
    if (match) {
      return match[1].trim();
    }
    
    // Pattern: "DRUGNAME, a..." - drug name at start of description
    const startMatch = desc.match(/^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+(?:is|are|combines|contains)/i);
    if (startMatch) {
      return startMatch[1].trim();
    }
  }
  
  return null;
}

// Extract drug class from mechanism_of_action, description, or clinical_pharmacology
function extractDrugClass(record) {
  const text = [
    ...(record.mechanism_of_action || []),
    ...(record.description || []),
    ...(record.clinical_pharmacology || [])
  ].join(' ').toLowerCase();
  
  // Direct drug class patterns (more specific first)
  const classPatterns = [
    { pattern: /ace\s+inhibitor|angiotensin\s+converting\s+enzyme\s+inhibitor/i, class: 'ACE Inhibitor' },
    { pattern: /beta[\s-]?blocker|beta[\s-]?adrenergic/i, class: 'Beta Blocker' },
    { pattern: /calcium\s+channel\s+blocker/i, class: 'Calcium Channel Blocker' },
    { pattern: /glp[\s-]?1|glucagon[\s-]?like\s+peptide/i, class: 'GLP-1 Receptor Agonist' },
    { pattern: /biguanide/i, class: 'Biguanide' },
    { pattern: /hmg[\s-]?coa\s+reductase|statin/i, class: 'Statin' },
    { pattern: /selective\s+serotonin\s+reuptake|ssri/i, class: 'SSRI' },
    { pattern: /nonsteroidal\s+anti[\s-]?inflammatory|nsaid/i, class: 'NSAID' },
    { pattern: /proton\s+pump\s+inhibitor|ppi/i, class: 'Proton Pump Inhibitor' },
    { pattern: /benzodiazepine/i, class: 'Benzodiazepine' },
    { pattern: /opioid|narcotic/i, class: 'Opioid' },
    { pattern: /analgesic|antipyretic/i, class: 'Analgesic' },
    { pattern: /anticoagulant|blood\s+thinner/i, class: 'Anticoagulant' },
    { pattern: /antihistamine/i, class: 'Antihistamine' },
    { pattern: /antibiotic|antimicrobial/i, class: 'Antibiotic' },
    { pattern: /antidepressant/i, class: 'Antidepressant' },
    { pattern: /antipsychotic/i, class: 'Antipsychotic' },
    { pattern: /anticonvulsant|antiepileptic/i, class: 'Anticonvulsant' },
    { pattern: /antidiabetic/i, class: 'Antidiabetic' },
    { pattern: /sulfonylurea/i, class: 'Sulfonylurea' },
    { pattern: /thiazide/i, class: 'Thiazide Diuretic' },
    { pattern: /diuretic/i, class: 'Diuretic' },
    { pattern: /insulin/i, class: 'Insulin' },
    { pattern: /thyroid\s+hormone/i, class: 'Thyroid Hormone' }
  ];
  
  for (const { pattern, class: drugClass } of classPatterns) {
    if (pattern.test(text)) {
      return drugClass;
    }
  }
  
  return null;
}

// Extract typical strengths/dosages from FDA label
function extractDosages(record) {
  const dosages = new Set();
  const text = [
    ...(record.description || []),
    ...(record.dosage_and_administration || []),
    ...(record.how_supplied || [])
  ].join(' ');
  
  // Pattern: "Xmg", "X mg", "X units", "X%", etc.
  const dosagePatterns = [
    /\b(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|units?|iu|ml|l|%)\b/gi,
    /\b(\d+(?:\.\d+)?)\s*(?:milligrams?|micrograms?|grams?|units?)\b/gi
  ];
  
  for (const pattern of dosagePatterns) {
    const matches = text.matchAll(pattern);
    for (const match of matches) {
      const dosage = match[0].trim();
      if (dosage.length <= 20) { // Reasonable dosage length
        dosages.add(dosage);
      }
    }
  }
  
  return Array.from(dosages).slice(0, 10); // Return up to 10 unique dosages
}

// Extract dosage forms from FDA label
function extractDosageForms(record) {
  const forms = new Set();
  const text = [
    ...(record.description || []),
    ...(record.how_supplied || [])
  ].join(' ').toLowerCase();
  
  const formPatterns = [
    /\b(tablets?|capsules?|injections?|solutions?|suspensions?|creams?|ointments?|gels?|patches?|suppositories?|drops?|sprays?|inhalers?|powders?|liquids?)\b/gi
  ];
  
  for (const pattern of formPatterns) {
    const matches = text.matchAll(pattern);
    for (const match of matches) {
      const form = match[1].trim();
      if (form.length >= 3) {
        forms.add(form.charAt(0).toUpperCase() + form.slice(1));
      }
    }
  }
  
  return Array.from(forms).slice(0, 5); // Return up to 5 unique forms
}

// Process JSON file using streaming to handle large files
function processJSONFile(filePath) {
  return new Promise((resolve, reject) => {
    const drugs = new Map();
    let buffer = '';
    let depth = 0;
    let inResults = false;
    let currentRecord = '';
    let recordCount = 0;
    let processedCount = 0;
    
    const stream = fs.createReadStream(filePath, { encoding: 'utf8', highWaterMark: 64 * 1024 });
    
    stream.on('data', (chunk) => {
      buffer += chunk;
      
      // Process complete JSON objects from buffer
      let startIdx = 0;
      for (let i = 0; i < buffer.length; i++) {
        if (buffer[i] === '{') {
          if (depth === 0) startIdx = i;
          depth++;
        } else if (buffer[i] === '}') {
          depth--;
          if (depth === 0) {
            // Complete object found
            const objStr = buffer.substring(startIdx, i + 1);
            try {
              const obj = JSON.parse(objStr);
              if (obj.results && Array.isArray(obj.results)) {
                // Process results array
                for (const record of obj.results) {
                  recordCount++;
                  const drugName = extractDrugName(record);
                  if (drugName && drugName.length >= 3) {
                    const key = drugName.toLowerCase();
                    
                    if (!drugs.has(key)) {
                      const drugClass = extractDrugClass(record);
                      const dosages = extractDosages(record);
                      const dosageForms = extractDosageForms(record);
                      
                      drugs.set(key, {
                        generic_name: key,
                        display_name: drugName,
                        drug_class: drugClass || 'Unknown',
                        brand_names: [],
                        dosage_forms: dosageForms,
                        typical_strengths: dosages,
                        source: 'fda_database',
                        record_count: 1
                      });
                      processedCount++;
                    } else {
                      const existing = drugs.get(key);
                      existing.record_count++;
                      
                      // Merge information
                      const newForms = extractDosageForms(record);
                      newForms.forEach(form => {
                        if (!existing.dosage_forms.includes(form)) {
                          existing.dosage_forms.push(form);
                        }
                      });
                      
                      const newStrengths = extractDosages(record);
                      newStrengths.forEach(strength => {
                        if (!existing.typical_strengths.includes(strength)) {
                          existing.typical_strengths.push(strength);
                        }
                      });
                      
                      if (existing.drug_class === 'Unknown') {
                        const drugClass = extractDrugClass(record);
                        if (drugClass) {
                          existing.drug_class = drugClass;
                        }
                      }
                    }
                  }
                }
              }
            } catch (e) {
              // Skip malformed JSON
            }
            buffer = buffer.substring(i + 1);
            i = -1;
          }
        }
      }
    });
    
    stream.on('end', () => {
      resolve({ drugs: Array.from(drugs.values()), recordCount, processedCount });
    });
    
    stream.on('error', reject);
  });
}

// Load pre-processed FDA database index
function loadFDADatabase() {
  if (drugDatabaseCache) {
    return drugDatabaseCache;
  }
  
  if (isLoading && loadPromise) {
    return loadPromise;
  }
  
  isLoading = true;
  loadPromise = (async () => {
    try {
      console.log('🚀 Loading FDA medication database index...');
      const startTime = Date.now();
      
      if (!fs.existsSync(INDEX_PATH)) {
        console.warn(`⚠️  Index file not found: ${INDEX_PATH}`);
        console.log('💡 Run: python3 backend/scripts/process_fda_database.py to create the index');
        isLoading = false;
        return [];
      }
      
      const indexContent = fs.readFileSync(INDEX_PATH, 'utf8');
      drugDatabaseCache = JSON.parse(indexContent);
      
      const loadTime = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`✅ Loaded ${drugDatabaseCache.length.toLocaleString()} medications in ${loadTime}s`);
      
      // Create Fuse index for fast searching
      console.log('🔍 Creating search index...');
      const indexStartTime = Date.now();
      
      drugDatabaseIndex = new Fuse(drugDatabaseCache, {
        keys: [
          { name: 'generic_name', weight: 0.8 },
          { name: 'display_name', weight: 0.9 }
        ],
        threshold: 0.3, // Stricter matching for accuracy
        includeScore: true,
        minMatchCharLength: 3,
        ignoreLocation: true,
        findAllMatches: false
      });
      
      const indexTime = ((Date.now() - indexStartTime) / 1000).toFixed(2);
      console.log(`✓ Search index created in ${indexTime}s\n`);
      
      isLoading = false;
      return drugDatabaseCache;
    } catch (error) {
      isLoading = false;
      console.error('❌ Error loading FDA database:', error);
      throw error;
    }
  })();
  
  return loadPromise;
}

// Medication name aliases/mappings
const MEDICATION_ALIASES = {
  'paracetamol': 'acetaminophen',
  'tylenol': 'acetaminophen',
  'ozempic': 'semaglutide',
  'wegovy': 'semaglutide',
  'rybelsus': 'semaglutide',
  'glucophage': 'metformin',
  'lipitor': 'atorvastatin',
  'zestril': 'lisinopril',
  'prinivil': 'lisinopril',
  'xanax': 'alprazolam',
  'zoloft': 'sertraline',
  'prozac': 'fluoxetine',
  'lexapro': 'escitalopram'
};

// Search the FDA database for a medication
function searchFDADatabase(medicationName) {
  if (!drugDatabaseIndex) {
    // If not loaded, try to load synchronously (will be slow first time)
    if (!isLoading) {
      loadFDADatabase().catch(() => {});
    }
    return null;
  }
  
  const normalizedName = medicationName.toLowerCase().trim();
  
  // Check aliases first
  const aliasedName = MEDICATION_ALIASES[normalizedName] || normalizedName;
  
  // Try exact match first (with and without alias)
  const exactMatch = drugDatabaseCache.find(drug => {
    const drugName = drug.generic_name.toLowerCase();
    const displayName = drug.display_name.toLowerCase();
    return drugName === normalizedName || 
           displayName === normalizedName ||
           drugName === aliasedName ||
           displayName === aliasedName ||
           // Also check if the search term is contained in the drug name (for "propranolol hydrochloride" -> "propranolol")
           drugName.startsWith(normalizedName + ' ') ||
           displayName.startsWith(normalizedName + ' ');
  });
  
  if (exactMatch) {
    return {
      generic_name: exactMatch.generic_name,
      display_name: exactMatch.display_name,
      drug_class: exactMatch.drug_class,
      brand_names: exactMatch.brand_names,
      dosage_forms: exactMatch.dosage_forms,
      typical_strengths: exactMatch.typical_strengths,
      confidence: 0.99,
      source: 'fda_database',
      alternatives: []
    };
  }
  
  // Fuzzy search with stricter threshold
  const results = drugDatabaseIndex.search(aliasedName, { limit: 10 });
  
  if (results.length === 0) {
    return null;
  }
  
  const bestMatch = results[0];
  const score = 1 - (bestMatch.score || 0);
  
  // Check if the match is close enough (stricter for accuracy)
  // For very similar names, require higher confidence
  const matchName = bestMatch.item.generic_name.toLowerCase();
  const matchDisplay = bestMatch.item.display_name.toLowerCase();
  
  // If the search term is a significant part of the match, require lower threshold
  const isSubstring = matchName.includes(aliasedName) || matchDisplay.includes(aliasedName) ||
                      aliasedName.includes(matchName) || aliasedName.includes(matchDisplay);
  
  const threshold = isSubstring ? 0.6 : 0.75; // Stricter for non-substring matches
  
  if (score >= threshold) {
    const match = bestMatch.item;
    return {
      generic_name: match.generic_name,
      display_name: match.display_name,
      drug_class: match.drug_class,
      brand_names: match.brand_names,
      dosage_forms: match.dosage_forms,
      typical_strengths: match.typical_strengths,
      confidence: score,
      source: 'fda_database',
      alternatives: results.slice(1, 6).map(r => ({
        generic_name: r.item.generic_name,
        display_name: r.item.display_name,
        drug_class: r.item.drug_class,
        confidence: 1 - (r.score || 0)
      }))
    };
  }
  
  return null;
}

// Initialize database (lazy loading)
function initializeDatabase() {
  if (!drugDatabaseCache && !isLoading) {
    // Start loading in background
    loadFDADatabase().catch(err => {
      console.error('Error initializing FDA database:', err);
    });
  }
}

module.exports = {
  searchFDADatabase,
  loadFDADatabase,
  initializeDatabase,
  getDatabaseSize: () => drugDatabaseCache ? drugDatabaseCache.length : 0,
  isLoaded: () => drugDatabaseCache !== null
};

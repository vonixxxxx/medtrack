const axios = require('axios');
const Fuse = require('fuse.js');
const path = require('path');
const fs = require('fs');
const { searchFDADatabase, initializeDatabase } = require('./fdaDrugDatabaseService');
const { fuzzyMedicationSearch } = require('../../utils/medicationFuzzySearch');

const CONFIDENCE_THRESHOLD = 0.80;
const MIN_FUZZY_SCORE = 0.75;
const BIOGPT_MIN_CONFIDENCE = 0.70;

// Drug class mapping for common medications (fallback when BioGPT unavailable)
const DRUG_CLASS_MAP = {
  'propranolol': 'Beta Blocker',
  'metoprolol': 'Beta Blocker',
  'atenolol': 'Beta Blocker',
  'semaglutide': 'GLP-1 Receptor Agonist',
  'paracetamol': 'Analgesic',
  'acetaminophen': 'Analgesic',
  'ibuprofen': 'NSAID',
  'naproxen': 'NSAID',
  'aspirin': 'NSAID',
  'metformin': 'Biguanide',
  'lisinopril': 'ACE Inhibitor',
  'amlodipine': 'Calcium Channel Blocker',
  'atorvastatin': 'Statin',
  'simvastatin': 'Statin',
  'omeprazole': 'Proton Pump Inhibitor',
  'sertraline': 'SSRI',
  'fluoxetine': 'SSRI',
  'escitalopram': 'SSRI',
  'alprazolam': 'Benzodiazepine',
  'warfarin': 'Anticoagulant',
  'levothyroxine': 'Thyroid Hormone',
  'amoxicillin': 'Antibiotic',
  'azithromycin': 'Antibiotic'
};

// Load local dictionary (JSON array of { canonical, aliases })
let dictionary = [];
const dictPath = path.join(__dirname, '../../data/drug_dictionary.json');
try {
  if (fs.existsSync(dictPath)) {
    dictionary = JSON.parse(fs.readFileSync(dictPath, 'utf8'));
  } else {
    // Fallback to minimal dictionary if file doesn't exist
    dictionary = [
      { "canonical": "paracetamol", "aliases": ["acetaminophen", "paracetmol", "tylenol"] },
      { "canonical": "ibuprofen", "aliases": ["advil", "motrin"] },
      { "canonical": "naproxen", "aliases": ["aleve"] },
      { "canonical": "lisinopril", "aliases": ["zestril", "prinivil"] },
      { "canonical": "alprazolam", "aliases": ["xanax"] },
      { "canonical": "metformin", "aliases": ["glucophage"] }
    ];
  }
} catch (error) {
  console.error('Error loading drug dictionary:', error.message);
  dictionary = [];
}

const fuse = new Fuse(dictionary, {
  keys: ['canonical', 'aliases'],
  threshold: 0.4,
  includeScore: true,
  minMatchCharLength: 3
});

function normalize(input) {
  if (!input || typeof input !== 'string') return '';
  return input
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, '') // remove punctuation, emoji
    .replace(/\s+/g, ' ');
}

function isGreeting(norm) {
  const blacklist = new Set(['hello', 'hi', 'hey', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no', 'sure', 'alright']);
  return blacklist.has(norm);
}

// Local fuzzy search
function localFuzzyMatch(input) {
  const results = fuse.search(input);
  if (!results || results.length === 0) return null;
  const top = results[0];
  // Fuse score: lower is better. Convert to similarity-like score.
  const score = 1 - (top.score || 0); // 1 = exact, lower = less similar
  return { candidate: top.item.canonical, score };
}

// RxNorm simple approximateTerm call
async function queryRxNorm(term) {
  try {
    const url = `https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=${encodeURIComponent(term)}&maxEntries=5`;
    const res = await axios.get(url, { timeout: 5000 });
    const data = res.data;
    const candidates = data.approximateGroup && data.approximateGroup.candidate;
    if (!candidates || candidates.length === 0) return null;
    
    // Get the best candidate (first one)
    const bestCandidate = candidates[0];
    
    // RxNorm approximateTerm response may have 'name' field, but we'll get it from properties if needed
    // If we have an rxcui, get more details about the drug to get the proper name
    if (bestCandidate.rxcui) {
      try {
        // Get drug properties to extract the proper name
        const propUrl = `https://rxnav.nlm.nih.gov/REST/rxcui/${bestCandidate.rxcui}/properties.json`;
        const propRes = await axios.get(propUrl, { timeout: 3000 });
        const propData = propRes.data?.properties;
        
        if (propData && propData.name) {
          // Use the proper name from RxNorm properties
          bestCandidate.name = propData.name;
          bestCandidate.synonym = propData.synonym || null;
        } else if (bestCandidate.name) {
          // Use name from candidate if properties don't have it
          bestCandidate.name = bestCandidate.name;
        } else {
          // Fallback to original term
          bestCandidate.name = term;
        }
      } catch (propError) {
        // If property lookup fails, use the candidate name or original term
        bestCandidate.name = bestCandidate.name || term;
      }
    } else {
      // No rxcui, use candidate name or original term
      bestCandidate.name = bestCandidate.name || term;
    }
    
    return bestCandidate;
  } catch (error) {
    console.error('RxNorm query error:', error.message);
    return null;
  }
}

// DM+D fallback (example endpoint - replace with your actual path)
async function queryDmD(term) {
  try {
    // NHS dm+d API endpoint (placeholder - adjust based on actual API)
    const url = `https://digital.nhs.uk/api/dmd/search?term=${encodeURIComponent(term)}`;
    const res = await axios.get(url, { timeout: 5000 });
    const data = res.data;
    if (!data || data.length === 0) return null;
    return data[0];
  } catch (err) {
    // NHS API may not be available, return null silently
    return null;
  }
}

// Get drug class from mapping or return null
function getDrugClassFromMap(drugName) {
  const normalized = drugName.toLowerCase().trim();
  return DRUG_CLASS_MAP[normalized] || null;
}

// Call BioGPT verifying the already-found candidate (constrained prompt)
async function callBioGPTVerify(drugName, callBioGPTFn) {
  // callBioGPTFn is injected for easier testing/mocking
  const prompt = `
You are a strict biomedical verifier. The input below is already matched by an authoritative drug dictionary.

Return ONLY JSON, no additional text.

Drug: "${drugName}"

Return:
{
  "is_medication": true|false,
  "drug_class": "<string|null>",
  "is_generic": true|false,
  "is_brand": true|false,
  "confidence": <float 0.0-1.0>
}
`;

  const resp = await callBioGPTFn(prompt);
  // defensive parsing (callers should pass a function that returns parsed JSON)
  if (!resp || typeof resp !== 'object') return null;
  const { is_medication, confidence } = resp;
  if (typeof is_medication !== 'boolean' || typeof confidence !== 'number') return null;
  
  // If BioGPT didn't provide drug_class, try to get it from our mapping
  if (!resp.drug_class) {
    resp.drug_class = getDrugClassFromMap(drugName);
  }
  
  return resp;
}

async function validateMedication(userInput, options = {}) {
  // options.callBioGPT(drugName) -> promise that resolves parsed JSON
  const callBioGPTFn = options.callBioGPT;

  const norm = normalize(userInput);
  if (!norm) return { found: false, reason: 'invalid_input', message: 'Input empty or invalid' };
  if (isGreeting(norm)) return { found: false, reason: 'greeting', message: 'Input looks like a greeting' };
  if (norm.length < 3) return { found: false, reason: 'too_short', message: 'Input too short to be a drug name' };

  // PRIMARY: Search FDA database (3+ million medications - comprehensive source)
  try {
    initializeDatabase(); // Ensure database is loaded (non-blocking)
    const fdaMatch = searchFDADatabase(norm);
    
    if (fdaMatch && fdaMatch.confidence >= 0.5) {
      // Found in FDA database - this is our authoritative source
      const drugName = fdaMatch.generic_name;
      const displayName = fdaMatch.display_name || drugName;
      const drugClass = fdaMatch.drug_class !== 'Unknown' ? fdaMatch.drug_class : getDrugClassFromMap(drugName);
      
      // Try BioGPT for additional verification (non-blocking)
      let bio = null;
      if (callBioGPTFn) {
        try {
          bio = await callBioGPTVerify(displayName, callBioGPTFn);
          if (!bio || !bio.is_medication || bio.confidence < BIOGPT_MIN_CONFIDENCE) {
            // Use FDA database drug class if BioGPT fails
            bio = {
              is_medication: true,
              drug_class: drugClass || 'Unknown',
              confidence: fdaMatch.confidence,
              is_generic: true,
              is_brand: false
            };
          } else if (!bio.drug_class || bio.drug_class === 'Unknown') {
            // Use FDA database drug class if BioGPT doesn't provide one
            bio.drug_class = drugClass || 'Unknown';
          }
        } catch (bioError) {
          console.error('BioGPT verification error (non-blocking):', bioError.message);
          bio = {
            is_medication: true,
            drug_class: drugClass || 'Unknown',
            confidence: fdaMatch.confidence,
            is_generic: true,
            is_brand: false
          };
        }
      } else {
        bio = {
          is_medication: true,
          drug_class: drugClass || 'Unknown',
          confidence: fdaMatch.confidence,
          is_generic: true,
          is_brand: false
        };
      }
      
      return {
        found: true,
        name: displayName,
        generic_name: drugName,
        score: fdaMatch.confidence,
        source: 'fda_database',
        bio: bio,
        brand_names: fdaMatch.brand_names || [],
        dosage_forms: fdaMatch.dosage_forms || [],
        typical_strengths: fdaMatch.typical_strengths || [],
        alternatives: fdaMatch.alternatives || []
      };
    }
  } catch (fdaError) {
    console.error('FDA database search error:', fdaError.message);
    // Continue to fallback methods
  }
  
  // FALLBACK 1: Search master medication database (medications.json)
  const { best, matches } = fuzzyMedicationSearch(norm, 5);
  if (best) {
    const drugName = best.generic || best.name;
    const drugClass = best.drug_class || getDrugClassFromMap(drugName);
    const confidence = best._score !== undefined ? Math.max(0.6, 1 - best._score) : 0.9;
    
    let bio = {
      is_medication: true,
      drug_class: drugClass || 'Unknown',
      confidence: confidence,
      is_generic: true,
      is_brand: false
    };
    
    return {
      found: true,
      name: drugName,
      score: confidence,
      source: 'master_database',
      bio: bio,
      brand_names: best.brand_names || [],
      dosage_forms: best.dosage_forms || [],
      typical_strengths: best.typical_strengths || [],
      alternatives: matches.slice(1).map(m => ({
        generic_name: m.generic || m.name,
        drug_class: m.drug_class || 'Unknown',
        confidence: m._score !== undefined ? Math.max(0.6, 1 - m._score) : 0.8
      }))
    };
  }

  // Try local fuzzy dictionary (for speed, but not required)
  const local = localFuzzyMatch(norm);
  const hasGoodLocalMatch = local && local.score >= MIN_FUZZY_SCORE;

  // Primary: RxNorm - try original input first, then canonical name if local match exists
  let rx = await queryRxNorm(norm); // Try original input first (e.g., "propranolol", "semaglutide")
  
  // If no good RxNorm match with original input and we have a local match, try canonical name
  if ((!rx || parseFloat(rx.score) < 5) && hasGoodLocalMatch) {
    rx = await queryRxNorm(local.candidate);
  }
  
  // RxNorm returns score as string, convert to number
  const rxScore = rx ? parseFloat(rx.score) : 0;
  
  // Accept RxNorm match if score >= 5 (good match threshold) OR if score >= 3 (moderate match)
  // This allows medications not in local dictionary to still be found
  if (rx && rxScore >= 3) {
    // Convert RxNorm score (0-10+) to 0-1 scale for consistency
    // Score of 3 = 0.3, score of 5 = 0.5, score of 10 = 0.99
    const normalizedScore = Math.min(Math.max(rxScore / 10, 0.3), 0.99);
    
    // Get drug name from RxNorm response
    const drugName = rx.name || rx.term || (hasGoodLocalMatch ? local.candidate : norm);
    
    // validate with BioGPT only if available (non-blocking - RxNorm is authoritative)
    let bio = null;
    if (callBioGPTFn) {
      try {
        bio = await callBioGPTVerify(drugName, callBioGPTFn);
        // If BioGPT fails or disagrees, log but don't block - RxNorm is authoritative
        if (!bio || !bio.is_medication || bio.confidence < BIOGPT_MIN_CONFIDENCE) {
          console.log(`BioGPT verification failed for ${drugName}, but RxNorm match is strong - accepting anyway`);
          // Still use RxNorm match, try to get drug_class from mapping
          const drugClass = getDrugClassFromMap(drugName);
          bio = { 
            is_medication: true, 
            drug_class: drugClass || 'Unknown', 
            confidence: 0.5,
            is_generic: true,
            is_brand: false
          };
        } else if (bio && !bio.drug_class) {
          // BioGPT succeeded but didn't provide drug_class, try mapping
          bio.drug_class = getDrugClassFromMap(drugName) || 'Unknown';
        }
      } catch (bioError) {
        console.error('BioGPT verification error (non-blocking):', bioError.message);
        // Continue without BioGPT - RxNorm is authoritative, but provide default bio info
        const drugClass = getDrugClassFromMap(drugName);
        bio = { 
          is_medication: true, 
          drug_class: drugClass || 'Unknown', 
          confidence: 0.5,
          is_generic: true,
          is_brand: false
        };
      }
    } else {
      // No BioGPT available, provide default bio info with drug class from mapping
      const drugClass = getDrugClassFromMap(drugName);
      bio = { 
        is_medication: true, 
        drug_class: drugClass || 'Unknown', 
        confidence: 0.5,
        is_generic: true,
        is_brand: false
      };
    }
    
    // Accept RxNorm match (with or without BioGPT)
    return {
      found: true,
      name: drugName,
      rxcui: rx.rxcui,
      score: normalizedScore,
      source: 'rxnorm',
      bio: bio || undefined
    };
  }

  // If we have a good local match but RxNorm failed, still accept it with lower confidence
  if (hasGoodLocalMatch) {
    let bio = null;
    if (callBioGPTFn) {
      try {
        bio = await callBioGPTVerify(local.candidate, callBioGPTFn);
        if (!bio || !bio.is_medication || bio.confidence < BIOGPT_MIN_CONFIDENCE) {
          const drugClass = getDrugClassFromMap(local.candidate);
          bio = { 
            is_medication: true, 
            drug_class: drugClass || 'Unknown', 
            confidence: 0.5,
            is_generic: true,
            is_brand: false
          };
        } else if (bio && !bio.drug_class) {
          bio.drug_class = getDrugClassFromMap(local.candidate) || 'Unknown';
        }
      } catch (bioError) {
        console.error('BioGPT verification error (non-blocking):', bioError.message);
        const drugClass = getDrugClassFromMap(local.candidate);
        bio = { 
          is_medication: true, 
          drug_class: drugClass || 'Unknown', 
          confidence: 0.5,
          is_generic: true,
          is_brand: false
        };
      }
    } else {
      const drugClass = getDrugClassFromMap(local.candidate);
      bio = { 
        is_medication: true, 
        drug_class: drugClass || 'Unknown', 
        confidence: 0.5,
        is_generic: true,
        is_brand: false
      };
    }
    
    return {
      found: true,
      name: local.candidate,
      score: local.score,
      source: 'local_dictionary',
      bio: bio || undefined
    };
  }

  // Fallback: dm+d (only if we have a local candidate)
  if (local && local.candidate) {
    const dmd = await queryDmD(local.candidate);
    if (dmd) {
      let bio = null;
      if (callBioGPTFn) {
        try {
          bio = await callBioGPTVerify(dmd.name || local.candidate, callBioGPTFn);
          if (!bio || !bio.is_medication || bio.confidence < BIOGPT_MIN_CONFIDENCE) {
            const drugClass = getDrugClassFromMap(dmd.name || local.candidate);
            bio = { 
              is_medication: true, 
              drug_class: drugClass || 'Unknown', 
              confidence: 0.5,
              is_generic: true,
              is_brand: false
            };
          } else if (bio && !bio.drug_class) {
            bio.drug_class = getDrugClassFromMap(dmd.name || local.candidate) || 'Unknown';
          }
        } catch (bioError) {
          console.error('BioGPT verification error (non-blocking):', bioError.message);
          const drugClass = getDrugClassFromMap(dmd.name || local.candidate);
          bio = { 
            is_medication: true, 
            drug_class: drugClass || 'Unknown', 
            confidence: 0.5,
            is_generic: true,
            is_brand: false
          };
        }
      } else {
        const drugClass = getDrugClassFromMap(dmd.name || local.candidate);
        bio = { 
          is_medication: true, 
          drug_class: drugClass || 'Unknown', 
          confidence: 0.5,
          is_generic: true,
          is_brand: false
        };
      }
      return {
        found: true,
        name: dmd.name || local.candidate,
        dmdId: dmd.id || null,
        score: local.score,
        source: 'dmd',
        bio: bio || undefined
      };
    }
  }

  // No authoritative source matched
  return {
    found: false,
    reason: 'no_authoritative_match',
    message: 'No close local dictionary match Please try: • Using the generic name (e.g., "acetaminophen" instead of "Tylenol") • Checking your spelling • Using a different medication name',
    suggestions: local ? [local.candidate] : []
  };
}

module.exports = { validateMedication, normalize, localFuzzyMatch };

const axios = require('axios');
const Fuse = require('fuse.js');
const path = require('path');
const fs = require('fs');

const CONFIDENCE_THRESHOLD = 0.80;
const MIN_FUZZY_SCORE = 0.75;
const BIOGPT_MIN_CONFIDENCE = 0.70;

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
    // candidate has properties: rxcui, score, rank
    return candidates[0];
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
  return resp;
}

async function validateMedication(userInput, options = {}) {
  // options.callBioGPT(drugName) -> promise that resolves parsed JSON
  const callBioGPTFn = options.callBioGPT;

  const norm = normalize(userInput);
  if (!norm) return { found: false, reason: 'invalid_input', message: 'Input empty or invalid' };
  if (isGreeting(norm)) return { found: false, reason: 'greeting', message: 'Input looks like a greeting' };
  if (norm.length < 3) return { found: false, reason: 'too_short', message: 'Input too short to be a drug name' };

  // local fuzzy dictionary
  const local = localFuzzyMatch(norm);
  if (!local || local.score < MIN_FUZZY_SCORE) {
    return { found: false, reason: 'no_local_match', message: 'No close local dictionary match', suggestions: [] };
  }

  // Primary: RxNorm - try both original input and canonical name
  let rx = await queryRxNorm(norm); // Try original input first (e.g., "acetaminophen")
  if (!rx || parseFloat(rx.score) < 5) { // RxNorm score is typically 0-10+, 5+ is good match
    // If original input doesn't match, try canonical name
    rx = await queryRxNorm(local.candidate);
  }
  // RxNorm returns score as string, convert to number
  const rxScore = rx ? parseFloat(rx.score) : 0;
  if (rx && rxScore >= 5) { // Accept if score >= 5 (good match threshold)
    // Convert RxNorm score (0-10+) to 0-1 scale for consistency
    const normalizedScore = Math.min(rxScore / 10, 0.99);
    const drugName = rx.name || rx.term || local.candidate;
    
    // validate with BioGPT only if available (non-blocking - RxNorm is authoritative)
    let bio = null;
    if (callBioGPTFn) {
      try {
        bio = await callBioGPTVerify(drugName, callBioGPTFn);
        // If BioGPT fails or disagrees, log but don't block - RxNorm is authoritative
        if (!bio || !bio.is_medication || bio.confidence < BIOGPT_MIN_CONFIDENCE) {
          console.log(`BioGPT verification failed for ${drugName}, but RxNorm match is strong - accepting anyway`);
          bio = null; // Don't include failed bio verification
        }
      } catch (bioError) {
        console.error('BioGPT verification error (non-blocking):', bioError.message);
        // Continue without BioGPT - RxNorm is authoritative
      }
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

  // Fallback: dm+d
  const dmd = await queryDmD(local.candidate);
  if (dmd) {
    let bio = null;
    if (callBioGPTFn) {
      try {
        bio = await callBioGPTVerify(dmd.name || local.candidate, callBioGPTFn);
        if (!bio || !bio.is_medication || bio.confidence < BIOGPT_MIN_CONFIDENCE) {
          console.log(`BioGPT verification failed for dm+d match, but accepting anyway`);
          bio = null;
        }
      } catch (bioError) {
        console.error('BioGPT verification error (non-blocking):', bioError.message);
      }
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

  // No authoritative source matched
  return {
    found: false,
    reason: 'no_authoritative_match',
    message: 'No authoritative database matched this term',
    suggestions: [local.candidate]
  };
}

module.exports = { validateMedication, normalize, localFuzzyMatch };

const express = require('express');
const axios = require('axios');
const router = express.Router();
const { fuzzyMedicationSearch } = require('../utils/medicationFuzzySearch');
const { runOllamaParser } = require('../utils/ollamaParser'); // Or your LLM connector

router.post('/validateMedication', async (req, res) => {
  // Accept multiple input keys for max compatibility
  const { medication, medication_name, query, name, text, term } = req.body;
  const inputName = (medication || medication_name || query || name || text || term || '').trim();
  if (!inputName || inputName.length < 2) {
    return res.status(400).json({ success: false, error: 'Medication name is required' });
  }

  // Fuzzy search local
  const { best, matches } = fuzzyMedicationSearch(inputName, 5);
  if (best) {
        return res.json({
          success: true,
          data: {
        ...best,
        confidence: best._score !== undefined ? 1 - best._score : 0.9,
        alternatives: matches.slice(1),
        original_input: inputName,
        source: 'fuzzy_local'
      }
    });
  }

  // Fallback: BioGPT/Ollama
  try {
    const bioGPTresult = await runOllamaParser(inputName);
    if (bioGPTresult && bioGPTresult.name) {
      return res.json({
        success: true,
        data: { ...bioGPTresult, confidence: 0.6, alternatives: [], original_input: inputName, source: 'llm_fallback' }
      });
    }
  } catch (e) {
    console.error('BioGPT/Ollama error:', e);
  }
  // No good match found
  return res.status(404).json({ success: false, error: 'Medication not found', original_input: inputName });
});

module.exports = router;
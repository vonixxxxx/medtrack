const express = require('express');
const axios = require('axios');
const router = express.Router();
require('dotenv').config();

// AI Backend URLs
const AI_BACKEND_URL = process.env.AI_BACKEND_URL || 'http://localhost:5003';
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;
const HUGGINGFACE_API_URL = 'https://api-inference.huggingface.co/models/microsoft/BioGPT-Large';

// New BioGPT validation endpoint
router.post('/validateMedication', async (req, res) => {
  try {
    const { medication_name } = req.body;
    
    if (!medication_name) {
      return res.status(400).json({
        success: false,
        error: 'Medication name is required'
      });
    }

    if (!HUGGINGFACE_API_KEY) {
      return res.status(500).json({
        success: false,
        error: 'Hugging Face API key not configured'
      });
    }

    // Call BioGPT via Hugging Face API (disabled due to API key permissions)
    // const biogptResult = await callBioGPT(medication_name);
    
    // if (biogptResult.success) {
    //   return res.json(biogptResult);
    // } else {
      // Fallback to local validation
      const fallbackResult = await enhancedMedicationSearch(medication_name, 1);
      
      if (fallbackResult.results.length > 0) {
        const result = fallbackResult.results[0];
        return res.json({
          success: true,
          data: {
            generic_name: result.name,
            brand_names: result.brands || [],
            drug_class: result.drug_class || 'Unknown',
            dosage_forms: result.dosage_forms || ['tablet'],
            typical_strengths: result.strengths || [],
            indications: result.indications || [],
            confidence: result.confidence || 0.7,
            alternatives: [],
            is_ambiguous: false,
            validation_notes: 'Validated using local database (BioGPT unavailable)',
            original_input: medication_name,
            sources: ['local_fallback']
          },
          suggested_metrics: getSuggestedMetrics(result)
        });
      }

      // No match found
      return res.json({
        success: true,
        data: {
          generic_name: medication_name,
          brand_names: [],
          drug_class: 'Unknown',
          dosage_forms: ['tablet'],
          typical_strengths: [],
          indications: [],
          confidence: 0.1,
          alternatives: [],
          is_ambiguous: true,
          validation_notes: 'No match found',
          original_input: medication_name,
          sources: ['no_match']
        },
        suggested_metrics: ['General Health', 'Side Effects']
      });
    // }

  } catch (error) {
    console.error('Error validating medication with BioGPT:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to validate medication'
    });
  }
});

// Original validation endpoint (kept for backward compatibility)
router.post('/validate', async (req, res) => {
  try {
    const { medication_name } = req.body;
    
    if (!medication_name) {
      return res.status(400).json({
        success: false,
        error: 'Medication name is required'
      });
    }

    // Try Python AI backend first
    try {
      const response = await axios.post(`${AI_BACKEND_URL}/api/medications/validate`, {
        medication_name,
        user_id: req.body.user_id
      }, {
        timeout: 10000
      });

      if (response.data.success) {
        return res.json(response.data);
      }
    } catch (aiError) {
      console.log('AI backend unavailable, using fallback:', aiError.message);
    }

    // Fallback to local validation
    const fallbackResult = await enhancedMedicationSearch(medication_name, 1);
    
    if (fallbackResult.results.length > 0) {
      const result = fallbackResult.results[0];
      return res.json({
        success: true,
        data: {
          generic_name: result.name,
          brand_names: result.brands || [],
          drug_class: result.drug_class || 'Unknown',
          dosage_forms: result.dosage_forms || ['tablet'],
          typical_strengths: result.strengths || [],
          indications: result.indications || [],
          confidence: result.confidence || 0.7,
          alternatives: [],
          is_ambiguous: false,
          validation_notes: 'Validated using local database',
          original_input: medication_name,
          sources: ['local_fallback']
        },
        suggested_metrics: getSuggestedMetrics(result)
      });
    }

    // No match found
    res.json({
      success: true,
      data: {
        generic_name: medication_name,
        brand_names: [],
        drug_class: 'Unknown',
        dosage_forms: ['tablet'],
        typical_strengths: [],
        indications: [],
        confidence: 0.1,
        alternatives: [],
        is_ambiguous: true,
        validation_notes: 'No match found',
        original_input: medication_name,
        sources: ['no_match']
      },
      suggested_metrics: ['General Health', 'Side Effects']
    });

  } catch (error) {
    console.error('Error validating medication:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to validate medication'
    });
  }
});

// Helper function to get suggested metrics based on medication
function getSuggestedMetrics(medication) {
  const name = medication.name?.toLowerCase() || '';
  const drugClass = medication.drug_class?.toLowerCase() || '';
  const indications = medication.indications || [];

  // Diabetes medications
  if (name.includes('metformin') || name.includes('insulin') || name.includes('glipizide')) {
    return ['Blood Glucose', 'Weight', 'Blood Pressure', 'General Health'];
  }

  // Cardiovascular medications
  if (drugClass.includes('ace inhibitor') || drugClass.includes('beta blocker') || drugClass.includes('statin')) {
    return ['Blood Pressure', 'Heart Rate', 'Weight', 'General Health'];
  }

  // Pain medications
  if (name.includes('ibuprofen') || name.includes('acetaminophen') || name.includes('naproxen')) {
    return ['Pain Level', 'Sleep Quality', 'General Health'];
  }

  // Blood pressure medications
  if (indications.some(ind => ind.toLowerCase().includes('hypertension') || ind.toLowerCase().includes('blood pressure'))) {
    return ['Blood Pressure', 'Heart Rate', 'Weight', 'General Health'];
  }

  // Pain relief
  if (indications.some(ind => ind.toLowerCase().includes('pain') || ind.toLowerCase().includes('inflammation'))) {
    return ['Pain Level', 'Sleep Quality', 'General Health'];
  }

  return ['General Health', 'Side Effects'];
}

// Enhanced medication search function with comprehensive database
async function enhancedMedicationSearch(query, limit = 5) {
  const medicationDatabase = {
    // Pain Relief & Fever
    'acetaminophen': {
      name: 'acetaminophen',
      brands: ['Tylenol', 'Panadol', 'Paracetamol', 'Tempra'],
      drug_class: 'Analgesic/Antipyretic',
      dosage_forms: ['tablet', 'capsule', 'liquid', 'suppository'],
      strengths: ['325mg', '500mg', '650mg', '1000mg'],
      indications: ['pain relief', 'fever reduction'],
      confidence: 0.9
    },
    'ibuprofen': {
      name: 'ibuprofen',
      brands: ['Advil', 'Motrin', 'Nurofen', 'Brufen'],
      drug_class: 'NSAID',
      dosage_forms: ['tablet', 'capsule', 'liquid', 'gel'],
      strengths: ['200mg', '400mg', '600mg', '800mg'],
      indications: ['pain relief', 'inflammation', 'fever'],
      confidence: 0.9
    },
    'aspirin': {
      name: 'aspirin',
      brands: ['Bayer', 'Ecotrin', 'Bufferin'],
      drug_class: 'NSAID/Antiplatelet',
      dosage_forms: ['tablet', 'chewable tablet', 'suppository'],
      strengths: ['81mg', '325mg', '500mg'],
      indications: ['pain relief', 'fever', 'heart attack prevention'],
      confidence: 0.9
    },
    'naproxen': {
      name: 'naproxen',
      brands: ['Aleve', 'Naprosyn', 'Anaprox'],
      drug_class: 'NSAID',
      dosage_forms: ['tablet', 'capsule', 'gel'],
      strengths: ['220mg', '250mg', '375mg', '500mg'],
      indications: ['pain relief', 'inflammation', 'arthritis'],
      confidence: 0.9
    },

    // Diabetes Medications
    'metformin': {
      name: 'metformin',
      brands: ['Glucophage', 'Fortamet', 'Glumetza'],
      drug_class: 'Biguanide',
      dosage_forms: ['tablet', 'extended-release tablet'],
      strengths: ['500mg', '850mg', '1000mg'],
      indications: ['type 2 diabetes'],
      confidence: 0.9
    },
    'insulin': {
      name: 'insulin',
      brands: ['Humalog', 'Novolog', 'Lantus', 'Levemir'],
      drug_class: 'Hormone',
      dosage_forms: ['injection', 'pen', 'pump'],
      strengths: ['U-100', 'U-200', 'U-300'],
      indications: ['diabetes', 'hyperglycemia'],
      confidence: 0.9
    },
    'glipizide': {
      name: 'glipizide',
      brands: ['Glucotrol', 'Glipizide XL'],
      drug_class: 'Sulfonylurea',
      dosage_forms: ['tablet', 'extended-release tablet'],
      strengths: ['5mg', '10mg'],
      indications: ['type 2 diabetes'],
      confidence: 0.9
    },

    // Cardiovascular Medications
    'lisinopril': {
      name: 'lisinopril',
      brands: ['Prinivil', 'Zestril'],
      drug_class: 'ACE Inhibitor',
      dosage_forms: ['tablet'],
      strengths: ['2.5mg', '5mg', '10mg', '20mg', '40mg'],
      indications: ['hypertension', 'heart failure'],
      confidence: 0.9
    },
    'amlodipine': {
      name: 'amlodipine',
      brands: ['Norvasc'],
      drug_class: 'Calcium Channel Blocker',
      dosage_forms: ['tablet'],
      strengths: ['2.5mg', '5mg', '10mg'],
      indications: ['hypertension', 'angina'],
      confidence: 0.9
    },
    'atorvastatin': {
      name: 'atorvastatin',
      brands: ['Lipitor'],
      drug_class: 'Statin',
      dosage_forms: ['tablet'],
      strengths: ['10mg', '20mg', '40mg', '80mg'],
      indications: ['high cholesterol', 'cardiovascular disease prevention'],
      confidence: 0.9
    },
    'metoprolol': {
      name: 'metoprolol',
      brands: ['Lopressor', 'Toprol XL'],
      drug_class: 'Beta Blocker',
      dosage_forms: ['tablet', 'extended-release tablet'],
      strengths: ['25mg', '50mg', '100mg', '200mg'],
      indications: ['hypertension', 'heart failure', 'angina'],
      confidence: 0.9
    },

    // Mental Health Medications
    'sertraline': {
      name: 'sertraline',
      brands: ['Zoloft'],
      drug_class: 'SSRI',
      dosage_forms: ['tablet', 'liquid'],
      strengths: ['25mg', '50mg', '100mg', '150mg', '200mg'],
      indications: ['depression', 'anxiety', 'panic disorder'],
      confidence: 0.9
    },
    'fluoxetine': {
      name: 'fluoxetine',
      brands: ['Prozac'],
      drug_class: 'SSRI',
      dosage_forms: ['tablet', 'capsule', 'liquid'],
      strengths: ['10mg', '20mg', '40mg'],
      indications: ['depression', 'anxiety', 'OCD'],
      confidence: 0.9
    },
    'alprazolam': {
      name: 'alprazolam',
      brands: ['Xanax'],
      drug_class: 'Benzodiazepine',
      dosage_forms: ['tablet', 'extended-release tablet'],
      strengths: ['0.25mg', '0.5mg', '1mg', '2mg'],
      indications: ['anxiety', 'panic disorder'],
      confidence: 0.9
    },

    // ADHD Medications
    'adderall': {
      name: 'amphetamine/dextroamphetamine',
      brands: ['Adderall', 'Adderall XR'],
      drug_class: 'Stimulant',
      dosage_forms: ['tablet', 'extended-release capsule'],
      strengths: ['5mg', '10mg', '15mg', '20mg', '25mg', '30mg'],
      indications: ['ADHD', 'narcolepsy'],
      confidence: 0.9
    },
    'methylphenidate': {
      name: 'methylphenidate',
      brands: ['Ritalin', 'Concerta', 'Daytrana'],
      drug_class: 'Stimulant',
      dosage_forms: ['tablet', 'extended-release tablet', 'patch'],
      strengths: ['5mg', '10mg', '15mg', '20mg', '30mg', '36mg', '54mg'],
      indications: ['ADHD', 'narcolepsy'],
      confidence: 0.9
    },

    // Antibiotics
    'amoxicillin': {
      name: 'amoxicillin',
      brands: ['Amoxil', 'Trimox'],
      drug_class: 'Penicillin Antibiotic',
      dosage_forms: ['tablet', 'capsule', 'liquid', 'chewable tablet'],
      strengths: ['250mg', '500mg', '875mg'],
      indications: ['bacterial infections'],
      confidence: 0.9
    },
    'azithromycin': {
      name: 'azithromycin',
      brands: ['Zithromax', 'Z-Pak'],
      drug_class: 'Macrolide Antibiotic',
      dosage_forms: ['tablet', 'liquid', 'powder'],
      strengths: ['250mg', '500mg', '600mg'],
      indications: ['bacterial infections', 'respiratory infections'],
      confidence: 0.9
    },

    // Gastrointestinal
    'omeprazole': {
      name: 'omeprazole',
      brands: ['Prilosec'],
      drug_class: 'Proton Pump Inhibitor',
      dosage_forms: ['tablet', 'capsule', 'powder'],
      strengths: ['10mg', '20mg', '40mg'],
      indications: ['GERD', 'ulcers', 'acid reflux'],
      confidence: 0.9
    },
    'ranitidine': {
      name: 'ranitidine',
      brands: ['Zantac'],
      drug_class: 'H2 Blocker',
      dosage_forms: ['tablet', 'liquid'],
      strengths: ['75mg', '150mg', '300mg'],
      indications: ['GERD', 'ulcers', 'acid reflux'],
      confidence: 0.9
    },

    // Respiratory
    'albuterol': {
      name: 'albuterol',
      brands: ['Ventolin', 'ProAir', 'Proventil'],
      drug_class: 'Bronchodilator',
      dosage_forms: ['inhaler', 'nebulizer solution', 'tablet'],
      strengths: ['90mcg', '108mcg'],
      indications: ['asthma', 'COPD', 'bronchospasm'],
      confidence: 0.9
    },

    // Common Brand Names
    'tylenol': {
      name: 'acetaminophen',
      brands: ['Tylenol', 'Panadol', 'Paracetamol'],
      drug_class: 'Analgesic/Antipyretic',
      dosage_forms: ['tablet', 'capsule', 'liquid', 'suppository'],
      strengths: ['325mg', '500mg', '650mg', '1000mg'],
      indications: ['pain relief', 'fever reduction'],
      confidence: 0.9
    },
    'advil': {
      name: 'ibuprofen',
      brands: ['Advil', 'Motrin', 'Nurofen'],
      drug_class: 'NSAID',
      dosage_forms: ['tablet', 'capsule', 'liquid', 'gel'],
      strengths: ['200mg', '400mg', '600mg', '800mg'],
      indications: ['pain relief', 'inflammation', 'fever'],
      confidence: 0.9
    },
    'motrin': {
      name: 'ibuprofen',
      brands: ['Advil', 'Motrin', 'Nurofen'],
      drug_class: 'NSAID',
      dosage_forms: ['tablet', 'capsule', 'liquid', 'gel'],
      strengths: ['200mg', '400mg', '600mg', '800mg'],
      indications: ['pain relief', 'inflammation', 'fever'],
      confidence: 0.9
    },
    'panadol': {
      name: 'acetaminophen',
      brands: ['Tylenol', 'Panadol', 'Paracetamol'],
      drug_class: 'Analgesic/Antipyretic',
      dosage_forms: ['tablet', 'capsule', 'liquid', 'suppository'],
      strengths: ['325mg', '500mg', '650mg', '1000mg'],
      indications: ['pain relief', 'fever reduction'],
      confidence: 0.9
    },
    'xanax': {
      name: 'alprazolam',
      brands: ['Xanax'],
      drug_class: 'Benzodiazepine',
      dosage_forms: ['tablet', 'extended-release tablet'],
      strengths: ['0.25mg', '0.5mg', '1mg', '2mg'],
      indications: ['anxiety', 'panic disorder'],
      confidence: 0.9
    },
    'prozac': {
      name: 'fluoxetine',
      brands: ['Prozac'],
      drug_class: 'SSRI',
      dosage_forms: ['tablet', 'capsule', 'liquid'],
      strengths: ['10mg', '20mg', '40mg'],
      indications: ['depression', 'anxiety', 'OCD'],
      confidence: 0.9
    },
    'zoloft': {
      name: 'sertraline',
      brands: ['Zoloft'],
      drug_class: 'SSRI',
      dosage_forms: ['tablet', 'liquid'],
      strengths: ['25mg', '50mg', '100mg', '150mg', '200mg'],
      indications: ['depression', 'anxiety', 'panic disorder'],
      confidence: 0.9
    },
    'lipitor': {
      name: 'atorvastatin',
      brands: ['Lipitor'],
      drug_class: 'Statin',
      dosage_forms: ['tablet'],
      strengths: ['10mg', '20mg', '40mg', '80mg'],
      indications: ['high cholesterol', 'cardiovascular disease prevention'],
      confidence: 0.9
    },
    'norvasc': {
      name: 'amlodipine',
      brands: ['Norvasc'],
      drug_class: 'Calcium Channel Blocker',
      dosage_forms: ['tablet'],
      strengths: ['2.5mg', '5mg', '10mg'],
      indications: ['hypertension', 'angina'],
      confidence: 0.9
    },
    'prilosec': {
      name: 'omeprazole',
      brands: ['Prilosec'],
      drug_class: 'Proton Pump Inhibitor',
      dosage_forms: ['tablet', 'capsule', 'powder'],
      strengths: ['10mg', '20mg', '40mg'],
      indications: ['GERD', 'ulcers', 'acid reflux'],
      confidence: 0.9
    }
  };

  const results = [];
  const queryLower = query.toLowerCase();

  // Direct match
  if (medicationDatabase[queryLower]) {
    results.push(medicationDatabase[queryLower]);
  }

  // Case-insensitive partial matching
  for (const [key, med] of Object.entries(medicationDatabase)) {
    if (key !== queryLower) {
      // Check if query is contained in key or vice versa
      if (key.includes(queryLower) || queryLower.includes(key)) {
        results.push({ ...med, confidence: 0.8 });
      }
      // Check brand names
      else if (med.brands && med.brands.some(brand => 
        brand.toLowerCase().includes(queryLower) || queryLower.includes(brand.toLowerCase())
      )) {
        results.push({ ...med, confidence: 0.8 });
      }
      // Fuzzy matching for typos
      else if (levenshteinDistance(queryLower, key) < 3) {
        results.push({ ...med, confidence: 0.7 });
      }
    }
  }

  return {
    results: results.slice(0, limit),
    total: results.length,
    query_processed: query
  };
}

function levenshteinDistance(str1, str2) {
  const matrix = [];
  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }
  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }
  return matrix[str2.length][str1.length];
}

// Call BioGPT via Hugging Face API
async function callBioGPT(medicationName) {
  try {
    const prompt = `Normalize the medication name '${medicationName}' to its generic equivalent and provide valid dosage forms. Return the response in JSON format with the following structure:
{
  "generic_name": "generic name",
  "brand_names": ["brand1", "brand2"],
  "drug_class": "drug class",
  "dosage_forms": ["tablet", "capsule", "liquid"],
  "typical_strengths": ["100mg", "200mg"],
  "indications": ["indication1", "indication2"],
  "confidence": 0.9,
  "alternatives": [],
  "is_ambiguous": false,
  "validation_notes": "BioGPT validation"
}`;

    const response = await axios.post(HUGGINGFACE_API_URL, {
      inputs: prompt
    }, {
      headers: {
        'Authorization': `Bearer ${HUGGINGFACE_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });

    if (response.data && response.data.length > 0) {
      const generatedText = response.data[0].generated_text;
      
      // Try to extract JSON from the response
      const jsonMatch = generatedText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          const parsedData = JSON.parse(jsonMatch[0]);
          return {
            success: true,
            data: {
              ...parsedData,
              original_input: medicationName,
              sources: ['biogpt_huggingface']
            },
            suggested_metrics: getSuggestedMetrics(parsedData)
          };
        } catch (parseError) {
          console.log('Failed to parse BioGPT JSON response:', parseError.message);
        }
      }
      
      // If JSON parsing fails, try to extract information from text
      const extractedData = extractMedicationInfoFromText(generatedText, medicationName);
      return {
        success: true,
        data: extractedData,
        suggested_metrics: getSuggestedMetrics(extractedData)
      };
    }

    return { success: false, error: 'No response from BioGPT' };

  } catch (error) {
    console.error('Error calling BioGPT:', error.message);
    return { success: false, error: error.message };
  }
}

// Extract medication information from BioGPT text response
function extractMedicationInfoFromText(text, originalInput) {
  const lines = text.split('\n');
  const result = {
    generic_name: originalInput,
    brand_names: [],
    drug_class: 'Unknown',
    dosage_forms: ['tablet'],
    typical_strengths: [],
    indications: [],
    confidence: 0.7,
    alternatives: [],
    is_ambiguous: false,
    validation_notes: 'BioGPT text extraction',
    original_input: originalInput,
    sources: ['biogpt_text']
  };

  // Simple text parsing to extract information
  for (const line of lines) {
    const lowerLine = line.toLowerCase();
    
    if (lowerLine.includes('generic') && lowerLine.includes(':')) {
      const match = line.match(/generic[:\s]+([^,\n]+)/i);
      if (match) result.generic_name = match[1].trim();
    }
    
    if (lowerLine.includes('brand') && lowerLine.includes(':')) {
      const match = line.match(/brand[:\s]+([^,\n]+)/i);
      if (match) result.brand_names = match[1].split(',').map(b => b.trim());
    }
    
    if (lowerLine.includes('class') && lowerLine.includes(':')) {
      const match = line.match(/class[:\s]+([^,\n]+)/i);
      if (match) result.drug_class = match[1].trim();
    }
    
    if (lowerLine.includes('dosage') || lowerLine.includes('form')) {
      const forms = ['tablet', 'capsule', 'liquid', 'injection', 'cream', 'gel'];
      result.dosage_forms = forms.filter(form => lowerLine.includes(form));
    }
    
    if (lowerLine.includes('mg') || lowerLine.includes('mcg')) {
      const strengthMatch = line.match(/(\d+(?:\.\d+)?\s*(?:mg|mcg|g))/gi);
      if (strengthMatch) result.typical_strengths = strengthMatch;
    }
  }

  return result;
}

module.exports = router;
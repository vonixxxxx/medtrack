import { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Accept multiple possible payload shapes from various frontends
    const rawInput =
      (req.body &&
        (req.body.medication ??
          req.body.medication_name ??
          req.body.name ??
          req.body.query ??
          req.body.text ??
          req.body.term)) ??
      '';
    const medication = typeof rawInput === 'string' ? rawInput : String(rawInput || '').trim();
    console.log('Validating medication:', medication);

    // Use the new comprehensive medication matching service
    // Note: These need to be converted to work in serverless context
    const { validateMedication } = require('../../utils/src/services/medicationMatchingService');
    const { callBioGPTProduction } = require('../../utils/biogptClient');

    // Inject BioGPT caller
    const callBioGPT = async (prompt: string) => {
      try {
        return await callBioGPTProduction(prompt);
      } catch (error: any) {
        console.error('BioGPT call failed:', error.message);
        // Return a default response if BioGPT fails (don't block validation)
        return {
          is_medication: true,
          confidence: 0.5,
          drug_class: null,
          is_generic: true,
          is_brand: false,
        };
      }
    };

    const result = await validateMedication(medication, { callBioGPTFn: callBioGPT });

    if (result.found) {
      // Success - return verified medication
      // Map result to expected frontend format
      return res.json({
        success: true,
        found: true,
        data: {
          generic_name: result.name,
          name: result.name,
          brand_names: [],
          drug_class: result.bio?.drug_class || 'Unknown',
          dosage_forms: ['tablet'],
          typical_strengths: [],
          confidence: result.score,
          alternatives: [],
          original_input: medication,
          source: result.source,
          rxcui: result.rxcui || null,
          bio: result.bio,
        },
      });
    } else {
      // No match found - return proper no-match response
      return res.json({
        success: false,
        found: false,
        error: result.message || `No medication found for "${medication}"`,
        reason: result.reason,
        suggestions: result.suggestions || [],
        original_input: medication,
      });
    }
  } catch (error: any) {
    console.error('Medication validation error:', error);
    res.status(500).json({
      success: false,
      found: false,
      error: 'Failed to validate medication',
      details: error.message,
    });
  }
}

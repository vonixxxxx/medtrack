const { getPolypharmacySideEffects } = require('../services/polypharmacySideEffectsService');

/**
 * Polypharmacy Side Effects Controller
 * Based on Confir-Med API endpoint
 */

const getPolySE = async (req, res) => {
  try {
    const { drug_1, drug_2 } = req.query;

    if (!drug_1 || !drug_2) {
      return res.status(400).json({ error: 'Both drug_1 and drug_2 parameters are required' });
    }

    const sideEffects = await getPolypharmacySideEffects(drug_1, drug_2);

    res.json({
      drug_1,
      drug_2,
      side_effects: sideEffects
    });
  } catch (error) {
    console.error('Error getting polypharmacy side effects:', error);
    res.status(500).json({ error: 'Failed to get polypharmacy side effects' });
  }
};

module.exports = {
  getPolySE
};




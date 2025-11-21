const { getMonopharmacySideEffects } = require('../services/monopharmacySideEffectsService');

/**
 * Monopharmacy Side Effects Controller
 * Based on Confir-Med API endpoint
 */

const getMonoSE = async (req, res) => {
  try {
    const { drug_name } = req.query;

    if (!drug_name) {
      return res.status(400).json({ error: 'drug_name parameter is required' });
    }

    const sideEffects = await getMonopharmacySideEffects(drug_name);

    res.json({
      drug_name,
      side_effects: sideEffects
    });
  } catch (error) {
    console.error('Error getting monopharmacy side effects:', error);
    res.status(500).json({ error: 'Failed to get monopharmacy side effects' });
  }
};

module.exports = {
  getMonoSE
};




const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

/**
 * Polypharmacy Side Effects Service
 * Based on Confir-Med: Get side effects for drug combinations
 */

// Comprehensive polypharmacy side effects database
// In production, integrate with external API or comprehensive database
const POLYPHARMACY_SIDE_EFFECTS = {
  'warfarin-aspirin': [
    'Increased bleeding risk', 'Bruising', 'Gastrointestinal bleeding',
    'Prolonged bleeding time', 'Hemorrhage', 'Easy bruising'
  ],
  'warfarin-ibuprofen': [
    'Increased bleeding risk', 'Gastrointestinal bleeding', 'Stomach ulcers',
    'Kidney problems', 'Bruising', 'Hemorrhage'
  ],
  'aspirin-ibuprofen': [
    'Reduced aspirin cardioprotection', 'Increased GI bleeding risk',
    'Stomach irritation', 'Ulcers'
  ],
  'metformin-alcohol': [
    'Lactic acidosis', 'Increased risk of hypoglycemia', 'Nausea',
    'Vomiting', 'Abdominal pain'
  ],
  'lisinopril-potassium': [
    'Hyperkalemia', 'Irregular heartbeat', 'Muscle weakness',
    'Numbness or tingling'
  ],
  'atorvastatin-grapefruit': [
    'Increased statin levels', 'Muscle pain', 'Muscle weakness',
    'Liver problems', 'Rhabdomyolysis risk'
  ],
  'omeprazole-clopidogrel': [
    'Reduced clopidogrel effectiveness', 'Increased cardiovascular risk',
    'Blood clotting issues'
  ],
  'aspirin-metformin': [
    'Increased risk of lactic acidosis', 'Kidney problems',
    'Bleeding risk'
  ],
  'ibuprofen-metformin': [
    'Increased lactic acidosis risk', 'Kidney problems',
    'GI bleeding risk'
  ]
};

/**
 * Normalize medication names for lookup
 */
function normalizeMedicationName(name) {
  if (!name) return '';
  return name.toLowerCase().trim().replace(/[^a-z0-9]/g, '');
}

/**
 * Get polypharmacy side effects for two medications
 */
async function getPolypharmacySideEffects(drug1, drug2) {
  try {
    const normalized1 = normalizeMedicationName(drug1);
    const normalized2 = normalizeMedicationName(drug2);
    
    // Check both combinations
    const key1 = `${normalized1}-${normalized2}`;
    const key2 = `${normalized2}-${normalized1}`;
    
    // Check in static database
    const sideEffects = POLYPHARMACY_SIDE_EFFECTS[key1] || POLYPHARMACY_SIDE_EFFECTS[key2];
    if (sideEffects) {
      return sideEffects;
    }

    // Check in database for drug interactions
    const interaction = await prisma.drugInteraction.findFirst({
      where: {
        OR: [
          {
            medication1: {
              OR: [
                { name: { contains: drug1, mode: 'insensitive' } },
                { genericName: { contains: drug1, mode: 'insensitive' } }
              ]
            },
            medication2: {
              OR: [
                { name: { contains: drug2, mode: 'insensitive' } },
                { genericName: { contains: drug2, mode: 'insensitive' } }
              ]
            }
          },
          {
            medication1: {
              OR: [
                { name: { contains: drug2, mode: 'insensitive' } },
                { genericName: { contains: drug2, mode: 'insensitive' } }
              ]
            },
            medication2: {
              OR: [
                { name: { contains: drug1, mode: 'insensitive' } },
                { genericName: { contains: drug1, mode: 'insensitive' } }
              ]
            }
          }
        ]
      }
    });

    if (interaction) {
      // Extract side effects from interaction description
      const sideEffectsList = [];
      if (interaction.description) {
        sideEffectsList.push(interaction.description);
      }
      if (interaction.clinicalSignificance) {
        sideEffectsList.push(interaction.clinicalSignificance);
      }
      return sideEffectsList;
    }

    return [];
  } catch (error) {
    console.error('Error getting polypharmacy side effects:', error);
    return [];
  }
}

/**
 * Get all polypharmacy side effects for multiple medications
 */
async function getAllPolypharmacySideEffects(medications) {
  const allSideEffects = [];
  const checkedPairs = new Set();

  for (let i = 0; i < medications.length; i++) {
    for (let j = i + 1; j < medications.length; j++) {
      const med1 = medications[i];
      const med2 = medications[j];
      
      const key = `${med1.id || med1.name}-${med2.id || med2.name}`;
      if (checkedPairs.has(key)) continue;
      checkedPairs.add(key);

      const sideEffects = await getPolypharmacySideEffects(
        med1.name || med1.genericName,
        med2.name || med2.genericName
      );

      if (sideEffects.length > 0) {
        allSideEffects.push({
          medication1: med1.name || med1.genericName,
          medication2: med2.name || med2.genericName,
          medication1Id: med1.id,
          medication2Id: med2.id,
          sideEffects
        });
      }
    }
  }

  return allSideEffects;
}

module.exports = {
  getPolypharmacySideEffects,
  getAllPolypharmacySideEffects,
  POLYPHARMACY_SIDE_EFFECTS,
  normalizeMedicationName
};




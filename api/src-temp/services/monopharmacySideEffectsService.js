const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

/**
 * Monopharmacy Side Effects Service
 * Based on Confir-Med: Get side effects for a single medication
 */

// Comprehensive monopharmacy side effects database
// In production, integrate with external API or comprehensive database
const MONOPHARMACY_SIDE_EFFECTS = {
  'ibuprofen': [
    'Nausea', 'Vomiting', 'Diarrhea', 'Constipation', 'Abdominal pain',
    'Headache', 'Dizziness', 'Drowsiness', 'Rash', 'Itching',
    'Heartburn', 'Stomach ulcers', 'Increased blood pressure',
    'Kidney problems', 'Liver problems'
  ],
  'aspirin': [
    'Nausea', 'Vomiting', 'Stomach pain', 'Heartburn', 'Stomach ulcers',
    'Bleeding', 'Bruising', 'Ringing in ears', 'Dizziness', 'Headache',
    'Allergic reactions', 'Rash', 'Hives', 'Swelling'
  ],
  'acetaminophen': [
    'Nausea', 'Vomiting', 'Stomach pain', 'Loss of appetite',
    'Dark urine', 'Yellowing of skin or eyes', 'Liver damage',
    'Allergic reactions', 'Rash', 'Itching'
  ],
  'metformin': [
    'Nausea', 'Vomiting', 'Diarrhea', 'Stomach upset', 'Metallic taste',
    'Weakness', 'Muscle pain', 'Lactic acidosis', 'Low blood sugar',
    'Vitamin B12 deficiency'
  ],
  'lisinopril': [
    'Dizziness', 'Headache', 'Cough', 'Fatigue', 'Nausea',
    'Diarrhea', 'Low blood pressure', 'High potassium', 'Kidney problems',
    'Allergic reactions', 'Rash', 'Swelling of face or throat'
  ],
  'atorvastatin': [
    'Muscle pain', 'Muscle weakness', 'Joint pain', 'Headache',
    'Nausea', 'Diarrhea', 'Constipation', 'Liver problems',
    'Memory problems', 'Confusion', 'Diabetes risk'
  ],
  'amlodipine': [
    'Dizziness', 'Swelling of ankles or feet', 'Flushing', 'Headache',
    'Fatigue', 'Nausea', 'Stomach pain', 'Palpitations',
    'Low blood pressure', 'Gum swelling'
  ],
  'omeprazole': [
    'Headache', 'Diarrhea', 'Stomach pain', 'Nausea', 'Vomiting',
    'Gas', 'Constipation', 'Vitamin B12 deficiency', 'Bone fractures',
    'Kidney problems', 'Low magnesium'
  ],
  'fish oil': [
    'Fishy aftertaste', 'Bad breath', 'Nausea', 'Diarrhea',
    'Stomach upset', 'Bleeding risk', 'Allergic reactions'
  ],
  'alaxan': [
    'Nausea', 'Stomach pain', 'Dizziness', 'Headache', 'Drowsiness'
  ],
  'bactidol': [
    'Nausea', 'Vomiting', 'Diarrhea', 'Stomach pain', 'Allergic reactions'
  ],
  'bioflu': [
    'Drowsiness', 'Dizziness', 'Dry mouth', 'Nausea', 'Headache'
  ],
  'biogesic': [
    'Nausea', 'Stomach pain', 'Liver problems', 'Allergic reactions'
  ],
  'dayzinc': [
    'Nausea', 'Stomach upset', 'Metallic taste', 'Vomiting'
  ],
  'kremil s': [
    'Constipation', 'Diarrhea', 'Stomach upset', 'Nausea'
  ],
  'medicol': [
    'Nausea', 'Stomach pain', 'Dizziness', 'Headache'
  ],
  'neozep': [
    'Drowsiness', 'Dry mouth', 'Dizziness', 'Nausea', 'Headache'
  ]
};

/**
 * Normalize medication name for lookup
 */
function normalizeMedicationName(name) {
  if (!name) return '';
  return name.toLowerCase().trim().replace(/[^a-z0-9]/g, '');
}

/**
 * Get monopharmacy side effects for a single medication
 */
async function getMonopharmacySideEffects(drugName) {
  try {
    const normalizedName = normalizeMedicationName(drugName);
    
    // Check in static database
    const sideEffects = MONOPHARMACY_SIDE_EFFECTS[normalizedName];
    if (sideEffects) {
      return sideEffects;
    }

    // Check in database (if we have a side effects table)
    // For now, return empty array if not found
    return [];
  } catch (error) {
    console.error('Error getting monopharmacy side effects:', error);
    return [];
  }
}

/**
 * Get side effects from database (if stored)
 */
async function getSideEffectsFromDB(drugName) {
  try {
    // Check if we have side effects stored in MedicationSideEffect table
    const sideEffects = await prisma.medicationSideEffect.findMany({
      where: {
        medication: {
          OR: [
            { name: { contains: drugName, mode: 'insensitive' } },
            { genericName: { contains: drugName, mode: 'insensitive' } }
          ]
        }
      },
      select: {
        symptom: true,
        severity: true
      },
      distinct: ['symptom']
    });

    return sideEffects.map(se => se.symptom);
  } catch (error) {
    console.error('Error getting side effects from DB:', error);
    return [];
  }
}

module.exports = {
  getMonopharmacySideEffects,
  getSideEffectsFromDB,
  MONOPHARMACY_SIDE_EFFECTS,
  normalizeMedicationName
};




const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

/**
 * Enhanced Drug Interaction Service
 * Based on Confir-Med features with comprehensive interaction checking
 */

// Comprehensive drug interaction database
// In production, use external API (DrugBank, RxNorm, etc.)
const INTERACTION_DATABASE = {
  // Major interactions
  'warfarin-aspirin': {
    type: 'major',
    severity: 'severe',
    description: 'Increased risk of bleeding',
    clinicalSignificance: 'Concurrent use significantly increases bleeding risk. Both medications affect platelet function and coagulation.',
    management: 'Monitor INR closely, consider alternative antiplatelet therapy. Avoid concurrent use if possible.',
    onset: 'immediate',
    documentation: 'Well-documented interaction with multiple case reports'
  },
  'warfarin-ibuprofen': {
    type: 'major',
    severity: 'severe',
    description: 'Increased risk of bleeding and GI complications',
    clinicalSignificance: 'NSAIDs increase bleeding risk when combined with warfarin. Also increases risk of GI bleeding.',
    management: 'Avoid concurrent use or monitor closely. Consider COX-2 selective inhibitors if NSAID needed.',
    onset: 'within days',
    documentation: 'Well-documented interaction'
  },
  'warfarin-metronidazole': {
    type: 'major',
    severity: 'severe',
    description: 'Increased warfarin effect, risk of bleeding',
    clinicalSignificance: 'Metronidazole inhibits warfarin metabolism, leading to increased anticoagulation.',
    management: 'Monitor INR frequently. May need to reduce warfarin dose by 30-50%.',
    onset: 'within days',
    documentation: 'Well-documented interaction'
  },
  'metformin-alcohol': {
    type: 'moderate',
    severity: 'moderate',
    description: 'Increased risk of lactic acidosis',
    clinicalSignificance: 'Alcohol can increase risk of lactic acidosis with metformin, especially with excessive consumption.',
    management: 'Limit alcohol consumption. Monitor for symptoms of lactic acidosis (nausea, vomiting, abdominal pain).',
    onset: 'variable',
    documentation: 'Case reports and theoretical risk'
  },
  'lisinopril-potassium': {
    type: 'moderate',
    severity: 'moderate',
    description: 'Risk of hyperkalemia',
    clinicalSignificance: 'ACE inhibitors can cause hyperkalemia, especially when combined with potassium supplements or potassium-sparing diuretics.',
    management: 'Monitor serum potassium levels. Avoid potassium supplements unless hypokalemic.',
    onset: 'within weeks',
    documentation: 'Well-documented interaction'
  },
  'atorvastatin-grapefruit': {
    type: 'moderate',
    severity: 'moderate',
    description: 'Increased statin levels, risk of myopathy',
    clinicalSignificance: 'Grapefruit juice inhibits CYP3A4, increasing atorvastatin levels and risk of muscle toxicity.',
    management: 'Avoid grapefruit juice. Monitor for muscle pain or weakness.',
    onset: 'within days',
    documentation: 'Well-documented interaction'
  },
  'aspirin-ibuprofen': {
    type: 'moderate',
    severity: 'moderate',
    description: 'Reduced aspirin cardioprotective effect',
    clinicalSignificance: 'Ibuprofen can interfere with aspirin\'s antiplatelet effect when taken together.',
    management: 'Take aspirin at least 2 hours before or 8 hours after ibuprofen if cardioprotection is needed.',
    onset: 'immediate',
    documentation: 'Clinical studies'
  },
  'omeprazole-clopidogrel': {
    type: 'moderate',
    severity: 'moderate',
    description: 'Reduced clopidogrel effectiveness',
    clinicalSignificance: 'PPIs can reduce clopidogrel activation, potentially reducing its antiplatelet effect.',
    management: 'Consider alternative PPI (pantoprazole) or H2 blocker. Monitor for cardiovascular events.',
    onset: 'within days',
    documentation: 'Mixed evidence, some studies show risk'
  }
};

/**
 * Normalize medication name for matching
 */
function normalizeMedicationName(name) {
  if (!name) return '';
  return name.toLowerCase().trim().replace(/[^a-z0-9]/g, '');
}

/**
 * Get medication generic name if available
 */
async function getMedicationInfo(medicationId) {
  const medication = await prisma.medication.findUnique({
    where: { id: medicationId },
    select: {
      id: true,
      name: true,
      genericName: true,
      drugClass: true
    }
  });
  return medication;
}

/**
 * Check interactions between medications
 */
async function checkInteractions(medications) {
  const interactions = [];
  const checkedPairs = new Set();

  // Check all pairs
  for (let i = 0; i < medications.length; i++) {
    for (let j = i + 1; j < medications.length; j++) {
      const med1 = medications[i];
      const med2 = medications[j];
      
      // Create normalized keys for lookup
      const med1Name = normalizeMedicationName(med1.name || med1.genericName || '');
      const med2Name = normalizeMedicationName(med2.name || med2.genericName || '');
      
      const key1 = `${med1Name}-${med2Name}`;
      const key2 = `${med2Name}-${med1Name}`;
      
      // Skip if already checked
      if (checkedPairs.has(key1) || checkedPairs.has(key2)) continue;
      checkedPairs.add(key1);
      checkedPairs.add(key2);

      // Check in static database
      const staticInteraction = INTERACTION_DATABASE[key1] || INTERACTION_DATABASE[key2];
      
      if (staticInteraction) {
        interactions.push({
          medication1: med1.name || med1.genericName,
          medication2: med2.name || med2.genericName,
          medication1Id: med1.id,
          medication2Id: med2.id,
          ...staticInteraction
        });
        continue;
      }

      // Check in database
      const dbInteraction = await prisma.drugInteraction.findFirst({
        where: {
          OR: [
            {
              medication1Id: med1.id,
              medication2Id: med2.id
            },
            {
              medication1Id: med2.id,
              medication2Id: med1.id
            }
          ]
        }
      });

      if (dbInteraction) {
        interactions.push({
          medication1: med1.name || med1.genericName,
          medication2: med2.name || med2.genericName,
          medication1Id: med1.id,
          medication2Id: med2.id,
          interactionType: dbInteraction.interactionType,
          severity: dbInteraction.severity,
          description: dbInteraction.description,
          clinicalSignificance: dbInteraction.clinicalSignificance,
          management: dbInteraction.management,
          source: dbInteraction.source,
          verified: dbInteraction.verified
        });
      }
    }
  }

  return interactions;
}

/**
 * Check interactions for newly recognized pill
 */
async function checkPillInteractions(recognizedPill, userId, patientId) {
  try {
    // Get user's current medications
    const currentMedications = await prisma.medication.findMany({
      where: {
        userId,
        patientId: patientId || undefined,
        status: 'active'
      },
      select: {
        id: true,
        name: true,
        genericName: true,
        drugClass: true
      }
    });

    // Create medication object for recognized pill
    const recognizedMed = {
      id: 'recognized-' + Date.now(),
      name: recognizedPill.medicationName,
      genericName: recognizedPill.genericName
    };

    // Check interactions with all current medications
    const allMeds = [recognizedMed, ...currentMedications];
    const interactions = await checkInteractions(allMeds);

    // Filter to only interactions involving the recognized pill
    return interactions.filter(interaction => 
      interaction.medication1Id === recognizedMed.id ||
      interaction.medication2Id === recognizedMed.id
    );
  } catch (error) {
    console.error('Error checking pill interactions:', error);
    return [];
  }
}

/**
 * Get interaction severity level
 */
function getSeverityLevel(severity) {
  const levels = {
    'severe': 3,
    'moderate': 2,
    'mild': 1
  };
  return levels[severity?.toLowerCase()] || 0;
}

/**
 * Sort interactions by severity
 */
function sortInteractionsBySeverity(interactions) {
  return interactions.sort((a, b) => {
    const severityA = getSeverityLevel(a.severity);
    const severityB = getSeverityLevel(b.severity);
    if (severityB !== severityA) {
      return severityB - severityA;
    }
    // If same severity, sort by type (major > moderate > minor)
    const typeOrder = { 'major': 3, 'moderate': 2, 'minor': 1 };
    return (typeOrder[b.type] || 0) - (typeOrder[a.type] || 0);
  });
}

module.exports = {
  checkInteractions,
  checkPillInteractions,
  sortInteractionsBySeverity,
  getSeverityLevel,
  INTERACTION_DATABASE,
  normalizeMedicationName
};




const prisma = require('../db/prisma');
const { checkPillInteractions } = require('../services/drugInteractionService');

/**
 * Enhanced Medication Tracking Controller
 * Based on Confir-Med features for comprehensive medication logging
 */

/**
 * Add medication from recognized pill
 */
const addMedicationFromPill = async (req, res) => {
  try {
    const { userId, patientId, recognitionId, medicationData } = req.body;

    if (!userId || !recognitionId) {
      return res.status(400).json({ error: 'User ID and recognition ID are required' });
    }

    // Get recognition record
    const recognition = await prisma.pillRecognition.findUnique({
      where: { id: recognitionId }
    });

    if (!recognition || !recognition.recognized) {
      return res.status(400).json({ error: 'Invalid or unrecognized pill' });
    }

    // Check for interactions before adding
    const interactions = await checkPillInteractions(
      {
        medicationName: recognition.medicationName,
        genericName: recognition.medicationName
      },
      userId,
      patientId
    );

    // Create medication from recognition
    const medication = await prisma.medication.create({
      data: {
        userId,
        patientId: patientId || null,
        name: recognition.medicationName,
        genericName: recognition.medicationName,
        startDate: new Date(),
        dosage: medicationData?.dosage || 'As directed',
        unit: medicationData?.unit || 'tablet',
        frequency: medicationData?.frequency || 'daily',
        route: medicationData?.route || 'oral',
        strength: medicationData?.strength || null,
        drugClass: medicationData?.drugClass || null,
        ndcCode: medicationData?.ndcCode || null,
        rxnormCode: medicationData?.rxnormCode || null,
        status: 'active',
        reminderEnabled: true
      }
    });

    // Update recognition to link with medication
    await prisma.pillRecognition.update({
      where: { id: recognitionId },
      data: { verified: true }
    });

    res.status(201).json({
      medication,
      interactions: interactions.length > 0 ? interactions : undefined,
      hasInteractions: interactions.length > 0
    });
  } catch (error) {
    console.error('Error adding medication from pill:', error);
    res.status(500).json({ error: 'Failed to add medication from pill' });
  }
};

/**
 * Get user's current medications with interaction warnings
 */
const getMedicationsWithWarnings = async (req, res) => {
  try {
    const { userId, patientId } = req.query;

    if (!userId) {
      return res.status(400).json({ error: 'User ID is required' });
    }

    const medications = await prisma.medication.findMany({
      where: {
        userId,
        patientId: patientId || undefined,
        status: 'active'
      },
      orderBy: { startDate: 'desc' }
    });

    // Check for interactions among current medications
    const { checkInteractions, sortInteractionsBySeverity } = require('../services/drugInteractionService');
    const interactions = await checkInteractions(medications);
    const sortedInteractions = sortInteractionsBySeverity(interactions);

    // Group interactions by medication
    const medicationWarnings = {};
    medications.forEach(med => {
      medicationWarnings[med.id] = sortedInteractions.filter(interaction =>
        interaction.medication1Id === med.id || interaction.medication2Id === med.id
      );
    });

    res.json({
      medications,
      interactions: sortedInteractions,
      medicationWarnings,
      hasInteractions: sortedInteractions.length > 0
    });
  } catch (error) {
    console.error('Error fetching medications with warnings:', error);
    res.status(500).json({ error: 'Failed to fetch medications' });
  }
};

module.exports = {
  addMedicationFromPill,
  getMedicationsWithWarnings
};


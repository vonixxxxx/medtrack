const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
const { checkInteractions: checkInteractionsService, sortInteractionsBySeverity } = require('../services/drugInteractionService');

// Drug interaction database moved to service

const checkInteractions = async (req, res) => {
  try {
    const { medicationIds, medicationNames } = req.body;
    
    if (!medicationIds && !medicationNames) {
      return res.status(400).json({ error: 'Medication IDs or names are required' });
    }

    let medications = [];
    
    if (medicationIds && medicationIds.length > 0) {
      medications = await prisma.medication.findMany({
        where: { id: { in: medicationIds } },
        select: { id: true, name: true, genericName: true }
      });
    } else if (medicationNames && medicationNames.length > 0) {
      medications = medicationNames.map(name => ({ name, genericName: null }));
    }

    if (medications.length < 2) {
      return res.json({ interactions: [], message: 'Need at least 2 medications to check interactions' });
    }

    // Use enhanced service to check interactions
    const interactions = await checkInteractionsService(medications);
    
    // Sort by severity (most severe first)
    const sortedInteractions = sortInteractionsBySeverity(interactions);

    res.json({
      interactions: sortedInteractions,
      checkedCount: medications.length,
      interactionCount: sortedInteractions.length,
      severeCount: sortedInteractions.filter(i => i.severity === 'severe').length,
      moderateCount: sortedInteractions.filter(i => i.severity === 'moderate').length,
      mildCount: sortedInteractions.filter(i => i.severity === 'mild').length
    });
  } catch (error) {
    console.error('Error checking drug interactions:', error);
    res.status(500).json({ error: 'Failed to check drug interactions' });
  }
};

const addInteraction = async (req, res) => {
  try {
    const { medication1Id, medication2Id, interactionType, severity, description, clinicalSignificance, management } = req.body;

    if (!medication1Id || !medication2Id || !interactionType || !description) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const interaction = await prisma.drugInteraction.create({
      data: {
        medication1Id,
        medication2Id,
        interactionType,
        severity: severity || 'moderate',
        description,
        clinicalSignificance: clinicalSignificance || null,
        management: management || null,
        source: 'manual',
        verified: false
      }
    });

    res.status(201).json(interaction);
  } catch (error) {
    console.error('Error adding drug interaction:', error);
    res.status(500).json({ error: 'Failed to add drug interaction' });
  }
};

const getInteractions = async (req, res) => {
  try {
    const { medicationId } = req.params;
    
    const interactions = await prisma.drugInteraction.findMany({
      where: {
        OR: [
          { medication1Id: medicationId },
          { medication2Id: medicationId }
        ]
      },
      orderBy: { severity: 'desc' }
    });

    res.json(interactions);
  } catch (error) {
    console.error('Error fetching drug interactions:', error);
    res.status(500).json({ error: 'Failed to fetch drug interactions' });
  }
};

module.exports = {
  checkInteractions,
  addInteraction,
  getInteractions
};


const prisma = require('../db/prisma');

const getSideEffects = async (req, res) => {
  try {
    const { medicationId, patientId } = req.query;
    const where = {};

    if (medicationId) where.medicationId = medicationId;
    if (patientId) {
      where.medication = {
        patientId: patientId
      };
    }

    const sideEffects = await prisma.medicationSideEffect.findMany({
      where,
      include: {
        medication: {
          select: {
            id: true,
            name: true,
            genericName: true
          }
        }
      },
      orderBy: { onsetDate: 'desc' }
    });

    res.json(sideEffects);
  } catch (error) {
    console.error('Error fetching side effects:', error);
    res.status(500).json({ error: 'Failed to fetch side effects' });
  }
};

const createSideEffect = async (req, res) => {
  try {
    const { medicationId, symptom, severity, onsetDate, resolvedDate, notes } = req.body;

    if (!medicationId || !symptom) {
      return res.status(400).json({ error: 'Medication ID and symptom are required' });
    }

    const sideEffect = await prisma.medicationSideEffect.create({
      data: {
        medicationId,
        symptom,
        severity: severity || null,
        onsetDate: onsetDate ? new Date(onsetDate) : new Date(),
        resolvedDate: resolvedDate ? new Date(resolvedDate) : null,
        notes: notes || null
      },
      include: {
        medication: {
          select: {
            id: true,
            name: true,
            genericName: true
          }
        }
      }
    });

    res.status(201).json(sideEffect);
  } catch (error) {
    console.error('Error creating side effect:', error);
    res.status(500).json({ error: 'Failed to create side effect' });
  }
};

const updateSideEffect = async (req, res) => {
  try {
    const { id } = req.params;
    const { symptom, severity, onsetDate, resolvedDate, notes } = req.body;

    const updateData = {};
    if (symptom !== undefined) updateData.symptom = symptom;
    if (severity !== undefined) updateData.severity = severity;
    if (onsetDate !== undefined) updateData.onsetDate = onsetDate ? new Date(onsetDate) : null;
    if (resolvedDate !== undefined) updateData.resolvedDate = resolvedDate ? new Date(resolvedDate) : null;
    if (notes !== undefined) updateData.notes = notes;

    const sideEffect = await prisma.medicationSideEffect.update({
      where: { id },
      data: updateData,
      include: {
        medication: {
          select: {
            id: true,
            name: true,
            genericName: true
          }
        }
      }
    });

    res.json(sideEffect);
  } catch (error) {
    console.error('Error updating side effect:', error);
    res.status(500).json({ error: 'Failed to update side effect' });
  }
};

const deleteSideEffect = async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.medicationSideEffect.delete({ where: { id } });
    res.json({ message: 'Side effect deleted successfully' });
  } catch (error) {
    console.error('Error deleting side effect:', error);
    res.status(500).json({ error: 'Failed to delete side effect' });
  }
};

module.exports = {
  getSideEffects,
  createSideEffect,
  updateSideEffect,
  deleteSideEffect
};


const prisma = require('../db/prisma');

const getAllergies = async (req, res) => {
  try {
    const { patientId, status } = req.query;
    const where = {};

    if (patientId) where.patientId = patientId;
    if (status) where.status = status;

    const allergies = await prisma.allergy.findMany({
      where,
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      },
      orderBy: { createdAt: 'desc' }
    });

    res.json(allergies);
  } catch (error) {
    console.error('Error fetching allergies:', error);
    res.status(500).json({ error: 'Failed to fetch allergies' });
  }
};

const getAllergy = async (req, res) => {
  try {
    const { id } = req.params;
    const allergy = await prisma.allergy.findUnique({
      where: { id },
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      }
    });

    if (!allergy) {
      return res.status(404).json({ error: 'Allergy not found' });
    }

    res.json(allergy);
  } catch (error) {
    console.error('Error fetching allergy:', error);
    res.status(500).json({ error: 'Failed to fetch allergy' });
  }
};

const createAllergy = async (req, res) => {
  try {
    const {
      patientId,
      allergen,
      allergenType,
      reaction,
      severity,
      onsetDate,
      status,
      notes
    } = req.body;

    if (!patientId || !allergen) {
      return res.status(400).json({ error: 'Patient ID and allergen are required' });
    }

    const allergy = await prisma.allergy.create({
      data: {
        patientId,
        allergen,
        allergenType: allergenType || null,
        reaction: reaction || null,
        severity: severity || null,
        onsetDate: onsetDate ? new Date(onsetDate) : null,
        status: status || 'active',
        notes: notes || null,
        createdBy: req.user?.id || null
      },
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      }
    });

    res.status(201).json(allergy);
  } catch (error) {
    console.error('Error creating allergy:', error);
    res.status(500).json({ error: 'Failed to create allergy' });
  }
};

const updateAllergy = async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = {};

    const fields = ['allergen', 'allergenType', 'reaction', 'severity', 'onsetDate', 'status', 'notes'];
    fields.forEach(field => {
      if (req.body[field] !== undefined) {
        if (field === 'onsetDate') {
          updateData[field] = req.body[field] ? new Date(req.body[field]) : null;
        } else {
          updateData[field] = req.body[field];
        }
      }
    });

    const allergy = await prisma.allergy.update({
      where: { id },
      data: updateData,
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      }
    });

    res.json(allergy);
  } catch (error) {
    console.error('Error updating allergy:', error);
    res.status(500).json({ error: 'Failed to update allergy' });
  }
};

const deleteAllergy = async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.allergy.delete({ where: { id } });
    res.json({ message: 'Allergy deleted successfully' });
  } catch (error) {
    console.error('Error deleting allergy:', error);
    res.status(500).json({ error: 'Failed to delete allergy' });
  }
};

module.exports = {
  getAllergies,
  getAllergy,
  createAllergy,
  updateAllergy,
  deleteAllergy
};


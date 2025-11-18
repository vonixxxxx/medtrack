const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const getEncounters = async (req, res) => {
  try {
    const { patientId, providerId, startDate, endDate, status } = req.query;
    const where = {};

    if (patientId) where.patientId = patientId;
    if (providerId) where.providerId = providerId;
    if (status) where.status = status;
    if (startDate || endDate) {
      where.encounterDate = {};
      if (startDate) where.encounterDate.gte = new Date(startDate);
      if (endDate) where.encounterDate.lte = new Date(endDate);
    }

    const encounters = await prisma.encounter.findMany({
      where,
      include: {
        patient: {
          include: {
            user: {
              select: { id: true, name: true, email: true }
            }
          }
        },
        provider: {
          select: { id: true, name: true, email: true }
        },
        soapNotes: true,
        vitals: true
      },
      orderBy: { encounterDate: 'desc' }
    });

    res.json(encounters);
  } catch (error) {
    console.error('Error fetching encounters:', error);
    res.status(500).json({ error: 'Failed to fetch encounters' });
  }
};

const getEncounter = async (req, res) => {
  try {
    const { id } = req.params;
    const encounter = await prisma.encounter.findUnique({
      where: { id },
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        },
        provider: { select: { id: true, name: true, email: true } },
        soapNotes: true,
        vitals: true,
        documents: true,
        charges: true,
        problems: true
      }
    });

    if (!encounter) {
      return res.status(404).json({ error: 'Encounter not found' });
    }

    res.json(encounter);
  } catch (error) {
    console.error('Error fetching encounter:', error);
    res.status(500).json({ error: 'Failed to fetch encounter' });
  }
};

const createEncounter = async (req, res) => {
  try {
    const {
      patientId,
      providerId,
      facilityId,
      encounterDate,
      encounterTime,
      encounterType,
      reason,
      status,
      priority
    } = req.body;

    if (!patientId || !encounterDate) {
      return res.status(400).json({ error: 'Patient ID and encounter date are required' });
    }

    const encounter = await prisma.encounter.create({
      data: {
        patientId,
        providerId: providerId || null,
        facilityId: facilityId || null,
        encounterDate: new Date(encounterDate),
        encounterTime: encounterTime ? new Date(`1970-01-01T${encounterTime}`) : null,
        encounterType: encounterType || 'office',
        reason: reason || null,
        status: status || 'finished',
        priority: priority || null,
        createdBy: req.user?.id || null
      },
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        },
        provider: { select: { id: true, name: true, email: true } }
      }
    });

    res.status(201).json(encounter);
  } catch (error) {
    console.error('Error creating encounter:', error);
    res.status(500).json({ error: 'Failed to create encounter' });
  }
};

const updateEncounter = async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = {};

    const fields = ['providerId', 'facilityId', 'encounterDate', 'encounterTime', 'encounterType', 'reason', 'status', 'priority'];
    fields.forEach(field => {
      if (req.body[field] !== undefined) {
        if (field === 'encounterDate') {
          updateData[field] = new Date(req.body[field]);
        } else if (field === 'encounterTime') {
          updateData[field] = new Date(`1970-01-01T${req.body[field]}`);
        } else {
          updateData[field] = req.body[field];
        }
      }
    });

    const encounter = await prisma.encounter.update({
      where: { id },
      data: updateData,
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        },
        provider: { select: { id: true, name: true, email: true } }
      }
    });

    res.json(encounter);
  } catch (error) {
    console.error('Error updating encounter:', error);
    res.status(500).json({ error: 'Failed to update encounter' });
  }
};

const deleteEncounter = async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.encounter.delete({ where: { id } });
    res.json({ message: 'Encounter deleted successfully' });
  } catch (error) {
    console.error('Error deleting encounter:', error);
    res.status(500).json({ error: 'Failed to delete encounter' });
  }
};

module.exports = {
  getEncounters,
  getEncounter,
  createEncounter,
  updateEncounter,
  deleteEncounter
};




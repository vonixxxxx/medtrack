const prisma = require('../db/prisma');

const getPrescriptions = async (req, res) => {
  try {
    const { patientId, status } = req.query;
    const where = {};

    if (patientId) where.patientId = patientId;
    if (status) where.status = status;

    const prescriptions = await prisma.prescription.findMany({
      where,
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      },
      orderBy: { datePrescribed: 'desc' }
    });

    res.json(prescriptions);
  } catch (error) {
    console.error('Error fetching prescriptions:', error);
    res.status(500).json({ error: 'Failed to fetch prescriptions' });
  }
};

const getPrescription = async (req, res) => {
  try {
    const { id } = req.params;
    const prescription = await prisma.prescription.findUnique({
      where: { id },
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      }
    });

    if (!prescription) {
      return res.status(404).json({ error: 'Prescription not found' });
    }

    res.json(prescription);
  } catch (error) {
    console.error('Error fetching prescription:', error);
    res.status(500).json({ error: 'Failed to fetch prescription' });
  }
};

const createPrescription = async (req, res) => {
  try {
    const {
      patientId,
      encounterId,
      providerId,
      medicationName,
      ndcCode,
      rxnormCode,
      dosage,
      unit,
      route,
      frequency,
      quantity,
      refills,
      startDate,
      endDate,
      status,
      instructions,
      pharmacyId
    } = req.body;

    if (!patientId || !medicationName || !startDate) {
      return res.status(400).json({ error: 'Patient ID, medication name, and start date are required' });
    }

    const prescription = await prisma.prescription.create({
      data: {
        patientId,
        encounterId: encounterId || null,
        providerId: providerId || null,
        medicationName,
        ndcCode: ndcCode || null,
        rxnormCode: rxnormCode || null,
        dosage: dosage || null,
        unit: unit || null,
        route: route || null,
        frequency: frequency || null,
        quantity: quantity || null,
        refills: refills || 0,
        startDate: new Date(startDate),
        endDate: endDate ? new Date(endDate) : null,
        status: status || 'active',
        instructions: instructions || null,
        pharmacyId: pharmacyId || null,
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

    res.status(201).json(prescription);
  } catch (error) {
    console.error('Error creating prescription:', error);
    res.status(500).json({ error: 'Failed to create prescription' });
  }
};

const updatePrescription = async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = {};

    const fields = ['medicationName', 'ndcCode', 'rxnormCode', 'dosage', 'unit', 'route', 'frequency', 'quantity', 'refills', 'startDate', 'endDate', 'status', 'instructions', 'pharmacyId'];
    fields.forEach(field => {
      if (req.body[field] !== undefined) {
        if (field === 'startDate' || field === 'endDate') {
          updateData[field] = req.body[field] ? new Date(req.body[field]) : null;
        } else {
          updateData[field] = req.body[field];
        }
      }
    });

    const prescription = await prisma.prescription.update({
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

    res.json(prescription);
  } catch (error) {
    console.error('Error updating prescription:', error);
    res.status(500).json({ error: 'Failed to update prescription' });
  }
};

const deletePrescription = async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.prescription.delete({ where: { id } });
    res.json({ message: 'Prescription deleted successfully' });
  } catch (error) {
    console.error('Error deleting prescription:', error);
    res.status(500).json({ error: 'Failed to delete prescription' });
  }
};

module.exports = {
  getPrescriptions,
  getPrescription,
  createPrescription,
  updatePrescription,
  deletePrescription
};


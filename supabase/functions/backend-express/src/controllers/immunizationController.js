const prisma = require('../db/prisma');

const getImmunizations = async (req, res) => {
  try {
    const { patientId } = req.query;
    const where = patientId ? { patientId } : {};

    const immunizations = await prisma.immunization.findMany({
      where,
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      },
      orderBy: { administrationDate: 'desc' }
    });

    res.json(immunizations);
  } catch (error) {
    console.error('Error fetching immunizations:', error);
    res.status(500).json({ error: 'Failed to fetch immunizations' });
  }
};

const getImmunization = async (req, res) => {
  try {
    const { id } = req.params;
    const immunization = await prisma.immunization.findUnique({
      where: { id },
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      }
    });

    if (!immunization) {
      return res.status(404).json({ error: 'Immunization not found' });
    }

    res.json(immunization);
  } catch (error) {
    console.error('Error fetching immunization:', error);
    res.status(500).json({ error: 'Failed to fetch immunization' });
  }
};

const createImmunization = async (req, res) => {
  try {
    const {
      patientId,
      vaccineName,
      vaccineCode,
      administrationDate,
      lotNumber,
      manufacturer,
      route,
      site,
      dose,
      provider,
      notes
    } = req.body;

    if (!patientId || !vaccineName || !administrationDate) {
      return res.status(400).json({ error: 'Patient ID, vaccine name, and administration date are required' });
    }

    const immunization = await prisma.immunization.create({
      data: {
        patientId,
        vaccineName,
        vaccineCode: vaccineCode || null,
        administrationDate: new Date(administrationDate),
        lotNumber: lotNumber || null,
        manufacturer: manufacturer || null,
        route: route || null,
        site: site || null,
        dose: dose || null,
        provider: provider || null,
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

    res.status(201).json(immunization);
  } catch (error) {
    console.error('Error creating immunization:', error);
    res.status(500).json({ error: 'Failed to create immunization' });
  }
};

const updateImmunization = async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = {};

    const fields = ['vaccineName', 'vaccineCode', 'administrationDate', 'lotNumber', 'manufacturer', 'route', 'site', 'dose', 'provider', 'notes'];
    fields.forEach(field => {
      if (req.body[field] !== undefined) {
        if (field === 'administrationDate') {
          updateData[field] = new Date(req.body[field]);
        } else {
          updateData[field] = req.body[field];
        }
      }
    });

    const immunization = await prisma.immunization.update({
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

    res.json(immunization);
  } catch (error) {
    console.error('Error updating immunization:', error);
    res.status(500).json({ error: 'Failed to update immunization' });
  }
};

const deleteImmunization = async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.immunization.delete({ where: { id } });
    res.json({ message: 'Immunization deleted successfully' });
  } catch (error) {
    console.error('Error deleting immunization:', error);
    res.status(500).json({ error: 'Failed to delete immunization' });
  }
};

module.exports = {
  getImmunizations,
  getImmunization,
  createImmunization,
  updateImmunization,
  deleteImmunization
};


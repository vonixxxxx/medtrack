const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const getSoapNotes = async (req, res) => {
  try {
    const { encounterId } = req.query;
    const where = encounterId ? { encounterId } : {};

    const soapNotes = await prisma.soapNote.findMany({
      where,
      include: {
        encounter: {
          include: {
            patient: {
              include: {
                user: { select: { id: true, name: true, email: true } }
              }
            }
          }
        }
      },
      orderBy: { createdAt: 'desc' }
    });

    res.json(soapNotes);
  } catch (error) {
    console.error('Error fetching SOAP notes:', error);
    res.status(500).json({ error: 'Failed to fetch SOAP notes' });
  }
};

const getSoapNote = async (req, res) => {
  try {
    const { id } = req.params;
    const soapNote = await prisma.soapNote.findUnique({
      where: { id },
      include: {
        encounter: {
          include: {
            patient: {
              include: {
                user: { select: { id: true, name: true, email: true } }
              }
            }
          }
        }
      }
    });

    if (!soapNote) {
      return res.status(404).json({ error: 'SOAP note not found' });
    }

    res.json(soapNote);
  } catch (error) {
    console.error('Error fetching SOAP note:', error);
    res.status(500).json({ error: 'Failed to fetch SOAP note' });
  }
};

const createSoapNote = async (req, res) => {
  try {
    const { encounterId, subjective, objective, assessment, plan } = req.body;

    if (!encounterId) {
      return res.status(400).json({ error: 'Encounter ID is required' });
    }

    const soapNote = await prisma.soapNote.create({
      data: {
        encounterId,
        subjective: subjective || null,
        objective: objective || null,
        assessment: assessment || null,
        plan: plan || null,
        createdBy: req.user?.id || null
      },
      include: {
        encounter: {
          include: {
            patient: {
              include: {
                user: { select: { id: true, name: true, email: true } }
              }
            }
          }
        }
      }
    });

    res.status(201).json(soapNote);
  } catch (error) {
    console.error('Error creating SOAP note:', error);
    res.status(500).json({ error: 'Failed to create SOAP note' });
  }
};

const updateSoapNote = async (req, res) => {
  try {
    const { id } = req.params;
    const { subjective, objective, assessment, plan } = req.body;

    const updateData = {};
    if (subjective !== undefined) updateData.subjective = subjective;
    if (objective !== undefined) updateData.objective = objective;
    if (assessment !== undefined) updateData.assessment = assessment;
    if (plan !== undefined) updateData.plan = plan;
    updateData.updatedBy = req.user?.id || null;

    const soapNote = await prisma.soapNote.update({
      where: { id },
      data: updateData,
      include: {
        encounter: {
          include: {
            patient: {
              include: {
                user: { select: { id: true, name: true, email: true } }
              }
            }
          }
        }
      }
    });

    res.json(soapNote);
  } catch (error) {
    console.error('Error updating SOAP note:', error);
    res.status(500).json({ error: 'Failed to update SOAP note' });
  }
};

const deleteSoapNote = async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.soapNote.delete({ where: { id } });
    res.json({ message: 'SOAP note deleted successfully' });
  } catch (error) {
    console.error('Error deleting SOAP note:', error);
    res.status(500).json({ error: 'Failed to delete SOAP note' });
  }
};

module.exports = {
  getSoapNotes,
  getSoapNote,
  createSoapNote,
  updateSoapNote,
  deleteSoapNote
};




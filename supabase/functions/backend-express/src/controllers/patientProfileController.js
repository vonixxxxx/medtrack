const prisma = require('../db/prisma');

const getPatientProfiles = async (req, res) => {
  try {
    const { userId } = req.query;
    
    if (!userId) {
      return res.status(400).json({ error: 'User ID is required' });
    }

    const profiles = await prisma.patientProfile.findMany({
      where: { userId },
      include: {
        patient: {
          include: {
            user: {
              select: {
                id: true,
                name: true,
                email: true
              }
            }
          }
        }
      },
      orderBy: [
        { isPrimary: 'desc' },
        { createdAt: 'asc' }
      ]
    });

    res.json(profiles);
  } catch (error) {
    console.error('Error fetching patient profiles:', error);
    res.status(500).json({ error: 'Failed to fetch patient profiles' });
  }
};

const createPatientProfile = async (req, res) => {
  try {
    const { userId, patientId, name, relationship, color, avatar, notes } = req.body;

    if (!userId || !patientId || !name) {
      return res.status(400).json({ error: 'User ID, patient ID, and name are required' });
    }

    // If this is the first profile or marked as primary, set isPrimary
    const existingProfiles = await prisma.patientProfile.findMany({
      where: { userId }
    });

    const isPrimary = existingProfiles.length === 0 || req.body.isPrimary === true;

    // If setting as primary, unset others
    if (isPrimary) {
      await prisma.patientProfile.updateMany({
        where: { userId, isPrimary: true },
        data: { isPrimary: false }
      });
    }

    const profile = await prisma.patientProfile.create({
      data: {
        userId,
        patientId,
        name,
        relationship: relationship || 'self',
        isPrimary,
        color: color || null,
        avatar: avatar || null,
        notes: notes || null
      },
      include: {
        patient: {
          include: {
            user: {
              select: {
                id: true,
                name: true,
                email: true
              }
            }
          }
        }
      }
    });

    res.status(201).json(profile);
  } catch (error) {
    console.error('Error creating patient profile:', error);
    res.status(500).json({ error: 'Failed to create patient profile' });
  }
};

const updatePatientProfile = async (req, res) => {
  try {
    const { id } = req.params;
    const { name, relationship, isPrimary, color, avatar, notes } = req.body;

    const updateData = {};
    if (name !== undefined) updateData.name = name;
    if (relationship !== undefined) updateData.relationship = relationship;
    if (color !== undefined) updateData.color = color;
    if (avatar !== undefined) updateData.avatar = avatar;
    if (notes !== undefined) updateData.notes = notes;

    // If setting as primary, unset others
    if (isPrimary === true) {
      const profile = await prisma.patientProfile.findUnique({
        where: { id },
        select: { userId: true }
      });

      if (profile) {
        await prisma.patientProfile.updateMany({
          where: { userId: profile.userId, isPrimary: true },
          data: { isPrimary: false }
        });
        updateData.isPrimary = true;
      }
    } else if (isPrimary === false) {
      updateData.isPrimary = false;
    }

    const updatedProfile = await prisma.patientProfile.update({
      where: { id },
      data: updateData,
      include: {
        patient: {
          include: {
            user: {
              select: {
                id: true,
                name: true,
                email: true
              }
            }
          }
        }
      }
    });

    res.json(updatedProfile);
  } catch (error) {
    console.error('Error updating patient profile:', error);
    res.status(500).json({ error: 'Failed to update patient profile' });
  }
};

const deletePatientProfile = async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.patientProfile.delete({ where: { id } });
    res.json({ message: 'Patient profile deleted successfully' });
  } catch (error) {
    console.error('Error deleting patient profile:', error);
    res.status(500).json({ error: 'Failed to delete patient profile' });
  }
};

module.exports = {
  getPatientProfiles,
  createPatientProfile,
  updatePatientProfile,
  deletePatientProfile
};


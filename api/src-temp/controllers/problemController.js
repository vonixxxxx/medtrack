const prisma = require('../db/prisma');

const getProblems = async (req, res) => {
  try {
    const { patientId, status } = req.query;
    const where = {};

    if (patientId) where.patientId = patientId;
    if (status) where.status = status;

    const problems = await prisma.problem.findMany({
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

    res.json(problems);
  } catch (error) {
    console.error('Error fetching problems:', error);
    res.status(500).json({ error: 'Failed to fetch problems' });
  }
};

const getProblem = async (req, res) => {
  try {
    const { id } = req.params;
    const problem = await prisma.problem.findUnique({
      where: { id },
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      }
    });

    if (!problem) {
      return res.status(404).json({ error: 'Problem not found' });
    }

    res.json(problem);
  } catch (error) {
    console.error('Error fetching problem:', error);
    res.status(500).json({ error: 'Failed to fetch problem' });
  }
};

const createProblem = async (req, res) => {
  try {
    const {
      patientId,
      encounterId,
      title,
      code,
      codeType,
      beginDate,
      endDate,
      status,
      severity,
      notes
    } = req.body;

    if (!patientId || !title) {
      return res.status(400).json({ error: 'Patient ID and title are required' });
    }

    const problem = await prisma.problem.create({
      data: {
        patientId,
        encounterId: encounterId || null,
        title,
        code: code || null,
        codeType: codeType || 'ICD10',
        beginDate: beginDate ? new Date(beginDate) : null,
        endDate: endDate ? new Date(endDate) : null,
        status: status || 'active',
        severity: severity || null,
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

    res.status(201).json(problem);
  } catch (error) {
    console.error('Error creating problem:', error);
    res.status(500).json({ error: 'Failed to create problem' });
  }
};

const updateProblem = async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = {};

    const fields = ['title', 'code', 'codeType', 'beginDate', 'endDate', 'status', 'severity', 'notes'];
    fields.forEach(field => {
      if (req.body[field] !== undefined) {
        if (field === 'beginDate' || field === 'endDate') {
          updateData[field] = req.body[field] ? new Date(req.body[field]) : null;
        } else {
          updateData[field] = req.body[field];
        }
      }
    });

    const problem = await prisma.problem.update({
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

    res.json(problem);
  } catch (error) {
    console.error('Error updating problem:', error);
    res.status(500).json({ error: 'Failed to update problem' });
  }
};

const deleteProblem = async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.problem.delete({ where: { id } });
    res.json({ message: 'Problem deleted successfully' });
  } catch (error) {
    console.error('Error deleting problem:', error);
    res.status(500).json({ error: 'Failed to delete problem' });
  }
};

module.exports = {
  getProblems,
  getProblem,
  createProblem,
  updateProblem,
  deleteProblem
};


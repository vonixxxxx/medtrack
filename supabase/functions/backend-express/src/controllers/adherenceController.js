const prisma = require('../db/prisma');

const getAdherence = async (req, res) => {
  try {
    const { medicationId, patientId, startDate, endDate } = req.query;
    const where = {};

    if (medicationId) where.medicationId = medicationId;
    if (patientId) {
      where.medication = {
        patientId: patientId
      };
    }
    if (startDate || endDate) {
      where.date = {};
      if (startDate) where.date.gte = new Date(startDate);
      if (endDate) where.date.lte = new Date(endDate);
    }

    const logs = await prisma.medicationAdherenceLog.findMany({
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
      orderBy: { date: 'desc' }
    });

    // Calculate adherence statistics
    const total = logs.length;
    const taken = logs.filter(l => l.status === 'taken').length;
    const missed = logs.filter(l => l.status === 'missed').length;
    const skipped = logs.filter(l => l.status === 'skipped').length;
    const adherenceRate = total > 0 ? (taken / total) * 100 : 0;

    res.json({
      logs,
      statistics: {
        total,
        taken,
        missed,
        skipped,
        adherenceRate: Math.round(adherenceRate * 100) / 100
      }
    });
  } catch (error) {
    console.error('Error fetching adherence:', error);
    res.status(500).json({ error: 'Failed to fetch adherence data' });
  }
};

const logAdherence = async (req, res) => {
  try {
    const { medicationId, date, status, takenTime, delayMinutes, notes } = req.body;

    if (!medicationId || !date || !status) {
      return res.status(400).json({ error: 'Medication ID, date, and status are required' });
    }

    const logDate = new Date(date);
    logDate.setHours(0, 0, 0, 0);

    const adherenceLog = await prisma.medicationAdherenceLog.upsert({
      where: {
        medicationId_date: {
          medicationId,
          date: logDate
        }
      },
      update: {
        status,
        takenTime: takenTime ? new Date(takenTime) : null,
        delayMinutes: delayMinutes || null,
        notes: notes || null
      },
      create: {
        medicationId,
        date: logDate,
        status,
        takenTime: takenTime ? new Date(takenTime) : null,
        delayMinutes: delayMinutes || null,
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

    res.status(201).json(adherenceLog);
  } catch (error) {
    console.error('Error logging adherence:', error);
    res.status(500).json({ error: 'Failed to log adherence' });
  }
};

const getAdherenceCalendar = async (req, res) => {
  try {
    const { medicationId, patientId, year, month } = req.query;
    
    if (!year || !month) {
      return res.status(400).json({ error: 'Year and month are required' });
    }

    const startDate = new Date(parseInt(year), parseInt(month) - 1, 1);
    const endDate = new Date(parseInt(year), parseInt(month), 0, 23, 59, 59);

    const where = {
      date: {
        gte: startDate,
        lte: endDate
      }
    };

    if (medicationId) where.medicationId = medicationId;
    if (patientId) {
      where.medication = {
        patientId: patientId
      };
    }

    const logs = await prisma.medicationAdherenceLog.findMany({
      where,
      include: {
        medication: {
          select: {
            id: true,
            name: true
          }
        }
      }
    });

    // Format for calendar view
    const calendar = {};
    logs.forEach(log => {
      const dateKey = log.date.toISOString().split('T')[0];
      if (!calendar[dateKey]) {
        calendar[dateKey] = [];
      }
      calendar[dateKey].push({
        medicationId: log.medicationId,
        medicationName: log.medication.name,
        status: log.status,
        takenTime: log.takenTime
      });
    });

    res.json({ calendar, logs });
  } catch (error) {
    console.error('Error fetching adherence calendar:', error);
    res.status(500).json({ error: 'Failed to fetch adherence calendar' });
  }
};

module.exports = {
  getAdherence,
  logAdherence,
  getAdherenceCalendar
};


const { subDays, addDays, isAfter, isBefore, startOfDay, isSameDay } = require('date-fns');

// List all cycles for logged-in user
exports.list = async (req, res) => {
  const prisma = req.prisma;
  try {
    const cycles = await prisma.medicationCycle.findMany({
      where: { userId: req.user.id },
      include: {
        metricLogs: true,
        doseLogs: true,
      },
      orderBy: { startDate: 'desc' },
    });
    res.json(cycles);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch cycles' });
  }
};

// Create new cycle
exports.create = async (req, res) => {
  const prisma = req.prisma;
  const { name, dosage, startDate, endDate, frequencyDays, dosesPerDay, metricsToMonitor } = req.body;
  
  if (!req.user?.id) {
    return res.status(401).json({ error: 'User not authenticated' });
  }
  
  try {
    const cycle = await prisma.medicationCycle.create({
      data: {
        userId: req.user.id,
        name,
        dosage,
        startDate: startDate ? new Date(startDate) : new Date(),
        endDate: endDate ? new Date(endDate) : null,
        frequencyDays: parseInt(frequencyDays, 10) || 1,
        dosesPerDay: parseInt(dosesPerDay ?? 1, 10),
        metricsToMonitor: metricsToMonitor ? JSON.stringify(metricsToMonitor) : null,
      },
    });

    // Create initial dose logs for today
    const today = startOfDay(new Date());
    const startDateCycle = startOfDay(startDate ? new Date(startDate) : new Date());
    
    // Create dose log for today if cycle starts today or earlier
    if (startDateCycle <= today) {
      await prisma.doseLog.create({
        data: {
          cycleId: cycle.id,
          date: today,
          taken: false,
        },
      });
    }

    res.status(201).json(cycle);
  } catch (err) {
    console.error('Create cycle error:', err);
    res.status(500).json({ error: 'Failed to create cycle' });
  }
};

// Get single cycle
exports.get = async (req, res) => {
  const prisma = req.prisma;
  const id = parseInt(req.params.id, 10);
  try {
    const cycle = await prisma.medicationCycle.findFirst({
      where: { id, userId: req.user.id },
      include: {
        metricLogs: true,
        doseLogs: true,
      },
    });
    if (!cycle) return res.status(404).json({ error: 'Not found' });
    res.json(cycle);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed' });
  }
};

exports.update = async (req, res) => {
  const prisma = req.prisma;
  const id = parseInt(req.params.id, 10);
  try {
    // First verify the cycle belongs to the authenticated user
    const existingCycle = await prisma.medicationCycle.findFirst({
      where: { 
        id,
        userId: req.user.id 
      }
    });
    
    if (!existingCycle) {
      return res.status(404).json({ error: 'Cycle not found or access denied' });
    }

    const cycle = await prisma.medicationCycle.update({
      where: { id },
      data: req.body,
    });
    res.json(cycle);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Update failed' });
  }
};

exports.remove = async (req, res) => {
  const prisma = req.prisma;
  const id = parseInt(req.params.id, 10);
  try {
    // First verify the cycle belongs to the authenticated user
    const existingCycle = await prisma.medicationCycle.findFirst({
      where: { 
        id,
        userId: req.user.id 
      }
    });
    
    if (!existingCycle) {
      return res.status(404).json({ error: 'Cycle not found or access denied' });
    }

    await prisma.medicationCycle.delete({ where: { id } });
    res.status(204).end();
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Delete failed' });
  }
};

// Add metric log
exports.addMetric = async (req, res) => {
  const prisma = req.prisma;
  const cycleId = parseInt(req.params.id, 10);
  const { kind, valueFloat, valueText, notes, date } = req.body;
  try {
    // First verify the cycle belongs to the authenticated user
    const existingCycle = await prisma.medicationCycle.findFirst({
      where: { 
        id: cycleId,
        userId: req.user.id 
      }
    });
    
    if (!existingCycle) {
      return res.status(404).json({ error: 'Cycle not found or access denied' });
    }

    const log = await prisma.metricLog.create({
      data: {
        cycleId,
        kind,
        valueFloat,
        valueText,
        notes,
        date: date ? new Date(date) : undefined,
      },
    });
    res.status(201).json(log);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Metric log failed' });
  }
};

exports.listMetrics = async (req, res) => {
  const prisma = req.prisma;
  const cycleId = parseInt(req.params.id, 10);
  try {
    // First verify the cycle belongs to the authenticated user
    const existingCycle = await prisma.medicationCycle.findFirst({
      where: { 
        id: cycleId,
        userId: req.user.id 
      }
    });
    
    if (!existingCycle) {
      return res.status(404).json({ error: 'Cycle not found or access denied' });
    }

    const logs = await prisma.metricLog.findMany({
      where: { cycleId },
      orderBy: { date: 'desc' },
    });
    res.json(logs);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Fetch metrics failed' });
  }
};

// Mark dose taken/not-taken
exports.markDose = async (req, res) => {
  const prisma = req.prisma;
  const cycleId = parseInt(req.params.id, 10);
  const { date, taken = true } = req.body;
  const d = startOfDay(date ? new Date(date) : new Date());
  try {
    // First verify the cycle belongs to the authenticated user
    const existingCycle = await prisma.medicationCycle.findFirst({
      where: { 
        id: cycleId,
        userId: req.user.id 
      }
    });
    
    if (!existingCycle) {
      return res.status(404).json({ error: 'Cycle not found or access denied' });
    }

    const log = await prisma.doseLog.upsert({
      where: {
        cycleId_date: {
          cycleId,
          date: d,
        },
      },
      update: { taken },
      create: { cycleId, date: d, taken },
    });

    // schedule next dose if taken and cycle still active
    if (taken) {
      const cycle = await prisma.medicationCycle.findUnique({ where: { id: cycleId } });
      if (cycle) {
        const next = addDays(d, cycle.frequencyDays);
        if (!cycle.endDate || isBefore(next, cycle.endDate)) {
          await prisma.doseLog.upsert({
            where: {
              cycleId_date: { cycleId, date: next },
            },
            update: {},
            create: { cycleId, date: next, taken: false },
          });
        }
      }
    }
    res.json(log);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Dose log failed' });
  }
};

// Today's doses that need to be taken
exports.todaysDoses = async (req, res) => {
  const prisma = req.prisma;
  const today = startOfDay(new Date());
  try {
    const doseLogs = await prisma.doseLog.findMany({
      where: {
        date: today,
        taken: false,
        cycle: { userId: req.user.id }
      },
      include: {
        cycle: true
      }
    });
    
    const todaysReminders = doseLogs.map(d => ({
      cycleId: d.cycleId,
      name: d.cycle.name,
      dosage: d.cycle.dosage,
      date: d.date
    }));
    
    res.json(todaysReminders);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Today doses failed' });
  }
};

// Get metric reminders based on frequency
exports.metricReminders = async (req, res) => {
  const prisma = req.prisma;
  const today = startOfDay(new Date());
  
  try {
    const cycles = await prisma.medicationCycle.findMany({
      where: { 
        userId: req.user.id,
        metricsToMonitor: { not: null }
      },
      include: {
        metricLogs: {
          orderBy: { date: 'desc' }
        }
      }
    });

    const reminders = [];

    for (const cycle of cycles) {
      if (cycle.metricsToMonitor) {
        const metricsConfig = JSON.parse(cycle.metricsToMonitor);
        
        for (const metric of metricsConfig) {
          // Find the last log for this metric type
          const lastLog = cycle.metricLogs.find(log => log.kind === metric.type);
          const frequencyDays = getFrequencyDays(metric.frequency);
          
          let nextDueDate;
          let daysSinceLastLog = 0;
          let status = 'upcoming';
          
          if (!lastLog) {
            // No logs yet, due today
            nextDueDate = today;
            status = 'due';
          } else {
            const lastLogDate = startOfDay(new Date(lastLog.date));
            daysSinceLastLog = Math.floor((today - lastLogDate) / (1000 * 60 * 60 * 24));
            
            // Calculate next due date
            nextDueDate = addDays(lastLogDate, frequencyDays);
            
            if (nextDueDate <= today) {
              status = daysSinceLastLog > frequencyDays ? 'overdue' : 'due';
            } else {
              status = 'upcoming';
            }
          }
          
          // Include all reminders within the next 7 days
          const daysUntilDue = Math.floor((nextDueDate - today) / (1000 * 60 * 60 * 24));
          
          if (daysUntilDue <= 7) { // Show next week's measurements
            reminders.push({
              cycleId: cycle.id,
              cycleName: cycle.name,
              metricType: metric.type,
              frequency: metric.frequency,
              daysSinceLastLog: Math.max(0, daysSinceLastLog),
              nextDueDate: nextDueDate,
              daysUntilDue: daysUntilDue,
              status: status
            });
          }
        }
      }
    }

    res.json(reminders);
  } catch (err) {
    console.error('Metric reminders error:', err);
    res.status(500).json({ error: 'Failed to get metric reminders' });
  }
};

// Helper function to convert frequency to days
function getFrequencyDays(frequency) {
  switch (frequency) {
    case 'daily': return 1;
    case 'weekly': return 7;
    case 'biweekly': return 14;
    case 'monthly': return 30;
    default: return 1;
  }
}

// Check if metrics are due for logging today for a specific cycle
exports.checkMetricsDue = async (req, res) => {
  const prisma = req.prisma;
  const cycleId = parseInt(req.params.id, 10);
  const today = startOfDay(new Date());
  
  try {
    const cycle = await prisma.medicationCycle.findUnique({
      where: { 
        id: cycleId,
        userId: req.user.id 
      },
      include: {
        metricLogs: {
          orderBy: { date: 'desc' }
        }
      }
    });

    if (!cycle) {
      return res.status(404).json({ error: 'Cycle not found' });
    }

    if (!cycle.metricsToMonitor) {
      return res.json({ canLog: false, dueMetrics: [], message: 'No metrics configured for monitoring' });
    }

    const metricsConfig = JSON.parse(cycle.metricsToMonitor);
    const dueMetrics = [];

    for (const metric of metricsConfig) {
      // Check if this metric type is due today
      const lastLog = cycle.metricLogs.find(log => log.kind === metric.type);
      
      let isDue = false;
      
      if (!lastLog) {
        // No logs yet, due on start date or today (whichever is later)
        const startDate = startOfDay(new Date(cycle.startDate));
        isDue = today >= startDate;
      } else {
        const lastLogDate = startOfDay(new Date(lastLog.date));
        const daysSinceLastLog = Math.floor((today - lastLogDate) / (1000 * 60 * 60 * 24));
        const frequencyDays = getFrequencyDays(metric.frequency);
        
        // Due if enough days have passed based on frequency
        isDue = daysSinceLastLog >= frequencyDays;
      }
      
      if (isDue) {
        dueMetrics.push(metric);
      }
    }

    const canLog = dueMetrics.length > 0;
    
    res.json({
      canLog,
      dueMetrics,
      allConfiguredMetrics: metricsConfig,
      message: canLog 
        ? `${dueMetrics.length} metric(s) due for logging today`
        : 'No metrics due for logging today'
    });

  } catch (err) {
    console.error('Check metrics due error:', err);
    res.status(500).json({ error: 'Failed to check metrics due' });
  }
};

// Upcoming medication intakes (next scheduled dose for each active medication)
exports.upcoming = async (req, res) => {
  const prisma = req.prisma;
  const now = new Date();
  const today = startOfDay(now);
  
  try {
    const cycles = await prisma.medicationCycle.findMany({ 
      where: { userId: req.user.id },
      include: {
        doseLogs: {
          where: {
            taken: false,
            date: { gte: today }
          },
          orderBy: { date: 'asc' },
          take: 1 // Get the next untaken dose
        }
      }
    });
    
    const upcoming = [];
    
    for (const cycle of cycles) {
      // Skip finished cycles
      if (cycle.endDate && isAfter(today, cycle.endDate)) continue;
      
      // Generate upcoming doses for the next few days
      const daysToShow = 3; // Show next 3 days of doses
      
      for (let dayOffset = 0; dayOffset < daysToShow; dayOffset++) {
        let targetDate = addDays(today, dayOffset);
        
        // Check if this day falls within the cycle schedule
        let cycleDate = new Date(cycle.startDate);
        while (isBefore(cycleDate, targetDate)) {
          cycleDate = addDays(cycleDate, cycle.frequencyDays);
        }
        
        // Skip if this date doesn't match the cycle frequency
        if (!isSameDay(cycleDate, targetDate)) continue;
        
        // Skip if beyond end date
        if (cycle.endDate && isAfter(targetDate, cycle.endDate)) continue;
        
        // Generate all doses for this day
        for (let doseIndex = 0; doseIndex < cycle.dosesPerDay; doseIndex++) {
          const doseTime = calculateDoseTime(targetDate, cycle.dosesPerDay, doseIndex);
          
          // Only include future doses
          if (doseTime > now) {
            upcoming.push({
              cycleId: cycle.id,
              name: cycle.name,
              dosage: cycle.dosage,
              date: doseTime,
              doseNumber: doseIndex + 1,
              totalDosesPerDay: cycle.dosesPerDay
            });
          }
        }
      }
    }
    
    // Sort by date
    upcoming.sort((a, b) => new Date(a.date) - new Date(b.date));
    
    res.json(upcoming);
  } catch (err) {
    console.error('Upcoming error:', err);
    res.status(500).json({ error: 'Upcoming failed' });
  }
};

// Helper function to calculate dose time based on doses per day
function calculateDoseTime(date, dosesPerDay, doseIndex) {
  const baseDate = new Date(date);
  
  // Distribute doses evenly throughout the day
  // First dose at 8 AM, last dose at 8 PM (12 hour span)
  const startHour = 8;
  const endHour = 20;
  const hourSpan = endHour - startHour;
  
  if (dosesPerDay === 1) {
    baseDate.setHours(startHour, 0, 0, 0);
  } else {
    const hoursBetweenDoses = hourSpan / (dosesPerDay - 1);
    const doseHour = startHour + (hoursBetweenDoses * doseIndex);
    baseDate.setHours(Math.floor(doseHour), (doseHour % 1) * 60, 0, 0);
  }
  
  return baseDate;
}

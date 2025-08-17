
exports.getAll = async (req, res) => {
  const prisma = req.prisma;

  try {
    const metrics = await prisma.metric.findMany({
      where: { userId: req.user.id },
      orderBy: { date: 'desc' },
    });

    res.json(metrics);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch metrics' });
  }
};

function calculateBMI(weight, height) {
  return weight / (height * height);
}

exports.create = async (req, res) => {
  const prisma = req.prisma;
  const { weight, height, bloodPressure, hipCircumference } = req.body;

  try {
    const bmi = calculateBMI(weight, height);

    const metric = await prisma.metric.create({
      data: {
        userId: req.user.id,
        weight,
        height,
        bmi,
        bloodPressure,
        hipCircumference,
      },
    });

    res.status(201).json(metric);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to create metric' });
  }
};

exports.update = async (req, res) => {
  const prisma = req.prisma;
  const id = parseInt(req.params.id, 10);
  const { weight, height } = req.body;

  try {
    let data = { ...req.body };

    if (weight && height) {
      data.bmi = calculateBMI(weight, height);
    }

    const metric = await prisma.metric.update({
      where: { id },
      data,
    });

    res.json(metric);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to update metric' });
  }
};

exports.remove = async (req, res) => {
  const prisma = req.prisma;
  const id = parseInt(req.params.id, 10);

  try {
    await prisma.metric.delete({ where: { id } });
    res.status(204).end();
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to delete metric' });
  }
};

exports.getMetricLogs = async (req, res) => {
  try {
    const { cycleId, metricType } = req.query;
    const prisma = req.prisma;

    // Build the query dynamically based on provided parameters
    const whereCondition = {};
    
    if (cycleId) {
      whereCondition.cycleId = parseInt(cycleId);
    }

    if (metricType) {
      whereCondition.kind = metricType;
    }

    // Fetch metric logs with the constructed filter
    const logs = await prisma.metricLog.findMany({
      where: whereCondition,
      orderBy: { date: 'desc' },
      take: 100, // Limit to prevent overfetching
      include: {
        cycle: {
          select: {
            name: true
          }
        }
      }
    });

    // Transform the data to match frontend expectations
    const transformedLogs = logs.map(log => ({
      id: log.id,
      date: log.date,
      cycle: log.cycle?.name || 'Unknown',
      kind: log.kind,
      valueFloat: log.valueFloat,
      valueText: log.valueText,
      notes: log.notes
    }));

    res.json(transformedLogs);
  } catch (error) {
    console.error('Error fetching metric logs:', error);
    res.status(500).json({ 
      error: 'Failed to fetch metric logs'
    });
  }
};

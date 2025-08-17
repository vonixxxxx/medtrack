exports.getAll = async (req, res) => {
  const prisma = req.prisma;

  try {
    const medications = await prisma.medication.findMany({
      where: { userId: req.user.id },
      include: { logs: true },
      orderBy: { startDate: 'desc' },
    });

    res.json(medications);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch medications' });
  }
};

exports.create = async (req, res) => {
  const prisma = req.prisma;
  const { name, startDate, endDate, dosage, frequency } = req.body;

  try {
    const medication = await prisma.medication.create({
      data: {
        userId: req.user.id,
        name,
        startDate: new Date(startDate),
        endDate: new Date(endDate),
        dosage,
        frequency,
      },
    });

    res.status(201).json(medication);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to create medication' });
  }
};

exports.update = async (req, res) => {
  const prisma = req.prisma;
  const id = parseInt(req.params.id, 10);

  try {
    const medication = await prisma.medication.update({
      where: { id },
      data: req.body,
    });

    res.json(medication);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to update medication' });
  }
};

exports.remove = async (req, res) => {
  const prisma = req.prisma;
  const id = parseInt(req.params.id, 10);

  try {
    await prisma.medication.delete({ where: { id } });
    res.status(204).end();
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to delete medication' });
  }
};

exports.addLog = async (req, res) => {
  const prisma = req.prisma;
  const medicationId = parseInt(req.params.id, 10);
  const { date, dosage } = req.body;

  try {
    const log = await prisma.medicationLog.create({
      data: {
        medicationId,
        date: date ? new Date(date) : new Date(),
        dosage,
      },
    });

    res.status(201).json(log);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to create medication log' });
  }
};

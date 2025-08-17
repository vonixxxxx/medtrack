const { Parser } = require('json2csv');

exports.csv = async (req, res) => {
  const prisma = req.prisma;
  const id = parseInt(req.params.id, 10);
  try {
    const cycle = await prisma.medicationCycle.findFirst({
      where: { id, userId: req.user.id },
      include: { metricLogs: true, doseLogs: true },
    });
    if (!cycle) return res.status(404).end();

    const data = cycle.metricLogs.map((m) => ({
      date: m.date.toISOString(),
      metric: m.kind,
      value: m.valueFloat ?? m.valueText,
      notes: m.notes || '',
    }));
    const parser = new Parser();
    const csv = parser.parse(data);

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename=cycle-${id}.csv`);
    res.send(csv);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Export failed' });
  }
};

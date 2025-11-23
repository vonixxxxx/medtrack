const prismaClient = require('@prisma/client');

exports.getUnread = async (req, res) => {
  const prisma = req.prisma;
  try {
    const reminders = await prisma.reminder.findMany({
      where: { userId: req.user.id, read: false },
      include: { medication: true },
      orderBy: { date: 'asc' },
    });
    res.json(reminders);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch reminders' });
  }
};

exports.markRead = async (req, res) => {
  const prisma = req.prisma;
  const id = parseInt(req.params.id, 10);

  try {
    await prisma.reminder.update({
      where: { id },
      data: { read: true },
    });

    res.status(204).end();
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to mark reminder as read' });
  }
};

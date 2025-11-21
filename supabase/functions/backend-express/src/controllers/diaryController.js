const prisma = require('../db/prisma');

const getDiaryEntries = async (req, res) => {
  try {
    const { userId, patientId, startDate, endDate, entryType, notebookId } = req.query;
    const where = { userId };

    if (patientId) where.patientId = patientId;
    if (entryType) where.entryType = entryType;
    if (notebookId) where.notebookId = notebookId;
    if (startDate || endDate) {
      where.date = {};
      if (startDate) where.date.gte = new Date(startDate);
      if (endDate) where.date.lte = new Date(endDate);
    }

    const entries = await prisma.diaryEntry.findMany({
      where,
      orderBy: { date: 'desc' }
    });

    res.json(entries);
  } catch (error) {
    console.error('Error fetching diary entries:', error);
    res.status(500).json({ error: 'Failed to fetch diary entries' });
  }
};

const createDiaryEntry = async (req, res) => {
  try {
    const { userId, patientId, date, entryType, title, content, attributes, tags, notebookId } = req.body;

    if (!userId || !date || !entryType) {
      return res.status(400).json({ error: 'User ID, date, and entry type are required' });
    }

    const entry = await prisma.diaryEntry.create({
      data: {
        userId,
        patientId: patientId || null,
        date: new Date(date),
        entryType,
        title: title || null,
        content: content || null,
        attributes: attributes ? JSON.stringify(attributes) : null,
        tags: tags ? JSON.stringify(tags) : null,
        notebookId: notebookId || null
      }
    });

    res.status(201).json(entry);
  } catch (error) {
    console.error('Error creating diary entry:', error);
    res.status(500).json({ error: 'Failed to create diary entry' });
  }
};

const updateDiaryEntry = async (req, res) => {
  try {
    const { id } = req.params;
    const { date, entryType, title, content, attributes, tags, notebookId } = req.body;

    const updateData = {};
    if (date !== undefined) updateData.date = new Date(date);
    if (entryType !== undefined) updateData.entryType = entryType;
    if (title !== undefined) updateData.title = title;
    if (content !== undefined) updateData.content = content;
    if (attributes !== undefined) updateData.attributes = attributes ? JSON.stringify(attributes) : null;
    if (tags !== undefined) updateData.tags = tags ? JSON.stringify(tags) : null;
    if (notebookId !== undefined) updateData.notebookId = notebookId;

    const entry = await prisma.diaryEntry.update({
      where: { id },
      data: updateData
    });

    res.json(entry);
  } catch (error) {
    console.error('Error updating diary entry:', error);
    res.status(500).json({ error: 'Failed to update diary entry' });
  }
};

const deleteDiaryEntry = async (req, res) => {
  try {
    const { id } = req.params;
    await prisma.diaryEntry.delete({ where: { id } });
    res.json({ message: 'Diary entry deleted successfully' });
  } catch (error) {
    console.error('Error deleting diary entry:', error);
    res.status(500).json({ error: 'Failed to delete diary entry' });
  }
};

module.exports = {
  getDiaryEntries,
  createDiaryEntry,
  updateDiaryEntry,
  deleteDiaryEntry
};


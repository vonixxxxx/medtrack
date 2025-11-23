const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { recognizePill } = require('../services/pillRecognitionService');
const { checkPillInteractions } = require('../services/drugInteractionService');

// Configure multer for image uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../../uploads/pills');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'pill-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

// Pill recognition now uses the enhanced service

const recognizePillFromImage = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Image file is required' });
    }

    const { userId, patientId } = req.body;
    
    if (!userId) {
      return res.status(400).json({ error: 'User ID is required' });
    }

    const imagePath = req.file.path;
    const imageUrl = `/uploads/pills/${req.file.filename}`;

    // Recognize pill using enhanced service
    const recognitionResult = await recognizePill(imagePath);

    // Check for interactions with current medications
    let interactions = [];
    if (recognitionResult.recognized && recognitionResult.medicationName) {
      interactions = await checkPillInteractions(recognitionResult, userId, patientId);
    }

    // Save recognition record
    const recognition = await prisma.pillRecognition.create({
      data: {
        userId,
        patientId: patientId || null,
        imagePath,
        imageUrl,
        recognized: recognitionResult.recognized,
        confidence: recognitionResult.confidence,
        medicationName: recognitionResult.medicationName || null,
        imprint: recognitionResult.imprint || null,
        shape: recognitionResult.shape || null,
        color: recognitionResult.color || null,
        size: recognitionResult.size || null,
        mlModel: 'enhanced-model-v1',
        rawResult: JSON.stringify(recognitionResult),
        verified: false
      }
    });

    // Return recognition with interactions
    res.status(201).json({
      ...recognition,
      interactions: interactions.length > 0 ? interactions : undefined,
      hasInteractions: interactions.length > 0
    });
  } catch (error) {
    console.error('Error recognizing pill:', error);
    res.status(500).json({ error: 'Failed to recognize pill' });
  }
};

const getRecognitionHistory = async (req, res) => {
  try {
    const { userId, patientId } = req.query;
    const where = {};

    if (userId) where.userId = userId;
    if (patientId) where.patientId = patientId;

    const recognitions = await prisma.pillRecognition.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      take: 50
    });

    res.json(recognitions);
  } catch (error) {
    console.error('Error fetching recognition history:', error);
    res.status(500).json({ error: 'Failed to fetch recognition history' });
  }
};

const verifyRecognition = async (req, res) => {
  try {
    const { id } = req.params;
    const { verified, medicationName } = req.body;

    const updateData = { verified: verified !== undefined ? verified : true };
    if (medicationName) updateData.medicationName = medicationName;

    const recognition = await prisma.pillRecognition.update({
      where: { id },
      data: updateData
    });

    res.json(recognition);
  } catch (error) {
    console.error('Error verifying recognition:', error);
    res.status(500).json({ error: 'Failed to verify recognition' });
  }
};

module.exports = {
  recognizePillFromImage: [upload.single('image'), recognizePillFromImage],
  getRecognitionHistory,
  verifyRecognition
};


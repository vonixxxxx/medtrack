const Joi = require('joi');

// Validation schemas
const surveyDataSchema = Joi.object({
  dateOfBirth: Joi.date().required(),
  biologicalSex: Joi.string().valid('Male', 'Female', 'Other').required(),
  ethnicity: Joi.string().required(),
  hasMenses: Joi.boolean().allow(null),
  ageAtMenarche: Joi.number().integer().min(8).max(20).allow(null),
  menstrualRegularity: Joi.string().allow(''),
  lastMenstrualPeriod: Joi.date().allow(null),
  cycleLength: Joi.number().integer().min(1).max(50).allow(null),
  periodDuration: Joi.number().integer().min(1).max(15).allow(null),
  usesContraception: Joi.boolean().allow(null),
  contraceptionType: Joi.string().allow(''),
  hasPreviousPregnancies: Joi.boolean().allow(null),
  isPerimenopausal: Joi.boolean().allow(null),
  isPostmenopausal: Joi.boolean().allow(null),
  ageAtMenopause: Joi.number().integer().min(30).max(70).allow(null),
  menopauseType: Joi.string().valid('natural', 'surgical').allow(''),
  isOnHRT: Joi.boolean().allow(null),
  hrtType: Joi.string().allow(''),
  iiefScore: Joi.number().integer().min(0).max(25).allow(null),
  lowTestosteroneSymptoms: Joi.string().allow(''),
  redFlagQuestions: Joi.string().allow(''),
  auditScore: Joi.number().integer().min(0).max(40).allow(null),
  smokingStatus: Joi.string().required(),
  smokingStartAge: Joi.number().integer().min(10).max(80).allow(null),
  cigarettesPerDay: Joi.number().integer().min(0).max(100).allow(null),
  vapingDevice: Joi.string().allow(''),
  nicotineMg: Joi.number().min(0).max(50).allow(null),
  pgVgRatio: Joi.string().allow(''),
  usagePattern: Joi.string().allow(''),
  psecdiScore: Joi.number().integer().min(0).max(100).allow(null),
  readinessToQuit: Joi.number().integer().min(0).max(10).allow(null),
  ipaqScore: Joi.number().integer().min(0).max(100).allow(null),
  weight: Joi.number().min(20).max(300).required(),
  height: Joi.number().min(100).max(250).required(),
  waistCircumference: Joi.number().min(50).max(200).allow(null),
  hipCircumference: Joi.number().min(50).max(200).allow(null),
  neckCircumference: Joi.number().min(25).max(60).allow(null),
  systolicBP: Joi.number().integer().min(70).max(250).allow(null),
  diastolicBP: Joi.number().integer().min(40).max(150).allow(null)
});

// Save survey data
exports.saveSurveyData = async (req, res) => {
  try {
    const userId = req.user.id;
    const prisma = req.prisma;

    // Validate input
    const { error, value } = surveyDataSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    // Check if survey data already exists
    const existingSurvey = await prisma.userSurveyData.findUnique({
      where: { userId }
    });

    if (existingSurvey) {
      // Update existing survey data
      const updatedSurvey = await prisma.userSurveyData.update({
        where: { userId },
        data: {
          ...value,
          updatedAt: new Date()
        }
      });
      
      res.json({ 
        message: 'Survey data updated successfully',
        surveyData: updatedSurvey
      });
    } else {
      // Create new survey data
      const newSurvey = await prisma.userSurveyData.create({
        data: {
          userId,
          ...value
        }
      });
      
      res.json({ 
        message: 'Survey data saved successfully',
        surveyData: newSurvey
      });
    }
  } catch (err) {
    console.error('Save survey data error:', err);
    res.status(500).json({ error: 'Failed to save survey data' });
  }
};

// Get survey data
exports.getSurveyData = async (req, res) => {
  try {
    const userId = req.user.id;
    const prisma = req.prisma;

    const surveyData = await prisma.userSurveyData.findUnique({
      where: { userId }
    });

    if (!surveyData) {
      return res.status(404).json({ error: 'Survey data not found' });
    }

    res.json({ surveyData });
  } catch (err) {
    console.error('Get survey data error:', err);
    res.status(500).json({ error: 'Failed to get survey data' });
  }
};

// Mark survey as completed
exports.completeSurvey = async (req, res) => {
  try {
    const userId = req.user.id;
    const prisma = req.prisma;

    await prisma.user.update({
      where: { id: userId },
      data: { surveyCompleted: true }
    });

    res.json({ message: 'Survey marked as completed' });
  } catch (err) {
    console.error('Complete survey error:', err);
    res.status(500).json({ error: 'Failed to complete survey' });
  }
};

// Check survey completion status
exports.checkSurveyStatus = async (req, res) => {
  try {
    const userId = req.user.id;
    const prisma = req.prisma;

    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { surveyCompleted: true }
    });

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json({ surveyCompleted: user.surveyCompleted });
  } catch (err) {
    console.error('Check survey status error:', err);
    res.status(500).json({ error: 'Failed to check survey status' });
  }
};

// Update password (for survey completion)
exports.updatePassword = async (req, res) => {
  try {
    const { password } = req.body;
    const userId = req.user.id;
    const prisma = req.prisma;

    if (!password || password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters' });
    }

    const bcrypt = require('bcrypt');
    const hashedPassword = await bcrypt.hash(password, 12);

    await prisma.user.update({
      where: { id: userId },
      data: { password: hashedPassword }
    });

    res.json({ message: 'Password updated successfully' });
  } catch (err) {
    console.error('Update password error:', err);
    res.status(500).json({ error: 'Failed to update password' });
  }
};



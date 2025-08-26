const Joi = require('joi');
const { calculateAge, calculateBMI, calculateWHR, calculateWHtR, calculateBRI } = require('../utils/calculations');

// Validation schemas
const registrationSchema = Joi.object({
  // Core Demographics
  dateOfBirth: Joi.date().max('now').required(),
  biologicalSex: Joi.string().valid('male', 'female', 'other').required(),
  ethnicity: Joi.string().required(),
  
  // Female-specific fields
  hasMenses: Joi.when('biologicalSex', {
    is: 'female',
    then: Joi.boolean().required(),
    otherwise: Joi.forbidden()
  }),
  ageAtMenarche: Joi.when('hasMenses', {
    is: true,
    then: Joi.number().integer().min(8).max(20).required(),
    otherwise: Joi.forbidden()
  }),
  menstrualRegularity: Joi.when('hasMenses', {
    is: true,
    then: Joi.string().valid('regular', 'irregular').required(),
    otherwise: Joi.forbidden()
  }),
  lastMenstrualPeriod: Joi.when('hasMenses', {
    is: true,
    then: Joi.date().max('now').required(),
    otherwise: Joi.forbidden()
  }),
  cycleLength: Joi.when('hasMenses', {
    is: true,
    then: Joi.number().integer().min(20).max(45).required(),
    otherwise: Joi.forbidden()
  }),
  periodDuration: Joi.when('hasMenses', {
    is: true,
    then: Joi.number().integer().min(1).max(14).required(),
    otherwise: Joi.forbidden()
  }),
  usesContraception: Joi.when('hasMenses', {
    is: true,
    then: Joi.boolean().required(),
    otherwise: Joi.forbidden()
  }),
  contraceptionType: Joi.when('usesContraception', {
    is: true,
    then: Joi.string().min(1).max(200).required(),
    otherwise: Joi.forbidden()
  }),
  hasPreviousPregnancies: Joi.when('biologicalSex', {
    is: 'female',
    then: Joi.boolean().required(),
    otherwise: Joi.forbidden()
  }),
  isPerimenopausal: Joi.when('biologicalSex', {
    is: 'female',
    then: Joi.boolean().required(),
    otherwise: Joi.forbidden()
  }),
  isPostmenopausal: Joi.when('biologicalSex', {
    is: 'female',
    then: Joi.boolean().required(),
    otherwise: Joi.forbidden()
  }),
  ageAtMenopause: Joi.when('isPostmenopausal', {
    is: true,
    then: Joi.number().integer().min(30).max(70).required(),
    otherwise: Joi.forbidden()
  }),
  menopauseType: Joi.when('isPostmenopausal', {
    is: true,
    then: Joi.string().valid('natural', 'early', 'premature_ovarian_insufficiency', 'surgical', 'induced').required(),
    otherwise: Joi.forbidden()
  }),
  onHRT: Joi.when('isPostmenopausal', {
    is: true,
    then: Joi.boolean().required(),
    otherwise: Joi.forbidden()
  }),
  hrtType: Joi.when('onHRT', {
    is: true,
    then: Joi.string().min(1).max(200).required(),
    otherwise: Joi.forbidden()
  }),
  
  // Male-specific fields
  iiefScore: Joi.when('biologicalSex', {
    is: 'male',
    then: Joi.number().integer().min(5).max(25).required(),
    otherwise: Joi.forbidden()
  }),
  lowTestosteroneSymptoms: Joi.when('biologicalSex', {
    is: 'male',
    then: Joi.array().items(Joi.string().valid(
      'low_libido',
      'reduced_morning_erections',
      'fatigue',
      'low_energy',
      'depressed_mood',
      'irritability',
      'reduced_muscle_mass',
      'reduced_strength',
      'increased_body_fat',
      'reduced_shaving_frequency',
      'reduced_body_hair',
      'decreased_bone_strength'
    )).required(),
    otherwise: Joi.forbidden()
  }),
  redFlagQuestions: Joi.when('biologicalSex', {
    is: 'male',
    then: Joi.object({
      gynecomastia: Joi.boolean().required(),
      testicularAtrophy: Joi.boolean().required(),
      infertility: Joi.boolean().required(),
      pituitaryDisease: Joi.boolean().required(),
      headTrauma: Joi.boolean().required(),
      chemoRadiation: Joi.boolean().required()
    }).required(),
    otherwise: Joi.forbidden()
  }),
  
  // Lifestyle
  auditScore: Joi.number().integer().min(0).max(40).required(),
  smokingStatus: Joi.string().valid('never', 'current', 'ex', 'vaping').required(),
  smokingStartAge: Joi.when('smokingStatus', {
    is: Joi.string().valid('current', 'ex'),
    then: Joi.number().integer().min(10).max(80).required(),
    otherwise: Joi.forbidden()
  }),
  cigarettesPerDay: Joi.when('smokingStatus', {
    is: Joi.string().valid('current', 'ex'),
    then: Joi.number().integer().min(1).max(100).required(),
    otherwise: Joi.forbidden()
  }),
  vapingInfo: Joi.when('smokingStatus', {
    is: 'vaping',
    then: Joi.object().required(),
    otherwise: Joi.forbidden()
  }),
  ipaqScore: Joi.number().integer().min(0).max(100).required(),
  
  // Anthropometrics & Vitals
  weight: Joi.number().positive().max(500).required(), // kg
  height: Joi.number().positive().max(3).required(),   // m
  waistCircumference: Joi.number().positive().max(200).optional(), // cm
  hipCircumference: Joi.number().positive().max(200).optional(),   // cm
  neckCircumference: Joi.number().positive().max(50).optional(),   // cm
  bloodPressure: Joi.string().pattern(/^\d{2,3}\/\d{2,3}\s*mmHg$/).optional(),
});

exports.completeRegistration = async (req, res) => {
  try {
    const userId = req.user.id;
    const prisma = req.prisma;
    
    // Validate input
    const { error, value } = registrationSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }
    
    // Calculate derived values
    const age = calculateAge(value.dateOfBirth);
    const bmi = calculateBMI(value.weight, value.height);
    const whr = value.waistCircumference && value.hipCircumference ? 
      calculateWHR(value.waistCircumference, value.hipCircumference) : null;
    const whtr = value.waistCircumference ? 
      calculateWHtR(value.waistCircumference, value.height) : null;
    const bri = value.waistCircumference && value.height ? 
      calculateBRI(value.waistCircumference, value.height, value.weight) : null;
    
    // Calculate pack years for smokers
    let packYears = null;
    if (value.smokingStatus === 'current' || value.smokingStatus === 'ex') {
      const yearsSmoked = age - value.smokingStartAge;
      packYears = (value.cigarettesPerDay / 20) * yearsSmoked;
    }
    
    // Update user with registration data
    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: {
        ...value,
        bmi,
        whr,
        whtr,
        bri,
        packYears,
        isRegistrationComplete: true
      }
    });
    
    res.json({
      message: 'Registration completed successfully',
      user: {
        id: updatedUser.id,
        email: updatedUser.email,
        age,
        biologicalSex: updatedUser.biologicalSex,
        ethnicity: updatedUser.ethnicity,
        isRegistrationComplete: updatedUser.isRegistrationComplete
      }
    });
    
  } catch (err) {
    console.error('Registration completion error:', err);
    res.status(500).json({ error: 'Registration failed', details: err.message });
  }
};

exports.getRegistrationStatus = async (req, res) => {
  try {
    const userId = req.user.id;
    const prisma = req.prisma;
    
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        isRegistrationComplete: true,
        dateOfBirth: true,
        biologicalSex: true,
        ethnicity: true
      }
    });
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    const age = user.dateOfBirth ? calculateAge(user.dateOfBirth) : null;
    
    res.json({
      isRegistrationComplete: user.isRegistrationComplete,
      demographics: {
        age,
        biologicalSex: user.biologicalSex,
        ethnicity: user.ethnicity
      }
    });
    
  } catch (err) {
    console.error('Get registration status error:', err);
    res.status(500).json({ error: 'Failed to get registration status' });
  }
};

exports.updateRegistration = async (req, res) => {
  try {
    const userId = req.user.id;
    const prisma = req.prisma;
    
    // Get current user data
    const currentUser = await prisma.user.findUnique({
      where: { id: userId }
    });
    
    if (!currentUser) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    // Validate input (allow partial updates)
    const { error, value } = Joi.object({
      // Allow any of the registration fields to be updated
      dateOfBirth: Joi.date().max('now').optional(),
      biologicalSex: Joi.string().valid('male', 'female', 'other').optional(),
      ethnicity: Joi.string().optional(),
      // ... add all other fields as optional
    }).validate(req.body);
    
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }
    
    // Recalculate derived values if relevant fields changed
    let updateData = { ...value };
    
    if (value.weight || value.height) {
      const weight = value.weight || currentUser.weight;
      const height = value.height || currentUser.height;
      if (weight && height) {
        updateData.bmi = calculateBMI(weight, height);
      }
    }
    
    if (value.waistCircumference || value.hipCircumference) {
      const waist = value.waistCircumference || currentUser.waistCircumference;
      const hip = value.hipCircumference || currentUser.hipCircumference;
      if (waist && hip) {
        updateData.whr = calculateWHR(waist, hip);
      }
    }
    
    if (value.waistCircumference || value.height) {
      const waist = value.waistCircumference || currentUser.waistCircumference;
      const height = value.height || currentUser.height;
      if (waist && height) {
        updateData.whtr = calculateWHtR(waist, height);
      }
    }
    
    if (value.waistCircumference || value.height || value.weight) {
      const waist = value.waistCircumference || currentUser.waistCircumference;
      const height = value.height || currentUser.height;
      const weight = value.weight || currentUser.weight;
      if (waist && height && weight) {
        updateData.bri = calculateBRI(waist, height, weight);
      }
    }
    
    // Update user
    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: updateData
    });
    
    res.json({
      message: 'Registration updated successfully',
      user: {
        id: updatedUser.id,
        email: updatedUser.email,
        isRegistrationComplete: updatedUser.isRegistrationComplete
      }
    });
    
  } catch (err) {
    console.error('Registration update error:', err);
    res.status(500).json({ error: 'Registration update failed', details: err.message });
  }
};

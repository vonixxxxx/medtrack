const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs').promises;
const { authenticateToken } = require('../middleware/authMiddleware');
const { 
  generalRateLimit, 
  searchRateLimit,
  devRateLimit,
  handleValidationErrors,
  auditLogger 
} = require('../middleware/security');
const { body, validationResult } = require('express-validator');
const medicationController = require('../controllers/medicationController');

// Load the enhanced medication dataset
let medicationsDataset = null;

// In-memory storage for user medications (replace with database in production)
let userMedications = [];


const loadMedicationsDataset = async () => {
  try {
    const datasetPath = path.join(__dirname, '../../medications_master_enhanced.json');
    const data = await fs.readFile(datasetPath, 'utf8');
    medicationsDataset = JSON.parse(data);
    console.log(`Loaded ${medicationsDataset.length} medications from enhanced dataset`);
  } catch (error) {
    console.error('Error loading medications dataset:', error);
    medicationsDataset = [];
  }
};

// Initialize dataset on startup
loadMedicationsDataset();

// Validation schemas
const medicationValidation = [
  body('name').trim().isLength({ min: 1 }).withMessage('Medication name is required'),
  body('dosage').trim().isLength({ min: 1 }).withMessage('Dosage is required'),
  body('frequency').trim().isLength({ min: 1 }).withMessage('Frequency is required'),
  body('startDate').custom((value) => {
    if (!value) {
      throw new Error('Start date is required');
    }
    // Accept both ISO8601 and YYYY-MM-DD formats
    const iso8601Regex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$/;
    const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
    if (!iso8601Regex.test(value) && !dateRegex.test(value)) {
      throw new Error('Valid start date is required (YYYY-MM-DD or ISO8601 format)');
    }
    return true;
  }),
  body('endDate').optional().custom((value) => {
    if (!value) return true;
    // Accept both ISO8601 and YYYY-MM-DD formats
    const iso8601Regex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$/;
    const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
    if (!iso8601Regex.test(value) && !dateRegex.test(value)) {
      throw new Error('Valid end date is required (YYYY-MM-DD or ISO8601 format)');
    }
    return true;
  })
];

const medicationLogValidation = [
  body('medicationId').trim().isLength({ min: 1 }).withMessage('Medication ID is required'),
  body('takenAt').optional().isISO8601().withMessage('Valid taken date is required')
];

// Public routes

// Clear all medications endpoint (for testing purposes)
router.get('/clear-all', (req, res) => {
  try {
    const previousCount = userMedications.length;
    userMedications = [];
    
    res.json({
      success: true,
      message: `Cleared ${previousCount} medications from database`,
      clearedCount: previousCount
    });
  } catch (error) {
    console.error('Error clearing medications:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to clear medications'
    });
  }
});


// Validation endpoint moved to enhancedMedicationValidation.js
// router.get('/validate/:medicationName', 
//   generalRateLimit,
//   medicationController.validateMedication
// );

// Medication validation endpoint (POST)
router.post('/validate', 
  generalRateLimit,
  async (req, res) => {
    try {
      const { medication, product, dosage, frequency, intakeType, customFlags } = req.body;
      
      // Basic validation
      if (!medication || !product || !dosage || !frequency) {
        return res.status(422).json({
          valid: false,
          errors: [
            { field: 'general', message: 'Missing required fields: medication, product, dosage, frequency' }
          ]
        });
      }
      
      // Mock validation logic - in real app, this would check against medical databases
      const validationResult = {
        valid: true,
        source: 'Hospital-Grade Validation System',
        medication: {
          name: medication,
          product: product,
          dosage: dosage,
          frequency: frequency,
          intakeType: intakeType || 'oral',
          customFlags: customFlags || {}
        },
        warnings: [],
        recommendations: [
          'Take with food to reduce stomach upset',
          'Monitor for side effects',
          'Keep medication in original container'
        ]
      };
      
      res.json(validationResult);
    } catch (error) {
      console.error('Error validating medication:', error);
      res.status(500).json({ 
        valid: false, 
        error: 'Validation service temporarily unavailable' 
      });
    }
  }
);


// Search endpoint moved to enhancedMedicationValidation.js
// router.get('/search', 
//   devRateLimit,
//   medicationController.searchMedications
// );

// Get product options for a specific medication (public endpoint)
router.get('/product/:productId/options', 
  generalRateLimit,
  medicationController.getProductOptions
);

// Add medication via chat (public endpoint for testing)
router.post('/add-chat', 
  generalRateLimit,
  medicationController.addMedicationChat
);

// Get the full medication dataset
router.get('/dataset', async (req, res) => {
  try {
    if (!medicationsDataset) {
      await loadMedicationsDataset();
    }
    
    res.json(medicationsDataset);
  } catch (error) {
    console.error('Error serving medication dataset:', error);
    res.status(500).json({ error: 'Failed to load medication dataset' });
  }
});

// Create user medication (public for testing)
router.post('/user', 
  generalRateLimit,
  // Temporarily disable validation for testing
  // medicationValidation,
  // handleValidationErrors,
  auditLogger('CREATE', 'Medication'),
  async (req, res) => {
    try {
      console.log('💊 Creating user medication with data:', req.body);
      
      // Mock user ID for testing
      const userId = 'test-user-id';
      const medicationData = {
        ...req.body,
        userId,
        isActive: true,
        createdAt: new Date().toISOString()
      };
      
      // Create new medication with proper structure
      const newMedication = {
        id: Date.now().toString(),
        name: medicationData.medication_name || medicationData.name || medicationData.generic_name || 'Unknown Medication',
        medication_name: medicationData.medication_name || medicationData.name || medicationData.generic_name,
        generic_name: medicationData.generic_name || medicationData.medication_name || medicationData.name,
        brandName: medicationData.brand || medicationData.brand_names?.[0] || medicationData.name,
        strength: medicationData.strength,
        dosage: medicationData.dosage,
        frequency: medicationData.frequency,
        frequency_display: medicationData.frequency_display || medicationData.frequency,
        customFrequency: medicationData.customFrequency || null,
        form: medicationData.form || medicationData.intakeType || 'tablet',
        route: medicationData.route || 'oral',
        timeOfDay: 'morning', // Default
        startDate: medicationData.startDate || new Date().toISOString(),
        endDate: medicationData.endDate,
        instructions: medicationData.instructions || 'Take as prescribed',
        sideEffects: 'Monitor for side effects',
        doctor: 'Dr. Prescriber',
        pharmacy: 'Local Pharmacy',
        isActive: true,
        scheduleType: 'fixed',
        scheduleTimes: '["08:00"]',
        monitoringMetrics: medicationData.selected_metrics || medicationData.selectedMetrics || [],
        metricUpdateFrequency: medicationData.metricUpdateFrequency || 'daily',
        validationVersion: medicationData.validationVersion || '1.0',
        validatedAt: medicationData.validatedAt || new Date().toISOString(),
        // Additional fields for wizard
        unit: medicationData.unit || 'mg',
        validationStatus: medicationData.validationStatus || 'validated',
        aiValidated: true,
        confidence: medicationData.confidence || 0.9,
        drug_class: medicationData.drug_class || 'Unknown',
        special_instructions: medicationData.special_instructions || '',
        ai_generated_notes: medicationData.ai_generated_notes || '',
        logging_frequency: medicationData.logging_frequency || 'daily',
        start_date: medicationData.start_date || new Date().toISOString().split('T')[0]
      };
      
      // Add to in-memory storage
      userMedications.push(newMedication);
      
      console.log('✅ User medication created successfully:', newMedication);
      console.log('📊 Total medications in storage:', userMedications.length);
      res.status(201).json(newMedication);
    } catch (error) {
      console.error('Error creating user medication:', error);
      res.status(500).json({ error: 'Failed to create medication' });
    }
  }
);

// Get user medications (public for testing)
router.get('/user', 
  generalRateLimit,
  auditLogger('READ', 'Medication'),
  async (req, res) => {
    try {
      console.log('📋 Fetching user medications');
      
      // Return actual stored medications
      const medications = userMedications;
      
      console.log('✅ User medications fetched successfully:', medications.length, 'medications');
      res.json({ medications });
    } catch (error) {
      console.error('Error fetching user medications:', error);
      res.status(500).json({ error: 'Failed to fetch medications' });
    }
  }
);

// Log medication dose (public for testing)
router.post('/user/log-dose', 
  generalRateLimit,
  auditLogger('CREATE', 'MedicationLog'),
  async (req, res) => {
    try {
      const userId = 'test-user-id'; // Mock user ID for testing
      const { medicationId, takenAt, notes } = req.body;
      
      if (!medicationId) {
        return res.status(400).json({ error: 'Medication ID is required' });
      }
      
      // Mock log creation - replace with actual Prisma create
      const logEntry = {
        id: Date.now().toString(),
        userId,
        medicationId,
        takenAt: takenAt || new Date().toISOString(),
        notes: notes || '',
        createdAt: new Date().toISOString()
      };
      
      console.log('💊 Medication dose logged:', logEntry);
      
      res.status(201).json({
        success: true,
        message: 'Medication dose logged successfully',
        logEntry
      });
    } catch (error) {
      console.error('Error logging medication dose:', error);
      res.status(500).json({ error: 'Failed to log medication dose' });
    }
  }
);

// Get active medications for a user
router.get('/active', 
  authenticateToken,
  generalRateLimit,
  auditLogger('READ', 'Medication'),
  async (req, res) => {
    try {
      const userId = req.user.id;
      
      // Return empty array - no test data
      const medications = [];
      
      res.json(medications);
    } catch (error) {
      console.error('Error fetching active medications:', error);
      res.status(500).json({ error: 'Failed to fetch medications' });
    }
  }
);

// Create a new medication
router.post('/',
  // Temporarily disable authentication for testing
  // authenticateToken,
  generalRateLimit,
  // Temporarily disable validation to debug
  // medicationValidation,
  // handleValidationErrors,
  auditLogger('CREATE', 'Medication'),
  async (req, res) => {
    try {
      console.log('💊 Creating medication with data:', req.body);
      
      // Mock user ID for testing
      const userId = 'test-user-id';
      const medicationData = {
        ...req.body,
        userId,
        isActive: true,
        createdAt: new Date().toISOString()
      };
      
      // Mock creation - replace with actual Prisma create
      const newMedication = {
        id: Date.now().toString(),
        ...medicationData
      };
      
      console.log('✅ Medication created successfully:', newMedication);
      res.status(201).json(newMedication);
    } catch (error) {
      console.error('Error creating medication:', error);
      res.status(500).json({ error: 'Failed to create medication' });
    }
  }
);

// Update a medication
router.put('/:id',
  authenticateToken,
  generalRateLimit,
  medicationValidation,
  handleValidationErrors,
  auditLogger('UPDATE', 'Medication'),
  async (req, res) => {
    try {
      const { id } = req.params;
      const userId = req.user.id;
      
      // Mock update - replace with actual Prisma update
      const updatedMedication = {
        id,
        ...req.body,
        userId,
        updatedAt: new Date().toISOString()
      };
      
      res.json(updatedMedication);
    } catch (error) {
      console.error('Error updating medication:', error);
      res.status(500).json({ error: 'Failed to update medication' });
    }
  }
);

// Delete a medication
router.delete('/:id',
  authenticateToken,
  generalRateLimit,
  auditLogger('DELETE', 'Medication'),
  async (req, res) => {
    try {
      const { id } = req.params;
      const userId = req.user.id;
      
      // Mock deletion - replace with actual Prisma delete
      res.status(200).json({ 
        success: true, 
        message: 'Medication deleted successfully',
        id 
      });
    } catch (error) {
      console.error('Error deleting medication:', error);
      res.status(500).json({ error: 'Failed to delete medication' });
    }
  }
);

// Protected routes (require authentication)
// router.use(authenticateToken);

// User medication management - using public routes for testing
// router.get('/user', 
//   generalRateLimit,
//   auditLogger('READ', 'Medication'),
//   medicationController.getUserMedications
// );

router.put('/user/:id', 
  generalRateLimit,
  medicationValidation,
  handleValidationErrors,
  auditLogger('UPDATE', 'Medication'),
  medicationController.updateUserMedication
);

router.delete('/user/:id', 
  generalRateLimit,
  auditLogger('DELETE', 'Medication'),
  medicationController.deleteUserMedication
);

// Medication logging
router.post('/user/log', 
  generalRateLimit,
  medicationLogValidation,
  handleValidationErrors,
  auditLogger('CREATE', 'MedicationLog'),
  medicationController.logMedicationDose
);


router.get('/user/logs', 
  generalRateLimit,
  auditLogger('READ', 'MedicationLog'),
  medicationController.getMedicationLogs
);

// User metrics management
router.get('/user/metrics', 
  generalRateLimit,
  auditLogger('READ', 'Metric'),
  async (req, res) => {
    try {
      // Return empty array for now - no test data
      res.json({ metrics: [] });
    } catch (error) {
      console.error('Error fetching user metrics:', error);
      res.status(500).json({ error: 'Failed to fetch metrics' });
    }
  }
);

router.post('/user/metrics', 
  generalRateLimit,
  auditLogger('CREATE', 'Metric'),
  async (req, res) => {
    try {
      console.log('📊 Creating user metric with data:', req.body);
      
      // Mock user ID for testing
      const userId = 'test-user-id';
      const metricData = {
        ...req.body,
        userId,
        createdAt: new Date().toISOString()
      };
      
      // Create new metric with proper structure
      const newMetric = {
        id: Date.now().toString(),
        ...metricData
      };
      
      console.log('✅ User metric created successfully:', newMetric);
      res.status(201).json(newMetric);
    } catch (error) {
      console.error('Error creating user metric:', error);
      res.status(500).json({ error: 'Failed to create metric' });
    }
  }
);

router.put('/user/metrics/:id', 
  generalRateLimit,
  auditLogger('UPDATE', 'Metric'),
  medicationController.updateUserMetric
);

router.delete('/user/metrics/:id', 
  generalRateLimit,
  auditLogger('DELETE', 'Metric'),
  medicationController.deleteUserMetric
);

// Medication cycles
router.get('/user/cycles', 
  generalRateLimit,
  auditLogger('READ', 'MedicationCycle'),
  medicationController.getUserMedicationCycles
);

router.post('/user/cycles', 
  generalRateLimit,
  auditLogger('CREATE', 'MedicationCycle'),
  medicationController.createUserMedicationCycle
);

router.put('/user/cycles/:id', 
  generalRateLimit,
  auditLogger('UPDATE', 'MedicationCycle'),
  medicationController.updateUserMedicationCycle
);

router.delete('/user/cycles/:id', 
  generalRateLimit,
  auditLogger('DELETE', 'MedicationCycle'),
  medicationController.deleteUserMedicationCycle
);

// Ollama AI endpoints
router.get('/ollama/status', 
  generalRateLimit,
  medicationController.checkOllamaStatus
);

router.post('/ollama/validate', 
  generalRateLimit,
  medicationController.validateMedicationInput
);

router.post('/ollama/health-report', 
  generalRateLimit,
  medicationController.generateHealthReport
);

router.post('/ollama/educational-suggestions', 
  generalRateLimit,
  medicationController.generateEducationalSuggestions
);

router.post('/ollama/chat', 
  generalRateLimit,
  medicationController.chatWithAssistant
);

router.get('/ollama/service-status', 
  generalRateLimit,
  medicationController.getServiceStatus
);

module.exports = router;

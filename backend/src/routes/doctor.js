const express = require('express');
const router = express.Router();
const { requireClinician } = require('../middleware/roleMiddleware');
const doctorController = require('../controllers/doctorController');

// All doctor routes require clinician role
router.use(requireClinician);

// Patient management
router.get('/patients', doctorController.getPatients);
router.get('/patients/:id', doctorController.getPatient);
router.post('/patients/:id/conditions', doctorController.addPatientConditions);

// Medical history parsing
router.post('/parse-history', doctorController.parseMedicalHistory);

// HbA1c adjustment calculator
router.post('/hba1c-adjust', doctorController.calculateHbA1cAdjustment);

// Analytics and reporting
router.get('/analytics', doctorController.getAnalytics);
router.get('/export/patients', doctorController.exportPatients);

module.exports = router;



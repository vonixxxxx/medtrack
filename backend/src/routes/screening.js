const express = require('express');
const router = express.Router();
const { authenticateToken } = require('../middleware/authMiddleware');
const screeningController = require('../controllers/screeningController');

// Apply authentication middleware to all routes
router.use(authenticateToken);

// IIEF-5 Assessment Routes
router.post('/iief5', screeningController.submitIIEF5Results);
router.get('/iief5', screeningController.getUserScreeningResults);

// AUDIT Assessment Routes
router.post('/audit', screeningController.submitAUDITResults);
router.get('/audit', screeningController.getUserScreeningResults);

// Heart Risk Assessment Routes
router.post('/heart-risk', screeningController.submitHeartRiskResults);

// Testosterone and BMI Routes
router.post('/testosterone', screeningController.submitTestosteroneResults);
router.post('/bmi', screeningController.submitBMIResult);

// General Screening Routes
router.get('/results', screeningController.getUserScreeningResults);
router.get('/results/:assessmentId', screeningController.getAssessmentResult);
router.get('/stats', screeningController.getScreeningStats);
router.get('/export', screeningController.exportScreeningResults);

module.exports = router;

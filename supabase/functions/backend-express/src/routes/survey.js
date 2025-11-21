const express = require('express');
const router = express.Router();
const surveyController = require('../controllers/surveyController');
const authMiddleware = require('../middleware/authMiddleware');

// Apply authentication middleware to all routes
router.use(authMiddleware);

// Survey data routes
router.post('/survey-data', surveyController.saveSurveyData);
router.get('/survey-data', surveyController.getSurveyData);
router.put('/complete-survey', surveyController.completeSurvey);
router.get('/survey-status', surveyController.checkSurveyStatus);
router.put('/update-password', surveyController.updatePassword);

module.exports = router;



const express = require('express');
const router = express.Router();
const { authenticateToken } = require('../middleware/authMiddleware');
const registrationController = require('../controllers/registrationController');

// Complete registration process
router.post('/complete', authenticateToken, registrationController.completeRegistration);

// Get registration status
router.get('/status', authenticateToken, registrationController.getRegistrationStatus);

// Update registration data
router.put('/update', authenticateToken, registrationController.updateRegistration);

module.exports = router;

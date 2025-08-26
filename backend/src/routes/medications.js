const express = require('express');
const router = express.Router();
const medicationController = require('../controllers/medicationController');
const { authenticateToken } = require('../middleware/authMiddleware');

// Public routes (for search)
router.get('/search', medicationController.searchMedications);
router.get('/product/:productId/options', medicationController.getProductOptions);

// Protected routes (require authentication)
router.use(authenticateToken);
router.post('/validate', medicationController.validateMedication);
router.post('/cycles', medicationController.createMedicationCycle);
router.get('/cycles', medicationController.getUserMedicationCycles);
router.put('/cycles/:cycleId', medicationController.updateMedicationCycle);
router.delete('/cycles/:cycleId', medicationController.deleteMedicationCycle);

// Optional Ollama integration
router.post('/ollama/suggest', medicationController.getOllamaSuggestions);

module.exports = router;

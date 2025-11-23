const express = require('express');
const router = express.Router();
const {
  recognizePillFromImage,
  getRecognitionHistory,
  verifyRecognition
} = require('../controllers/pillRecognitionController');
const {
  addMedicationFromPill,
  getMedicationsWithWarnings
} = require('../controllers/medicationTrackingController');

router.post('/recognize', recognizePillFromImage);
router.get('/history', getRecognitionHistory);
router.patch('/:id/verify', verifyRecognition);
router.post('/add-medication', addMedicationFromPill);

module.exports = router;


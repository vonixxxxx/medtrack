const express = require('express');
const router = express.Router();
const {
  getPatientProfiles,
  createPatientProfile,
  updatePatientProfile,
  deletePatientProfile
} = require('../controllers/patientProfileController');

router.get('/', getPatientProfiles);
router.post('/', createPatientProfile);
router.put('/:id', updatePatientProfile);
router.delete('/:id', deletePatientProfile);

module.exports = router;




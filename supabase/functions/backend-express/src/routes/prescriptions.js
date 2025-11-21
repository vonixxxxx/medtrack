const express = require('express');
const router = express.Router();
const {
  getPrescriptions,
  getPrescription,
  createPrescription,
  updatePrescription,
  deletePrescription
} = require('../controllers/prescriptionController');

router.get('/', getPrescriptions);
router.get('/:id', getPrescription);
router.post('/', createPrescription);
router.put('/:id', updatePrescription);
router.delete('/:id', deletePrescription);

module.exports = router;




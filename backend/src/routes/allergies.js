const express = require('express');
const router = express.Router();
const {
  getAllergies,
  getAllergy,
  createAllergy,
  updateAllergy,
  deleteAllergy
} = require('../controllers/allergyController');

router.get('/', getAllergies);
router.get('/:id', getAllergy);
router.post('/', createAllergy);
router.put('/:id', updateAllergy);
router.delete('/:id', deleteAllergy);

module.exports = router;




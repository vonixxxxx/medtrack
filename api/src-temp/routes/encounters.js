const express = require('express');
const router = express.Router();
const {
  getEncounters,
  getEncounter,
  createEncounter,
  updateEncounter,
  deleteEncounter
} = require('../controllers/encounterController');

router.get('/', getEncounters);
router.get('/:id', getEncounter);
router.post('/', createEncounter);
router.put('/:id', updateEncounter);
router.delete('/:id', deleteEncounter);

module.exports = router;




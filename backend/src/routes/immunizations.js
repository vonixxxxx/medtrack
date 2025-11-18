const express = require('express');
const router = express.Router();
const {
  getImmunizations,
  getImmunization,
  createImmunization,
  updateImmunization,
  deleteImmunization
} = require('../controllers/immunizationController');

router.get('/', getImmunizations);
router.get('/:id', getImmunization);
router.post('/', createImmunization);
router.put('/:id', updateImmunization);
router.delete('/:id', deleteImmunization);

module.exports = router;




const express = require('express');
const router = express.Router();
const {
  getSideEffects,
  createSideEffect,
  updateSideEffect,
  deleteSideEffect
} = require('../controllers/sideEffectController');

router.get('/', getSideEffects);
router.post('/', createSideEffect);
router.put('/:id', updateSideEffect);
router.delete('/:id', deleteSideEffect);

module.exports = router;




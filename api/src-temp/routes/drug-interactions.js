const express = require('express');
const router = express.Router();
const {
  checkInteractions,
  addInteraction,
  getInteractions
} = require('../controllers/drugInteractionController');

router.post('/check', checkInteractions);
router.get('/medication/:medicationId', getInteractions);
router.post('/', addInteraction);

module.exports = router;




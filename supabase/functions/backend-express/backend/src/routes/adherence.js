const express = require('express');
const router = express.Router();
const {
  getAdherence,
  logAdherence,
  getAdherenceCalendar
} = require('../controllers/adherenceController');

router.get('/', getAdherence);
router.post('/', logAdherence);
router.get('/calendar', getAdherenceCalendar);

module.exports = router;




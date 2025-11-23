const express = require('express');
const router = express.Router();
const {
  getMedicationsWithWarnings
} = require('../controllers/medicationTrackingController');

router.get('/with-warnings', getMedicationsWithWarnings);

module.exports = router;




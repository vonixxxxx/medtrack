const express = require('express');
const router = express.Router();
const { getPolySE } = require('../controllers/polypharmacyController');

router.get('/', getPolySE);

module.exports = router;




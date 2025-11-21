const express = require('express');
const router = express.Router();
const { getMonoSE } = require('../controllers/monopharmacyController');

router.get('/', getMonoSE);

module.exports = router;




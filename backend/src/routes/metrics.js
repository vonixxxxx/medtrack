const express = require('express');
const router = express.Router();
const { authenticateToken } = require('../middleware/authMiddleware');
const metricController = require('../controllers/metricController');

router.use(authenticateToken);

router.get('/', metricController.getAll);
router.post('/', metricController.create);
router.put('/:id', metricController.update);
router.delete('/:id', metricController.remove);

module.exports = router;

const express = require('express');
const router = express.Router();
const auth = require('../middleware/authMiddleware');
const metricController = require('../controllers/metricController');

router.use(auth);

router.get('/', metricController.getMetricLogs);
router.post('/', metricController.create);
router.put('/:id', metricController.update);
router.delete('/:id', metricController.remove);
router.get('/logs', metricController.getMetricLogs);

module.exports = router;

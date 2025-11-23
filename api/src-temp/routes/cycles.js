const express = require('express');
const router = express.Router();
const auth = require('../middleware/authMiddleware');
const cycleController = require('../controllers/cycleController');
const reportController = require('../controllers/reportController');

router.use(auth);

router.get('/', cycleController.list);
router.post('/', cycleController.create);
router.get('/upcoming', cycleController.upcoming);
router.get('/today', cycleController.todaysDoses);
router.get('/metric-reminders', cycleController.metricReminders);
router.get('/:id', cycleController.get);
router.put('/:id', cycleController.update);
router.delete('/:id', cycleController.remove);

// Metrics
router.get('/:id/metrics', cycleController.listMetrics);
router.post('/:id/metrics', cycleController.addMetric);
router.get('/:id/metrics-due', cycleController.checkMetricsDue);

// Dose log
router.post('/:id/dose', cycleController.markDose);

// Export
router.get('/:id/report/csv', reportController.csv);

module.exports = router;

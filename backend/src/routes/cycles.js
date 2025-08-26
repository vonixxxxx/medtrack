const express = require('express');
const router = express.Router();
const { authenticateToken } = require('../middleware/authMiddleware');
const cycleController = require('../controllers/cycleController');
const reportController = require('../controllers/reportController');

router.use(authenticateToken);

router.get('/', cycleController.list);
router.post('/', cycleController.create);
router.put('/:id', cycleController.update);
router.delete('/:id', cycleController.remove);
router.get('/:id/metrics', cycleController.listMetrics);
router.post('/:id/metrics', cycleController.addMetric);
router.get('/:id/doses', cycleController.todaysDoses);
router.post('/:id/doses', cycleController.markDose);
router.get('/upcoming', cycleController.upcoming);

// Export
router.get('/:id/report/csv', reportController.csv);

module.exports = router;

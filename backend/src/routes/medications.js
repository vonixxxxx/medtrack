const express = require('express');
const router = express.Router();
const auth = require('../middleware/authMiddleware');
const medController = require('../controllers/medicationController');

router.use(auth);

router.get('/', medController.getAll);
router.post('/', medController.create);
router.put('/:id', medController.update);
router.delete('/:id', medController.remove);
router.post('/:id/logs', medController.addLog);

module.exports = router;

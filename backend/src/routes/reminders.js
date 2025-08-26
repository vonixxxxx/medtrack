const express = require('express');
const router = express.Router();
const { authenticateToken } = require('../middleware/authMiddleware');
const reminderController = require('../controllers/reminderController');

router.use(authenticateToken);

router.get('/', reminderController.getUnread);
router.patch('/:id/read', reminderController.markRead);

module.exports = router;

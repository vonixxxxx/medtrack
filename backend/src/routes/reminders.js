const express = require('express');
const router = express.Router();
const auth = require('../middleware/authMiddleware');
const reminderController = require('../controllers/reminderController');

router.use(auth);

router.get('/', reminderController.getUnread);
router.patch('/:id/read', reminderController.markRead);

module.exports = router;

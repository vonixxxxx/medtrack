const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');
const { authenticateToken } = require('../middleware/authMiddleware');

// Public routes
router.post('/signup', authController.signup);
router.post('/login', authController.login);
router.post('/forgot-password', authController.forgotPassword);
router.post('/reset-password', authController.resetPassword);

// Protected routes (require authentication)
router.get('/me', authenticateToken, authController.getCurrentUser);
router.post('/change-password', authenticateToken, authController.changePassword);
router.post('/2fa/generate', authenticateToken, authController.generate2FA);
router.post('/2fa/verify', authenticateToken, authController.verify2FA);
router.post('/2fa/disable', authenticateToken, authController.disable2FA);

module.exports = router;

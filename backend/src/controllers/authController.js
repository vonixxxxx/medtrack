const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const nodemailer = require('nodemailer');
const speakeasy = require('speakeasy');
const QRCode = require('qrcode');
const Joi = require('joi');

const SALT_ROUNDS = 12; // Increased for better security

// Configure email transporter
const createEmailTransporter = () => {
  if (!process.env.SMTP_HOST) {
    console.warn('SMTP not configured - email features disabled');
    return null;
  }
  
  return nodemailer.createTransporter({
    host: process.env.SMTP_HOST,
    port: parseInt(process.env.SMTP_PORT) || 587,
    secure: false, // true for 465, false for other ports
    auth: {
      user: process.env.SMTP_USER,
      pass: process.env.SMTP_PASS,
    },
  });
};

exports.signup = async (req, res) => {
  console.log('Signup request received:', req.body);
  const { email, password, role = 'patient', hospitalCode } = req.body;
  
  if (!email || !password) {
    console.log('Missing email or password');
    return res.status(400).json({ error: 'Email and password required' });
  }

  // Validate role
  if (role && !['patient', 'clinician'].includes(role)) {
    return res.status(400).json({ error: 'Invalid role. Must be "patient" or "clinician"' });
  }

  // Validate hospitalCode - now required for both patients and clinicians
  if (!hospitalCode) {
    return res.status(400).json({ error: 'Hospital code is required.' });
  }

  // Validate hospital code format (must be exactly 123456789 for now)
  const VALID_HOSPITAL_CODE = '123456789';
  if (hospitalCode !== VALID_HOSPITAL_CODE) {
    return res.status(400).json({ error: 'Invalid hospital code. Please contact your institution.' });
  }

  const prisma = req.prisma;

  try {
    const existingUser = await prisma.user.findUnique({ where: { email } });
    if (existingUser) {
      console.log('User already exists:', email);
      return res.status(400).json({ error: 'User already exists' });
    }

    const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
    console.log('Creating user with email:', email, 'role:', role, 'hospitalCode:', hospitalCode);
    console.log('hospitalCode type:', typeof hospitalCode, 'value:', hospitalCode);
    console.log('hospitalCode is null?', hospitalCode === null);
    console.log('hospitalCode is undefined?', hospitalCode === undefined);
    
    const userData = { 
      email, 
      password: hashedPassword, 
      role,
      hospitalCode: hospitalCode
    };
    console.log('User data to create:', userData);
    
    const user = await prisma.user.create({
      data: userData,
    });

    const token = jwt.sign(
      { id: user.id, email: user.email, role: user.role, hospitalCode: user.hospitalCode },
      process.env.JWT_SECRET || 'supersecret',
      { expiresIn: '7d' }
    );

    console.log('User created successfully:', user.id);
    res.status(201).json({ 
      token, 
      user: { 
        id: user.id, 
        email: user.email, 
        role: user.role, 
        hospitalCode: user.hospitalCode 
      } 
    });
  } catch (err) {
    console.error('Signup error:', err);
    res.status(500).json({ error: 'Signup failed', details: err.message });
  }
};

exports.login = async (req, res) => {
  const { email, password } = req.body;
  const prisma = req.prisma;

  try {
    const user = await prisma.user.findUnique({ where: { email } });
    if (!user) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign(
      { id: user.id, email: user.email, role: user.role, hospitalCode: user.hospitalCode },
      process.env.JWT_SECRET || 'supersecret',
      { expiresIn: '7d' }
    );

    res.json({ 
      token, 
      user: { 
        id: user.id, 
        email: user.email, 
        role: user.role, 
        hospitalCode: user.hospitalCode,
        is2FAEnabled: user.is2FAEnabled || false 
      } 
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Login failed' });
  }
};

// Validation schemas
const forgotPasswordSchema = Joi.object({
  email: Joi.string().email().required()
});

const resetPasswordSchema = Joi.object({
  token: Joi.string().required(),
  newPassword: Joi.string().min(8).required()
});

// Forgot Password - Sends reset email
exports.forgotPassword = async (req, res) => {
  try {
    // Validate input
    const { error, value } = forgotPasswordSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { email } = value;
    const prisma = req.prisma;
    
    const user = await prisma.user.findUnique({ where: { email } });
    
    // Always return success to prevent email enumeration
    if (!user) {
      return res.status(200).json({ 
        message: 'If an account with that email exists, a reset link has been sent' 
      });
    }
    
    // Generate secure reset token
    const resetToken = crypto.randomBytes(32).toString('hex');
    const resetTokenExpiry = new Date(Date.now() + 3600000); // 1 hour
    
    await prisma.user.update({
      where: { email },
      data: { 
        resetToken, 
        resetTokenExpiry 
      },
    });
    
    // Send email if SMTP is configured
    const transporter = createEmailTransporter();
    if (transporter) {
      const resetUrl = `${process.env.FRONTEND_URL || 'http://localhost:3000'}/reset-password?token=${resetToken}`;
      
      const emailHtml = `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #3b82f6;">Password Reset Request</h2>
          <p>You requested to reset your password for your MedTrack account.</p>
          <p>Click the button below to reset your password:</p>
          <a href="${resetUrl}" style="background-color: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 16px 0;">
            Reset Password
          </a>
          <p>This link expires in 1 hour.</p>
          <p>If you didn't request this, please ignore this email.</p>
          <hr style="margin: 24px 0; border: 1px solid #e5e7eb;">
          <p style="color: #6b7280; font-size: 14px;">MedTrack - Medication Tracking App</p>
        </div>
      `;
      
      await transporter.sendMail({
        from: process.env.SMTP_FROM || 'noreply@medtrack.com',
        to: user.email,
        subject: 'Reset Your MedTrack Password',
        html: emailHtml,
      });
    }
    
    res.status(200).json({ 
      message: 'If an account with that email exists, a reset link has been sent' 
    });
  } catch (err) {
    console.error('Forgot password error:', err);
    res.status(500).json({ error: 'Password reset request failed' });
  }
};

// Reset Password - Updates password using token
exports.resetPassword = async (req, res) => {
  try {
    // Validate input
    const { error, value } = resetPasswordSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { token, newPassword } = value;
    const prisma = req.prisma;
    
    const user = await prisma.user.findFirst({
      where: {
        resetToken: token,
        resetTokenExpiry: { gt: new Date() },
      },
    });
    
    if (!user) {
      return res.status(400).json({ error: 'Invalid or expired reset token' });
    }
    
    const hashedPassword = await bcrypt.hash(newPassword, SALT_ROUNDS);
    
    await prisma.user.update({
      where: { id: user.id },
      data: {
        password: hashedPassword,
        resetToken: null,
        resetTokenExpiry: null,
      },
    });
    
    res.status(200).json({ message: 'Password updated successfully' });
  } catch (err) {
    console.error('Reset password error:', err);
    res.status(500).json({ error: 'Password reset failed' });
  }
};

// Generate 2FA Secret and QR Code
exports.generate2FA = async (req, res) => {
  try {
    const userId = req.user.id;
    const prisma = req.prisma;
    
    const user = await prisma.user.findUnique({ where: { id: userId } });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    // Generate secret
    const secret = speakeasy.generateSecret({
      name: `MedTrack (${user.email})`,
      issuer: process.env.TOTP_ISSUER || 'MedTrack',
      length: 32,
    });
    
    // Generate QR code
    const qrCodeUrl = await QRCode.toDataURL(secret.otpauth_url);
    
    // Store the secret temporarily (not enabled until verified)
    await prisma.user.update({
      where: { id: userId },
      data: { twoFASecret: secret.base32 }
    });
    
    res.json({
      secret: secret.base32,
      qrCode: qrCodeUrl,
      manualEntryKey: secret.base32
    });
  } catch (err) {
    console.error('Generate 2FA error:', err);
    res.status(500).json({ error: '2FA setup failed' });
  }
};

// Verify and Enable 2FA
exports.verify2FA = async (req, res) => {
  try {
    const { token } = req.body;
    const userId = req.user.id;
    const prisma = req.prisma;
    
    if (!token) {
      return res.status(400).json({ error: 'Token required' });
    }
    
    const user = await prisma.user.findUnique({ where: { id: userId } });
    if (!user || !user.twoFASecret) {
      return res.status(400).json({ error: 'No 2FA setup found' });
    }
    
    // Verify the token
    const verified = speakeasy.totp.verify({
      secret: user.twoFASecret,
      encoding: 'base32',
      token: token,
      window: parseInt(process.env.TOTP_WINDOW) || 2,
    });
    
    if (!verified) {
      return res.status(400).json({ error: 'Invalid token' });
    }
    
    // Enable 2FA
    await prisma.user.update({
      where: { id: userId },
      data: { is2FAEnabled: true }
    });
    
    res.json({ message: '2FA enabled successfully' });
  } catch (err) {
    console.error('Verify 2FA error:', err);
    res.status(500).json({ error: '2FA verification failed' });
  }
};

// Disable 2FA
exports.disable2FA = async (req, res) => {
  try {
    const { token } = req.body;
    const userId = req.user.id;
    const prisma = req.prisma;
    
    if (!token) {
      return res.status(400).json({ error: 'Token required to disable 2FA' });
    }
    
    const user = await prisma.user.findUnique({ where: { id: userId } });
    if (!user || !user.is2FAEnabled) {
      return res.status(400).json({ error: '2FA not enabled' });
    }
    
    // Verify the token before disabling
    const verified = speakeasy.totp.verify({
      secret: user.twoFASecret,
      encoding: 'base32',
      token: token,
      window: parseInt(process.env.TOTP_WINDOW) || 2,
    });
    
    if (!verified) {
      return res.status(400).json({ error: 'Invalid token' });
    }
    
    // Disable 2FA
    await prisma.user.update({
      where: { id: userId },
      data: { 
        is2FAEnabled: false,
        twoFASecret: null
      }
    });
    
    res.json({ message: '2FA disabled successfully' });
  } catch (err) {
    console.error('Disable 2FA error:', err);
    res.status(500).json({ error: '2FA disable failed' });
  }
};

// Change Password
exports.changePassword = async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body;
    const userId = req.user.id;
    const prisma = req.prisma;
    
    if (!currentPassword || !newPassword) {
      return res.status(400).json({ error: 'Current and new password required' });
    }
    
    if (newPassword.length < 8) {
      return res.status(400).json({ error: 'New password must be at least 8 characters' });
    }
    
    const user = await prisma.user.findUnique({ where: { id: userId } });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    // Verify current password
    const isMatch = await bcrypt.compare(currentPassword, user.password);
    if (!isMatch) {
      return res.status(400).json({ error: 'Current password is incorrect' });
    }
    
    // Update password
    const hashedPassword = await bcrypt.hash(newPassword, SALT_ROUNDS);
    await prisma.user.update({
      where: { id: userId },
      data: { password: hashedPassword }
    });
    
    res.json({ message: 'Password changed successfully' });
  } catch (err) {
    console.error('Change password error:', err);
    res.status(500).json({ error: 'Password change failed' });
  }
};

// Get current user info
exports.getCurrentUser = async (req, res) => {
  try {
    const userId = req.user.id;
    const prisma = req.prisma;
    
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        hospitalCode: true,
        is2FAEnabled: true,
        surveyCompleted: true,
        // Don't include sensitive fields like password, resetToken, etc.
      }
    });
    
    if (user) {
      // Ensure boolean fields have defaults
      user.is2FAEnabled = user.is2FAEnabled || false;
      user.surveyCompleted = user.surveyCompleted || false;
    }
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json(user);
  } catch (err) {
    console.error('Get current user error:', err);
    res.status(500).json({ error: 'Failed to get user info' });
  }
};

// Alias for getCurrentUser (for compatibility)
exports.me = exports.getCurrentUser;

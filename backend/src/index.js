const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { PrismaClient } = require('@prisma/client');
const cron = require('node-cron');
const { startOfDay } = require('date-fns');

// Load environment variables from .env if present
dotenv.config();

// Import routes
const authRoutes = require('./routes/auth');
const medicationRoutes = require('./routes/medications');
const cycleRoutes = require('./routes/cycles');
const reminderRoutes = require('./routes/reminders');
const metricRoutes = require('./routes/metrics');

// Import security middleware
const { generalLimiter, authLimiter, passwordResetLimiter, helmetConfig } = require('./middleware/security');

const app = express();
const prisma = new PrismaClient();

// Security middleware
app.use(helmetConfig);

// REPLACE lines 27-33 with enhanced CORS logic
const allowedOrigins = (process.env.CORS_ORIGIN || 'http://localhost:3000')
  .split(',')
  .map((o) => o.trim());

const corsOptions = {
  origin: (origin, callback) => {
    // Allow requests with no origin (e.g. mobile apps, curl) or in the whitelist
    if (!origin || allowedOrigins.includes(origin)) {
      return callback(null, true);
    }
    return callback(new Error('Not allowed by CORS'));
  },
  credentials: true,
  optionsSuccessStatus: 200,
};
app.use(cors(corsOptions));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// General rate limiting for all routes
app.use(generalLimiter);

// Attach Prisma instance to each request for easy access in controllers
app.use((req, _res, next) => {
  req.prisma = prisma;
  next();
});

// Health check endpoint (before rate limiting)
app.get('/health', async (req, res) => {
  try {
    // Check database connection
    await prisma.$queryRaw`SELECT 1`;
    res.status(200).json({ 
      status: 'healthy', 
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV || 'development',
      version: process.env.npm_package_version || '1.0.0'
    });
  } catch (error) {
    console.error('Health check failed:', error);
    res.status(503).json({ 
      status: 'unhealthy', 
      timestamp: new Date().toISOString(),
      error: 'Database connection failed'
    });
  }
});

// ADD after the /health endpoint (around line 67)
app.get('/api/health', async (req, res) => {
  res.redirect('/health');
});

// Simple root route for convenience
app.get('/', (_req, res) => {
  res.json({ status: 'running', message: 'MedTrack API' });
});

// API Routes with specific rate limiting
app.use('/api/auth/login', authLimiter);
app.use('/api/auth/signup', authLimiter);
app.use('/api/auth/forgot-password', passwordResetLimiter);
app.use('/api/auth/reset-password', passwordResetLimiter);

// Protected API routes
app.use('/api/auth', authRoutes);
app.use('/api/medications', medicationRoutes);
app.use('/api/cycles', cycleRoutes);
app.use('/api/upcoming', cycleRoutes); // convenience endpoint
app.use('/api/reminders', reminderRoutes);
app.use('/api/metrics', metricRoutes);

// Global error handler
app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: 'Internal Server Error' });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// Cron job: run every day at 08:00 server time to create daily reminders
cron.schedule('0 8 * * *', async () => {
  console.log('Running daily medication reminder check');

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  try {
    // Fetch medications active today
    const medications = await prisma.medication.findMany({
      where: {
        startDate: { lte: today },
        endDate: { gte: today },
      },
    });

    for (const med of medications) {
      // Prevent duplicate reminders for the same day
      const existing = await prisma.reminder.findFirst({
        where: {
          userId: med.userId,
          medicationId: med.id,
          date: today,
        },
      });

      if (!existing) {
        await prisma.reminder.create({
          data: {
            userId: med.userId,
            medicationId: med.id,
            date: today,
            message: `Take ${med.name} today`,
          },
        });
      }
    }

    console.log('Daily reminders created');
  } catch (err) {
    console.error('Error creating reminders', err);
  }
});

// Daily email at 7am
const nodemailer = require('nodemailer');
const transporter = nodemailer.createTransport({
  // For demo use Ethereal; replace with real SMTP in prod
  host: 'smtp.ethereal.email',
  port: 587,
  auth: {
    user: process.env.ETHEREAL_USER,
    pass: process.env.ETHEREAL_PASS,
  },
});

cron.schedule('0 7 * * *', async () => {
  console.log('Sending daily medication emails');
  const today = startOfDay(new Date());
  const due = await prisma.doseLog.findMany({
    where: { date: today, taken: false },
    include: { cycle: { include: { user: true } } },
  });
  for (const d of due) {
    if (!d.cycle.user.email) continue;
    const mailOptions = {
      from: 'no-reply@medtrack.local',
      to: d.cycle.user.email,
      subject: 'Medication Reminder',
      text: `Hello ${d.cycle.user.name || ''},\nToday is the day to take ${d.cycle.dosage} of ${d.cycle.name}. Did you take it?`,
    };
    try {
      await transporter.sendMail(mailOptions);
    } catch (err) {
      console.error('Email error', err);
    }
  }
});

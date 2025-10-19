const express = require('express');
const cors = require('cors');
const compression = require('compression');
const morgan = require('morgan');
const dotenv = require('dotenv');
const { PrismaClient } = require('@prisma/client');
const { createServer } = require('http');
const { Server } = require('socket.io');
const cron = require('node-cron');
const winston = require('winston');
const Redis = require('ioredis');

// Load environment variables
dotenv.config();

// Import security middleware
const {
  securityHeaders,
  authRateLimit,
  generalRateLimit,
  strictRateLimit,
  sanitizeInput,
  handleValidationErrors,
  auditLogger,
  requestSizeLimit,
  logger
} = require('./middleware/security');

// Import routes
const authRoutes = require('./routes/auth');
const medicationRoutes = require('./routes/medications');
const metricRoutes = require('./routes/metrics');
// const screeningRoutes = require('./routes/screening');
const reminderRoutes = require('./routes/reminders');
// const dashboardRoutes = require('./routes/dashboard');
// const adherenceRoutes = require('./routes/adherence');
const aiRoutes = require('./routes/ai');
const enhancedMedicationValidationRoutes = require('./routes/enhancedMedicationValidation');
const surveyRoutes = require('./routes/survey');
const doctorRoutes = require('./routes/doctor');

// Import services
// const notificationService = require('./services/notificationService');
// const reminderService = require('./services/reminderService');
// const auditService = require('./services/auditService');

// Initialize Prisma
const prisma = new PrismaClient({
  log: ['query', 'info', 'warn', 'error'],
});

// Initialize Redis for caching and sessions (optional)
const redis = process.env.REDIS_HOST ? new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD,
  retryDelayOnFailover: 100,
  maxRetriesPerRequest: 3,
  lazyConnect: true
}) : null;

// Create Express app
const app = express();
const server = createServer(app);

// Initialize Socket.IO for real-time features
const io = new Server(server, {
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST"],
    credentials: true
  }
});

// Configure Winston logger
const winstonLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'medtrack-api' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Make logger available globally
global.logger = winstonLogger;

// Security middleware
app.use(securityHeaders);
app.use(compression());
app.use(requestSizeLimit(10 * 1024 * 1024)); // 10MB limit

// CORS configuration
const allowedOrigins = (process.env.CORS_ORIGIN || 'http://localhost:3000')
  .split(',')
  .map(origin => origin.trim());

const corsOptions = {
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));

// Request logging
app.use(morgan('combined', {
  stream: {
    write: (message) => winstonLogger.info(message.trim())
  }
}));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Input sanitization
app.use(sanitizeInput);

// Rate limiting
app.use('/api/auth', authRateLimit);
app.use('/api', generalRateLimit);

// Make Prisma and Redis available to routes
app.use((req, res, next) => {
  req.prisma = prisma;
  req.redis = redis;
  req.io = io;
  next();
});

// Test endpoint
app.get('/test-public', (req, res) => {
  res.json({ message: 'Public endpoint working!' });
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    // Check database connection
    await prisma.$queryRaw`SELECT 1`;
    
    // Check Redis connection (if available) - don't fail if Redis is not available
    let redisStatus = 'not configured';
    if (redis) {
      try {
        await redis.ping();
        redisStatus = 'connected';
      } catch (redisError) {
        winstonLogger.warn('Redis connection failed', { error: redisError.message });
        redisStatus = 'disconnected';
      }
    }
    
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        database: 'connected',
        redis: redisStatus,
        socketio: 'connected'
      }
    });
  } catch (error) {
    winstonLogger.error('Health check failed', { error: error.message });
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: error.message
    });
  }
});

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/meds', medicationRoutes);
app.use('/api/metrics', metricRoutes);
// app.use('/api/screening', screeningRoutes);
app.use('/api/reminders', reminderRoutes);
// app.use('/api/dashboard', dashboardRoutes);
// app.use('/api/adherence', adherenceRoutes);
app.use('/api/ai', aiRoutes);
app.use('/api/medications', enhancedMedicationValidationRoutes);
app.use('/api/auth', surveyRoutes);
app.use('/api/doctor', doctorRoutes);

// Socket.IO connection handling
io.on('connection', (socket) => {
  winstonLogger.info('Client connected', { socketId: socket.id });

  // Join user-specific room for notifications
  socket.on('join-user-room', (userId) => {
    socket.join(`user-${userId}`);
    winstonLogger.info('User joined room', { userId, socketId: socket.id });
  });

  // Handle medication reminders
  socket.on('medication-reminder-response', async (data) => {
    try {
      const { medicationId, response, userId } = data;
      
      // Log the response
      await prisma.medicationLog.create({
        data: {
          medicationId,
          userId,
          takenAt: new Date(),
          notes: `Reminder response: ${response}`,
          verified: true
        }
      });

      // Notify other connected devices
      socket.to(`user-${userId}`).emit('medication-taken', {
        medicationId,
        timestamp: new Date()
      });

      winstonLogger.info('Medication reminder response', { 
        userId, 
        medicationId, 
        response 
      });
    } catch (error) {
      winstonLogger.error('Error handling medication reminder response', { 
        error: error.message 
      });
    }
  });

  socket.on('disconnect', () => {
    winstonLogger.info('Client disconnected', { socketId: socket.id });
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  winstonLogger.error('Unhandled error', {
    error: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
    ip: req.ip
  });

  res.status(error.status || 500).json({
    error: process.env.NODE_ENV === 'production' 
      ? 'Internal server error' 
      : error.message,
    ...(process.env.NODE_ENV !== 'production' && { stack: error.stack })
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.originalUrl,
    method: req.method
  });
});

// Scheduled tasks
cron.schedule('0 8 * * *', async () => {
  // Daily medication reminders at 8 AM
  try {
    await reminderService.sendDailyReminders(prisma, io);
    winstonLogger.info('Daily medication reminders sent');
  } catch (error) {
    winstonLogger.error('Error sending daily reminders', { error: error.message });
  }
});

cron.schedule('0 0 * * 0', async () => {
  // Weekly health metric reminders on Sundays
  try {
    await reminderService.sendWeeklyMetricReminders(prisma, io);
    winstonLogger.info('Weekly metric reminders sent');
  } catch (error) {
    winstonLogger.error('Error sending weekly reminders', { error: error.message });
  }
});

cron.schedule('0 2 * * *', async () => {
  // Daily cleanup of expired sessions at 2 AM
  try {
    const result = await prisma.userSession.deleteMany({
      where: {
        expiresAt: {
          lt: new Date()
        }
      }
    });
    winstonLogger.info('Expired sessions cleaned up', { count: result.count });
  } catch (error) {
    winstonLogger.error('Error cleaning up expired sessions', { error: error.message });
  }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  winstonLogger.info('SIGTERM received, shutting down gracefully');
  
  server.close(() => {
    winstonLogger.info('HTTP server closed');
  });

  await prisma.$disconnect();
  await redis.disconnect();
  
  process.exit(0);
});

process.on('SIGINT', async () => {
  winstonLogger.info('SIGINT received, shutting down gracefully');
  
  server.close(() => {
    winstonLogger.info('HTTP server closed');
  });

  await prisma.$disconnect();
  await redis.disconnect();
  
  process.exit(0);
});

// Start server
const PORT = process.env.PORT || 4000;
const HOST = process.env.HOST || '0.0.0.0';

server.listen(PORT, HOST, () => {
  winstonLogger.info(`MedTrack API server running on ${HOST}:${PORT}`);
  winstonLogger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
  winstonLogger.info(`Database: ${process.env.DATABASE_URL ? 'Connected' : 'Not configured'}`);
  winstonLogger.info(`Redis: ${process.env.REDIS_HOST || 'localhost'}:${process.env.REDIS_PORT || 6379}`);
});

module.exports = { app, server, io };
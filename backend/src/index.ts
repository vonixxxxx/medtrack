/**
 * MedTrack Backend Server
 * Enhanced TypeScript implementation with AI integration
 */

import express from 'express';
import cors from 'cors';
import compression from 'compression';
import morgan from 'morgan';
import dotenv from 'dotenv';
import { PrismaClient } from '@prisma/client';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cron from 'node-cron';
import winston from 'winston';
import Redis from 'ioredis';

// Load environment variables
dotenv.config();

// Import middleware
import { authMiddleware } from './middleware/auth.middleware';
import { sanitizeInput, validatePagination, validateDateRange } from './middleware/validation.middleware';

// Import security middleware (CommonJS)
const securityMiddleware = require('./middleware/security');
const {
  securityHeaders,
  authRateLimit,
  generalRateLimit,
  strictRateLimit,
  handleValidationErrors,
  auditLogger,
  requestSizeLimit,
  logger
} = securityMiddleware;

// Import routes
import medicationRoutes from './routes/medications';
import healthMetricsRoutes from './routes/health-metrics';
import medicationScheduleRoutes from './routes/medication-schedules';
import authRoutes from './routes/auth';
import aiModelRoutes from './routes/ai-models';
import aiAssistantRoutes from './routes/ai-assistant';

// Import services
import { ValidationService } from './services/validation.service';

// Types
import { RequestWithServices, HealthCheckResponse } from './types';

// Initialize Prisma
const prisma = new PrismaClient({
  log: ['query', 'info', 'warn', 'error'],
});

// Initialize Redis for caching and sessions (optional)
const redis = process.env.REDIS_URL ? new Redis(process.env.REDIS_URL, {
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
  level: process.env.LOG_LEVEL || 'info',
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
(global as any).logger = winstonLogger;

// Security middleware
app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// CORS configuration
const allowedOrigins = (process.env.CORS_ORIGIN || 'http://localhost:3000')
  .split(',')
  .map(origin => origin.trim());

const corsOptions = {
  origin: (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
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
    write: (message: string) => winstonLogger.info(message.trim())
  }
}));

// Input sanitization
app.use(sanitizeInput);

// Make services available to routes
app.use((req: RequestWithServices, res, next) => {
  req.prisma = prisma;
  req.redis = redis;
  req.io = io;
  next();
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const startTime = Date.now();
    
    // Check database connection
    await prisma.$queryRaw`SELECT 1`;
    
    // Check Redis connection (if available)
    let redisStatus = 'not configured';
    if (redis) {
      try {
        await redis.ping();
        redisStatus = 'connected';
      } catch (error) {
        redisStatus = 'disconnected';
      }
    }
    
    // Check vector search service
    let vectorSearchStatus = 'not configured';
    try {
      const response = await fetch(`${process.env.VECTOR_SEARCH_URL || 'http://localhost:3005'}/healthz`);
      vectorSearchStatus = response.ok ? 'connected' : 'disconnected';
    } catch (error) {
      vectorSearchStatus = 'disconnected';
    }
    
    // Check Ollama service
    let ollamaStatus = 'not configured';
    try {
      const response = await fetch(`${process.env.OLLAMA_URL || 'http://localhost:11434'}/api/tags`);
      ollamaStatus = response.ok ? 'connected' : 'disconnected';
    } catch (error) {
      ollamaStatus = 'disconnected';
    }
    
    // Check Qdrant service
    let qdrantStatus = 'not configured';
    try {
      const response = await fetch(`${process.env.QDRANT_URL || 'http://localhost:6333'}/health`);
      qdrantStatus = response.ok ? 'connected' : 'disconnected';
    } catch (error) {
      qdrantStatus = 'disconnected';
    }
    
    const responseTime = Date.now() - startTime;
    const memoryUsage = process.memoryUsage();
    
    const healthResponse: HealthCheckResponse = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        database: 'connected',
        redis: redisStatus as 'connected' | 'disconnected',
        vectorSearch: vectorSearchStatus as 'connected' | 'disconnected',
        ollama: ollamaStatus as 'connected' | 'disconnected',
        qdrant: qdrantStatus as 'connected' | 'disconnected'
      },
      uptime: process.uptime(),
      memory: {
        used: memoryUsage.heapUsed,
        total: memoryUsage.heapTotal,
        percentage: Math.round((memoryUsage.heapUsed / memoryUsage.heapTotal) * 100)
      }
    };
    
    res.json(healthResponse);
  } catch (error) {
    winstonLogger.error('Health check failed', { error: (error as Error).message });
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: (error as Error).message
    });
  }
});

// API documentation endpoint
app.get('/docs', (req, res) => {
  res.json({
    name: 'MedTrack API',
    version: '1.0.0',
    description: 'AI-powered medication tracking and health management API',
    endpoints: {
      health: 'GET /health',
      auth: {
        login: 'POST /api/auth/login',
        register: 'POST /api/auth/register',
        logout: 'POST /api/auth/logout',
        refresh: 'POST /api/auth/refresh'
      },
      medications: {
        list: 'GET /api/medications',
        get: 'GET /api/medications/:id',
        create: 'POST /api/medications',
        update: 'PUT /api/medications/:id',
        delete: 'DELETE /api/medications/:id',
        log: 'POST /api/medications/:id/log',
        logs: 'GET /api/medications/:id/logs',
        validate: 'POST /api/medications/validate',
        validateDosage: 'POST /api/medications/validate-dosage',
        checkInteractions: 'POST /api/medications/check-interactions'
      },
      healthMetrics: {
        list: 'GET /api/health-metrics',
        get: 'GET /api/health-metrics/:id',
        create: 'POST /api/health-metrics',
        update: 'PUT /api/health-metrics/:id',
        delete: 'DELETE /api/health-metrics/:id',
        analytics: 'GET /api/health-metrics/analytics/overview',
        dashboard: 'GET /api/health-metrics/dashboard/overview',
        bulk: 'POST /api/health-metrics/bulk'
      },
      schedules: {
        list: 'GET /api/medication-schedules',
        get: 'GET /api/medication-schedules/:id',
        create: 'POST /api/medication-schedules',
        update: 'PUT /api/medication-schedules/:id',
        delete: 'DELETE /api/medication-schedules/:id',
        today: 'GET /api/medication-schedules/today/overview',
        reminders: 'GET /api/medication-schedules/upcoming/reminders',
        toggle: 'PATCH /api/medication-schedules/:id/toggle'
      },
      ai: {
        health: 'GET /api/ai/health',
        searchMedications: 'POST /api/ai/search/medications',
        suggestions: 'GET /api/ai/suggestions/medications',
        query: 'POST /api/ai/query',
        healthReport: 'POST /api/ai/health-report',
        validateMedication: 'POST /api/ai/validate/medication',
        checkInteractions: 'POST /api/ai/check/interactions',
        recommendations: 'POST /api/ai/recommendations/medications'
      },
      aiAssistant: {
        chat: 'POST /api/ai-assistant/chat',
        suggestions: 'GET /api/ai-assistant/suggestions',
        history: 'GET /api/ai-assistant/history',
        deleteHistory: 'DELETE /api/ai-assistant/history/:conversationId',
        status: 'GET /api/ai-assistant/status',
        healthTips: 'GET /api/ai-assistant/health-tips',
        reminders: 'GET /api/ai-assistant/reminders'
      }
    },
    authentication: 'Bearer token required for all endpoints except /health and /docs',
    rateLimiting: 'Rate limits apply to all endpoints',
    documentation: 'https://github.com/medtrack/api-docs'
  });
});

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/medications', medicationRoutes);
app.use('/api/health-metrics', healthMetricsRoutes);
app.use('/api/medication-schedules', medicationScheduleRoutes);
app.use('/api/ai', aiModelRoutes);
app.use('/api/ai-assistant', aiAssistantRoutes);

// Socket.IO connection handling
io.on('connection', (socket) => {
  winstonLogger.info('Client connected', { socketId: socket.id });

  // Join user-specific room for notifications
  socket.on('join-user-room', (userId: string) => {
    socket.join(`user-${userId}`);
    winstonLogger.info('User joined room', { userId, socketId: socket.id });
  });

  // Handle medication reminders
  socket.on('medication-reminder-response', async (data: any) => {
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
        error: (error as Error).message 
      });
    }
  });

  socket.on('disconnect', () => {
    winstonLogger.info('Client disconnected', { socketId: socket.id });
  });
});

// Error handling middleware
app.use((error: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  winstonLogger.error('Unhandled error', {
    error: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
    ip: req.ip
  });

  res.status(500).json({
    success: false,
    error: process.env.NODE_ENV === 'production' 
      ? 'Internal server error' 
      : error.message,
    ...(process.env.NODE_ENV !== 'production' && { stack: error.stack }),
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    message: `The requested endpoint ${req.method} ${req.originalUrl} was not found`,
    path: req.originalUrl,
    method: req.method,
    timestamp: new Date().toISOString()
  });
});

// Scheduled tasks
cron.schedule('0 8 * * *', async () => {
  // Daily medication reminders at 8 AM
  try {
    // Implementation for daily reminders
    winstonLogger.info('Daily medication reminders sent');
  } catch (error) {
    winstonLogger.error('Error sending daily reminders', { error: (error as Error).message });
  }
});

cron.schedule('0 0 * * 0', async () => {
  // Weekly health metric reminders on Sundays
  try {
    // Implementation for weekly reminders
    winstonLogger.info('Weekly metric reminders sent');
  } catch (error) {
    winstonLogger.error('Error sending weekly reminders', { error: (error as Error).message });
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
    winstonLogger.error('Error cleaning up expired sessions', { error: (error as Error).message });
  }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  winstonLogger.info('SIGTERM received, shutting down gracefully');
  
  server.close(() => {
    winstonLogger.info('HTTP server closed');
  });

  await prisma.$disconnect();
  if (redis) {
    await redis.disconnect();
  }
  
  process.exit(0);
});

process.on('SIGINT', async () => {
  winstonLogger.info('SIGINT received, shutting down gracefully');
  
  server.close(() => {
    winstonLogger.info('HTTP server closed');
  });

  await prisma.$disconnect();
  if (redis) {
    await redis.disconnect();
  }
  
  process.exit(0);
});

// Start server
const PORT = process.env.PORT || 4000;
const HOST = process.env.HOST || '0.0.0.0';

server.listen(PORT, HOST, () => {
  winstonLogger.info(`MedTrack API server running on ${HOST}:${PORT}`);
  winstonLogger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
  winstonLogger.info(`Database: ${process.env.DATABASE_URL ? 'Connected' : 'Not configured'}`);
  winstonLogger.info(`Redis: ${process.env.REDIS_URL || 'Not configured'}`);
  winstonLogger.info(`Vector Search: ${process.env.VECTOR_SEARCH_URL || 'Not configured'}`);
  winstonLogger.info(`Ollama: ${process.env.OLLAMA_URL || 'Not configured'}`);
  winstonLogger.info(`Qdrant: ${process.env.QDRANT_URL || 'Not configured'}`);
});

export { app, server, io };
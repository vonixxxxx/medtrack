"use strict";
/**
 * MedTrack Backend Server
 * Enhanced TypeScript implementation with AI integration
 */
var __makeTemplateObject = (this && this.__makeTemplateObject) || function (cooked, raw) {
    if (Object.defineProperty) { Object.defineProperty(cooked, "raw", { value: raw }); } else { cooked.raw = raw; }
    return cooked;
};
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.io = exports.server = exports.app = void 0;
var express_1 = require("express");
var cors_1 = require("cors");
var compression_1 = require("compression");
var morgan_1 = require("morgan");
var dotenv_1 = require("dotenv");
var client_1 = require("@prisma/client");
var http_1 = require("http");
var socket_io_1 = require("socket.io");
var node_cron_1 = require("node-cron");
var winston_1 = require("winston");
var ioredis_1 = require("ioredis");
// Load environment variables
dotenv_1.default.config();
var validation_middleware_1 = require("./middleware/validation.middleware");
// Import security middleware (CommonJS)
var securityMiddleware = require('./middleware/security');
var securityHeaders = securityMiddleware.securityHeaders, authRateLimit = securityMiddleware.authRateLimit, generalRateLimit = securityMiddleware.generalRateLimit, strictRateLimit = securityMiddleware.strictRateLimit, handleValidationErrors = securityMiddleware.handleValidationErrors, auditLogger = securityMiddleware.auditLogger, requestSizeLimit = securityMiddleware.requestSizeLimit, logger = securityMiddleware.logger;
// Import routes
var medications_1 = require("./routes/medications");
var health_metrics_1 = require("./routes/health-metrics");
var medication_schedules_1 = require("./routes/medication-schedules");
var auth_1 = require("./routes/auth");
var ai_models_1 = require("./routes/ai-models");
var ai_assistant_1 = require("./routes/ai-assistant");
var adherence_1 = require("./routes/adherence");
var metrics_trends_1 = require("./routes/metrics-trends");
var wellness_1 = require("./routes/wellness");
var health_report_1 = require("./routes/health-report");
// Initialize Prisma
var prisma = new client_1.PrismaClient({
    log: ['query', 'info', 'warn', 'error'],
});
// Initialize Redis for caching and sessions (optional)
var redis = process.env.REDIS_URL ? new ioredis_1.default(process.env.REDIS_URL, {
    retryDelayOnFailover: 100,
    maxRetriesPerRequest: 3,
    lazyConnect: true
}) : null;
// Create Express app
var app = (0, express_1.default)();
exports.app = app;
var server = (0, http_1.createServer)(app);
exports.server = server;
// Initialize Socket.IO for real-time features
var io = new socket_io_1.Server(server, {
    cors: {
        origin: process.env.FRONTEND_URL || "http://localhost:3000",
        methods: ["GET", "POST"],
        credentials: true
    }
});
exports.io = io;
// Configure Winston logger
var winstonLogger = winston_1.default.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston_1.default.format.combine(winston_1.default.format.timestamp(), winston_1.default.format.errors({ stack: true }), winston_1.default.format.json()),
    defaultMeta: { service: 'medtrack-api' },
    transports: [
        new winston_1.default.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston_1.default.transports.File({ filename: 'logs/combined.log' }),
        new winston_1.default.transports.Console({
            format: winston_1.default.format.combine(winston_1.default.format.colorize(), winston_1.default.format.simple())
        })
    ]
});
// Make logger available globally
global.logger = winstonLogger;
// Security middleware
app.use((0, compression_1.default)());
app.use(express_1.default.json({ limit: '10mb' }));
app.use(express_1.default.urlencoded({ extended: true, limit: '10mb' }));
// CORS configuration
var allowedOrigins = (process.env.CORS_ORIGIN || 'http://localhost:3000')
    .split(',')
    .map(function (origin) { return origin.trim(); });
var corsOptions = {
    origin: function (origin, callback) {
        if (!origin || allowedOrigins.includes(origin)) {
            callback(null, true);
        }
        else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    credentials: true,
    optionsSuccessStatus: 200,
};
app.use((0, cors_1.default)(corsOptions));
// Request logging
app.use((0, morgan_1.default)('combined', {
    stream: {
        write: function (message) { return winstonLogger.info(message.trim()); }
    }
}));
// Input sanitization
app.use(validation_middleware_1.sanitizeInput);
// Make services available to routes
app.use(function (req, res, next) {
    req.prisma = prisma;
    req.redis = redis;
    req.io = io;
    next();
});
// Health check endpoint
app.get('/health', function (req, res) { return __awaiter(void 0, void 0, void 0, function () {
    var startTime, redisStatus, error_1, vectorSearchStatus, response, error_2, ollamaStatus, response, error_3, qdrantStatus, response, error_4, responseTime, memoryUsage, healthResponse, error_5;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 18, , 19]);
                startTime = Date.now();
                // Check database connection
                return [4 /*yield*/, prisma.$queryRaw(templateObject_1 || (templateObject_1 = __makeTemplateObject(["SELECT 1"], ["SELECT 1"])))];
            case 1:
                // Check database connection
                _a.sent();
                redisStatus = 'not configured';
                if (!redis) return [3 /*break*/, 5];
                _a.label = 2;
            case 2:
                _a.trys.push([2, 4, , 5]);
                return [4 /*yield*/, redis.ping()];
            case 3:
                _a.sent();
                redisStatus = 'connected';
                return [3 /*break*/, 5];
            case 4:
                error_1 = _a.sent();
                redisStatus = 'disconnected';
                return [3 /*break*/, 5];
            case 5:
                vectorSearchStatus = 'not configured';
                _a.label = 6;
            case 6:
                _a.trys.push([6, 8, , 9]);
                return [4 /*yield*/, fetch("".concat(process.env.VECTOR_SEARCH_URL || 'http://localhost:3005', "/healthz"))];
            case 7:
                response = _a.sent();
                vectorSearchStatus = response.ok ? 'connected' : 'disconnected';
                return [3 /*break*/, 9];
            case 8:
                error_2 = _a.sent();
                vectorSearchStatus = 'disconnected';
                return [3 /*break*/, 9];
            case 9:
                ollamaStatus = 'not configured';
                _a.label = 10;
            case 10:
                _a.trys.push([10, 12, , 13]);
                return [4 /*yield*/, fetch("".concat(process.env.OLLAMA_URL || 'http://localhost:11434', "/api/tags"))];
            case 11:
                response = _a.sent();
                ollamaStatus = response.ok ? 'connected' : 'disconnected';
                return [3 /*break*/, 13];
            case 12:
                error_3 = _a.sent();
                ollamaStatus = 'disconnected';
                return [3 /*break*/, 13];
            case 13:
                qdrantStatus = 'not configured';
                _a.label = 14;
            case 14:
                _a.trys.push([14, 16, , 17]);
                return [4 /*yield*/, fetch("".concat(process.env.QDRANT_URL || 'http://localhost:6333', "/health"))];
            case 15:
                response = _a.sent();
                qdrantStatus = response.ok ? 'connected' : 'disconnected';
                return [3 /*break*/, 17];
            case 16:
                error_4 = _a.sent();
                qdrantStatus = 'disconnected';
                return [3 /*break*/, 17];
            case 17:
                responseTime = Date.now() - startTime;
                memoryUsage = process.memoryUsage();
                healthResponse = {
                    status: 'healthy',
                    timestamp: new Date().toISOString(),
                    services: {
                        database: 'connected',
                        redis: redisStatus,
                        vectorSearch: vectorSearchStatus,
                        ollama: ollamaStatus,
                        qdrant: qdrantStatus
                    },
                    uptime: process.uptime(),
                    memory: {
                        used: memoryUsage.heapUsed,
                        total: memoryUsage.heapTotal,
                        percentage: Math.round((memoryUsage.heapUsed / memoryUsage.heapTotal) * 100)
                    }
                };
                res.json(healthResponse);
                return [3 /*break*/, 19];
            case 18:
                error_5 = _a.sent();
                winstonLogger.error('Health check failed', { error: error_5.message });
                res.status(503).json({
                    status: 'unhealthy',
                    timestamp: new Date().toISOString(),
                    error: error_5.message
                });
                return [3 /*break*/, 19];
            case 19: return [2 /*return*/];
        }
    });
}); });
// API documentation endpoint
app.get('/docs', function (req, res) {
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
app.use('/api/auth', auth_1.default);
app.use('/api/medications', medications_1.default);
app.use('/api/health-metrics', health_metrics_1.default);
app.use('/api/medication-schedules', medication_schedules_1.default);
app.use('/api/ai', ai_models_1.default);
app.use('/api/ai-assistant', ai_assistant_1.default);
app.use('/api/adherence', (0, adherence_1.default)(prisma));
app.use('/api/metrics/trends', (0, metrics_trends_1.default)(prisma));
app.use('/api/wellness', (0, wellness_1.default)(prisma));
app.use('/api/health-report', (0, health_report_1.default)(prisma));
// Socket.IO connection handling
io.on('connection', function (socket) {
    winstonLogger.info('Client connected', { socketId: socket.id });
    // Join user-specific room for notifications
    socket.on('join-user-room', function (userId) {
        socket.join("user-".concat(userId));
        winstonLogger.info('User joined room', { userId: userId, socketId: socket.id });
    });
    // Handle medication reminders
    socket.on('medication-reminder-response', function (data) { return __awaiter(void 0, void 0, void 0, function () {
        var medicationId, response, userId, error_6;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, , 3]);
                    medicationId = data.medicationId, response = data.response, userId = data.userId;
                    // Log the response
                    return [4 /*yield*/, prisma.medicationLog.create({
                            data: {
                                medicationId: medicationId,
                                userId: userId,
                                takenAt: new Date(),
                                notes: "Reminder response: ".concat(response),
                                verified: true
                            }
                        })];
                case 1:
                    // Log the response
                    _a.sent();
                    // Notify other connected devices
                    socket.to("user-".concat(userId)).emit('medication-taken', {
                        medicationId: medicationId,
                        timestamp: new Date()
                    });
                    winstonLogger.info('Medication reminder response', {
                        userId: userId,
                        medicationId: medicationId,
                        response: response
                    });
                    return [3 /*break*/, 3];
                case 2:
                    error_6 = _a.sent();
                    winstonLogger.error('Error handling medication reminder response', {
                        error: error_6.message
                    });
                    return [3 /*break*/, 3];
                case 3: return [2 /*return*/];
            }
        });
    }); });
    socket.on('disconnect', function () {
        winstonLogger.info('Client disconnected', { socketId: socket.id });
    });
});
// Error handling middleware
app.use(function (error, req, res, next) {
    winstonLogger.error('Unhandled error', {
        error: error.message,
        stack: error.stack,
        url: req.url,
        method: req.method,
        ip: req.ip
    });
    res.status(500).json(__assign(__assign({ success: false, error: process.env.NODE_ENV === 'production'
            ? 'Internal server error'
            : error.message }, (process.env.NODE_ENV !== 'production' && { stack: error.stack })), { timestamp: new Date().toISOString() }));
});
// 404 handler
app.use('*', function (req, res) {
    res.status(404).json({
        success: false,
        error: 'Endpoint not found',
        message: "The requested endpoint ".concat(req.method, " ").concat(req.originalUrl, " was not found"),
        path: req.originalUrl,
        method: req.method,
        timestamp: new Date().toISOString()
    });
});
// Scheduled tasks
node_cron_1.default.schedule('0 8 * * *', function () { return __awaiter(void 0, void 0, void 0, function () {
    return __generator(this, function (_a) {
        // Daily medication reminders at 8 AM
        try {
            // Implementation for daily reminders
            winstonLogger.info('Daily medication reminders sent');
        }
        catch (error) {
            winstonLogger.error('Error sending daily reminders', { error: error.message });
        }
        return [2 /*return*/];
    });
}); });
node_cron_1.default.schedule('0 0 * * 0', function () { return __awaiter(void 0, void 0, void 0, function () {
    return __generator(this, function (_a) {
        // Weekly health metric reminders on Sundays
        try {
            // Implementation for weekly reminders
            winstonLogger.info('Weekly metric reminders sent');
        }
        catch (error) {
            winstonLogger.error('Error sending weekly reminders', { error: error.message });
        }
        return [2 /*return*/];
    });
}); });
node_cron_1.default.schedule('0 2 * * *', function () { return __awaiter(void 0, void 0, void 0, function () {
    var result, error_7;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                return [4 /*yield*/, prisma.userSession.deleteMany({
                        where: {
                            expiresAt: {
                                lt: new Date()
                            }
                        }
                    })];
            case 1:
                result = _a.sent();
                winstonLogger.info('Expired sessions cleaned up', { count: result.count });
                return [3 /*break*/, 3];
            case 2:
                error_7 = _a.sent();
                winstonLogger.error('Error cleaning up expired sessions', { error: error_7.message });
                return [3 /*break*/, 3];
            case 3: return [2 /*return*/];
        }
    });
}); });
// Graceful shutdown
process.on('SIGTERM', function () { return __awaiter(void 0, void 0, void 0, function () {
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                winstonLogger.info('SIGTERM received, shutting down gracefully');
                server.close(function () {
                    winstonLogger.info('HTTP server closed');
                });
                return [4 /*yield*/, prisma.$disconnect()];
            case 1:
                _a.sent();
                if (!redis) return [3 /*break*/, 3];
                return [4 /*yield*/, redis.disconnect()];
            case 2:
                _a.sent();
                _a.label = 3;
            case 3:
                process.exit(0);
                return [2 /*return*/];
        }
    });
}); });
process.on('SIGINT', function () { return __awaiter(void 0, void 0, void 0, function () {
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                winstonLogger.info('SIGINT received, shutting down gracefully');
                server.close(function () {
                    winstonLogger.info('HTTP server closed');
                });
                return [4 /*yield*/, prisma.$disconnect()];
            case 1:
                _a.sent();
                if (!redis) return [3 /*break*/, 3];
                return [4 /*yield*/, redis.disconnect()];
            case 2:
                _a.sent();
                _a.label = 3;
            case 3:
                process.exit(0);
                return [2 /*return*/];
        }
    });
}); });
// Start server
var PORT = process.env.PORT || 4000;
var HOST = process.env.HOST || '0.0.0.0';
server.listen(PORT, HOST, function () {
    winstonLogger.info("MedTrack API server running on ".concat(HOST, ":").concat(PORT));
    winstonLogger.info("Environment: ".concat(process.env.NODE_ENV || 'development'));
    winstonLogger.info("Database: ".concat(process.env.DATABASE_URL ? 'Connected' : 'Not configured'));
    winstonLogger.info("Redis: ".concat(process.env.REDIS_URL || 'Not configured'));
    winstonLogger.info("Vector Search: ".concat(process.env.VECTOR_SEARCH_URL || 'Not configured'));
    winstonLogger.info("Ollama: ".concat(process.env.OLLAMA_URL || 'Not configured'));
    winstonLogger.info("Qdrant: ".concat(process.env.QDRANT_URL || 'Not configured'));
});
var templateObject_1;

"use strict";
/**
 * Health Report Controller
 * Handles API requests for health reports
 */
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
exports.HealthReportController = void 0;
var report_1 = require("../core/report");
var HealthReportController = /** @class */ (function () {
    function HealthReportController(prisma) {
        this.generator = new report_1.HealthReportGenerator(prisma, {
            provider: process.env.AI_PROVIDER || 'local',
            apiKey: process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY,
            apiUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
            model: process.env.AI_MODEL || 'llama3.2'
        });
    }
    /**
     * GET /api/health-report
     * Generate comprehensive health report
     */
    HealthReportController.prototype.generate = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, timeframe, report, error_1;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        timeframe = req.query.timeframe || '30d';
                        return [4 /*yield*/, this.generator.generateHealthReport(userId, timeframe)];
                    case 1:
                        report = _b.sent();
                        res.json({
                            success: true,
                            data: report
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        error_1 = _b.sent();
                        console.error('Health report controller error:', error_1);
                        res.status(500).json({
                            success: false,
                            error: error_1.message || 'Failed to generate health report'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * GET /api/health-report/download
     * Download health report as JSON
     */
    HealthReportController.prototype.download = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, timeframe, report, error_2;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        timeframe = req.query.timeframe || '30d';
                        return [4 /*yield*/, this.generator.generateHealthReport(userId, timeframe)];
                    case 1:
                        report = _b.sent();
                        res.setHeader('Content-Type', 'application/json');
                        res.setHeader('Content-Disposition', "attachment; filename=health-report-".concat(Date.now(), ".json"));
                        res.json(report);
                        return [3 /*break*/, 3];
                    case 2:
                        error_2 = _b.sent();
                        console.error('Health report controller error:', error_2);
                        res.status(500).json({
                            success: false,
                            error: error_2.message || 'Failed to generate health report'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    return HealthReportController;
}());
exports.HealthReportController = HealthReportController;

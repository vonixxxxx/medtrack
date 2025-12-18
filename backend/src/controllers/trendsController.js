"use strict";
/**
 * Trends Controller
 * Handles API requests for metric trends
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
exports.TrendsController = void 0;
var trends_1 = require("../core/trends");
var TrendsController = /** @class */ (function () {
    function TrendsController(prisma) {
        this.engine = new trends_1.TrendsEngine(prisma);
    }
    /**
     * GET /api/metrics/trends
     * Get trends for all metrics
     */
    TrendsController.prototype.getAll = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, days, trendsMap, trends, _i, _a, _b, metricName, analysis, error_1;
            var _c;
            return __generator(this, function (_d) {
                switch (_d.label) {
                    case 0:
                        _d.trys.push([0, 2, , 3]);
                        userId = (_c = req.user) === null || _c === void 0 ? void 0 : _c.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        days = parseInt(req.query.days) || 30;
                        return [4 /*yield*/, this.engine.getAllMetricsTrends(userId, days)];
                    case 1:
                        trendsMap = _d.sent();
                        trends = {};
                        for (_i = 0, _a = trendsMap.entries(); _i < _a.length; _i++) {
                            _b = _a[_i], metricName = _b[0], analysis = _b[1];
                            trends[metricName] = analysis;
                        }
                        res.json({
                            success: true,
                            data: trends,
                            timeframe: "".concat(days, " days")
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        error_1 = _d.sent();
                        console.error('Trends controller error:', error_1);
                        res.status(500).json({
                            success: false,
                            error: error_1.message || 'Failed to fetch trends'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * GET /api/metrics/trends/:metricName
     * Get trends for a specific metric
     */
    TrendsController.prototype.getOne = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, metricName, days, analysis, error_2;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        metricName = req.params.metricName;
                        days = parseInt(req.query.days) || 30;
                        return [4 /*yield*/, this.engine.computeMetricTrends(metricName, userId, days)];
                    case 1:
                        analysis = _b.sent();
                        res.json({
                            success: true,
                            data: analysis
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        error_2 = _b.sent();
                        console.error('Trends controller error:', error_2);
                        res.status(500).json({
                            success: false,
                            error: error_2.message || 'Failed to fetch metric trends'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * GET /api/metrics/trends/:metricName/classification
     * Get trend classification
     */
    TrendsController.prototype.getClassification = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, metricName, days, analysis, error_3;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        metricName = req.params.metricName;
                        days = parseInt(req.query.days) || 30;
                        return [4 /*yield*/, this.engine.computeMetricTrends(metricName, userId, days)];
                    case 1:
                        analysis = _b.sent();
                        res.json({
                            success: true,
                            data: analysis.trendClassification
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        error_3 = _b.sent();
                        console.error('Trends controller error:', error_3);
                        res.status(500).json({
                            success: false,
                            error: error_3.message || 'Failed to get trend classification'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * GET /api/metrics/trends/:metricName/anomalies
     * Detect anomalies for a metric
     */
    TrendsController.prototype.getAnomalies = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, metricName, days, threshold, anomalies, error_4;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        metricName = req.params.metricName;
                        days = parseInt(req.query.days) || 30;
                        threshold = parseFloat(req.query.threshold) || 2.5;
                        return [4 /*yield*/, this.engine.detectMetricAnomalies(metricName, userId, days, threshold)];
                    case 1:
                        anomalies = _b.sent();
                        res.json({
                            success: true,
                            data: anomalies
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        error_4 = _b.sent();
                        console.error('Trends controller error:', error_4);
                        res.status(500).json({
                            success: false,
                            error: error_4.message || 'Failed to detect anomalies'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * GET /api/metrics/trends/:metricName/trajectory
     * Get metric trajectory
     */
    TrendsController.prototype.getTrajectory = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, metricName, days, trajectory, error_5;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        metricName = req.params.metricName;
                        days = parseInt(req.query.days) || 30;
                        return [4 /*yield*/, this.engine.getMetricTrajectory(metricName, userId, days)];
                    case 1:
                        trajectory = _b.sent();
                        res.json({
                            success: true,
                            data: trajectory
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        error_5 = _b.sent();
                        console.error('Trends controller error:', error_5);
                        res.status(500).json({
                            success: false,
                            error: error_5.message || 'Failed to get trajectory'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    return TrendsController;
}());
exports.TrendsController = TrendsController;

"use strict";
/**
 * Progress-over-Time Engine for Metrics
 *
 * Computes moving averages, trends, variability, and anomalies
 * Inspired by Daylio, Bearable, Levels, Tidepool
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
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.TrendsEngine = void 0;
var date_fns_1 = require("date-fns");
var utils_1 = require("./utils");
var TrendsEngine = /** @class */ (function () {
    function TrendsEngine(prisma) {
        this.prisma = prisma;
    }
    /**
     * Compute metric trends
     */
    TrendsEngine.prototype.computeMetricTrends = function (metricName_1, userId_1) {
        return __awaiter(this, arguments, void 0, function (metricName, userId, days) {
            var endDate, startDate, cycles, dataPoints, _i, cycles_1, cycle, _a, _b, log, value, sevenDayMA, fourteenDayMA, thirtyDayMA, values, mean, sortedValues, median, standardDeviation, trendClassification, coefficientOfVariation, variabilityScore, anomalies, current, projected, changeFromBaseline;
            var _c;
            if (days === void 0) { days = 30; }
            return __generator(this, function (_d) {
                switch (_d.label) {
                    case 0:
                        endDate = (0, date_fns_1.endOfDay)(new Date());
                        startDate = (0, date_fns_1.startOfDay)((0, date_fns_1.subDays)(endDate, days));
                        return [4 /*yield*/, this.prisma.medicationCycle.findMany({
                                where: {
                                    userId: userId
                                },
                                include: {
                                    metricLogs: {
                                        where: {
                                            kind: metricName.toUpperCase(),
                                            date: {
                                                gte: startDate,
                                                lte: endDate
                                            }
                                        },
                                        orderBy: {
                                            date: 'asc'
                                        }
                                    }
                                }
                            })];
                    case 1:
                        cycles = _d.sent();
                        dataPoints = [];
                        for (_i = 0, cycles_1 = cycles; _i < cycles_1.length; _i++) {
                            cycle = cycles_1[_i];
                            for (_a = 0, _b = cycle.metricLogs; _a < _b.length; _a++) {
                                log = _b[_a];
                                value = log.valueFloat || parseFloat(log.valueText || '0');
                                if (!isNaN(value)) {
                                    dataPoints.push({
                                        timestamp: log.date,
                                        value: value,
                                        unit: log.notes || undefined
                                    });
                                }
                            }
                        }
                        if (dataPoints.length === 0) {
                            throw new Error("No data points found for metric: ".concat(metricName));
                        }
                        sevenDayMA = (0, utils_1.calculateMovingAverage)(dataPoints, 7, endDate);
                        fourteenDayMA = (0, utils_1.calculateMovingAverage)(dataPoints, 14, endDate);
                        thirtyDayMA = (0, utils_1.calculateMovingAverage)(dataPoints, Math.min(30, dataPoints.length), endDate);
                        values = dataPoints.map(function (p) { return p.value; });
                        mean = values.reduce(function (a, b) { return a + b; }, 0) / values.length;
                        sortedValues = __spreadArray([], values, true).sort(function (a, b) { return a - b; });
                        median = sortedValues.length % 2 === 0
                            ? (sortedValues[sortedValues.length / 2 - 1] + sortedValues[sortedValues.length / 2]) / 2
                            : sortedValues[Math.floor(sortedValues.length / 2)];
                        standardDeviation = (0, utils_1.calculateStandardDeviation)(values);
                        trendClassification = this.getTrendClassification(dataPoints, mean, standardDeviation);
                        coefficientOfVariation = (0, utils_1.calculateCoefficientOfVariation)(mean, standardDeviation);
                        variabilityScore = {
                            standardDeviation: standardDeviation,
                            coefficientOfVariation: coefficientOfVariation,
                            score: Math.min(100, Math.max(0, 100 - (coefficientOfVariation * 100)))
                        };
                        anomalies = (0, utils_1.detectAnomalies)(dataPoints, mean, standardDeviation);
                        current = ((_c = dataPoints[dataPoints.length - 1]) === null || _c === void 0 ? void 0 : _c.value) || mean;
                        projected = (0, utils_1.calculateLinearProjection)(dataPoints, 7);
                        changeFromBaseline = current - mean;
                        return [2 /*return*/, {
                                metricName: metricName,
                                movingAverages: {
                                    sevenDay: sevenDayMA,
                                    fourteenDay: fourteenDayMA,
                                    thirtyDay: thirtyDayMA
                                },
                                trendClassification: trendClassification,
                                variabilityScore: variabilityScore,
                                anomalies: anomalies,
                                baseline: {
                                    mean: mean,
                                    median: median,
                                    standardDeviation: standardDeviation
                                },
                                trajectory: {
                                    current: current,
                                    projected: projected,
                                    changeFromBaseline: changeFromBaseline
                                }
                            }];
                }
            });
        });
    };
    /**
     * Get trend classification
     */
    TrendsEngine.prototype.getTrendClassification = function (dataPoints, baselineMean, baselineStdDev) {
        if (dataPoints.length < 2) {
            return {
                trend: 'stable',
                confidence: 0,
                changePercentage: 0,
                direction: 0
            };
        }
        var sortedData = __spreadArray([], dataPoints, true).sort(function (a, b) {
            return a.timestamp.getTime() - b.timestamp.getTime();
        });
        // Compare first third to last third
        var thirdSize = Math.floor(sortedData.length / 3);
        var firstThird = sortedData.slice(0, thirdSize);
        var lastThird = sortedData.slice(-thirdSize);
        var firstThirdAvg = firstThird.reduce(function (acc, p) { return acc + p.value; }, 0) / firstThird.length;
        var lastThirdAvg = lastThird.reduce(function (acc, p) { return acc + p.value; }, 0) / lastThird.length;
        var change = lastThirdAvg - firstThirdAvg;
        var changePercentage = (change / baselineMean) * 100;
        var direction = change > 0 ? 1 : change < 0 ? -1 : 0;
        // Calculate volatility
        var allValues = sortedData.map(function (p) { return p.value; });
        var volatility = (0, utils_1.calculateStandardDeviation)(allValues) / baselineMean;
        // Determine trend
        var trend = 'stable';
        var confidence = 0.5;
        if (volatility > 0.3) {
            trend = 'volatile';
            confidence = 0.7;
        }
        else if (Math.abs(changePercentage) > 10) {
            trend = changePercentage > 0 ? 'improving' : 'declining';
            confidence = Math.min(0.95, 0.5 + Math.abs(changePercentage) / 50);
        }
        else {
            trend = 'stable';
            confidence = 0.8;
        }
        return {
            trend: trend,
            confidence: confidence,
            changePercentage: changePercentage,
            direction: direction
        };
    };
    /**
     * Get metric trajectory
     */
    TrendsEngine.prototype.getMetricTrajectory = function (metricName_1, userId_1) {
        return __awaiter(this, arguments, void 0, function (metricName, userId, days) {
            var analysis;
            if (days === void 0) { days = 30; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.computeMetricTrends(metricName, userId, days)];
                    case 1:
                        analysis = _a.sent();
                        return [2 /*return*/, analysis.trajectory];
                }
            });
        });
    };
    /**
     * Detect metric anomalies
     */
    TrendsEngine.prototype.detectMetricAnomalies = function (metricName_1, userId_1) {
        return __awaiter(this, arguments, void 0, function (metricName, userId, days, threshold) {
            var analysis, endDate, startDate, cycles, dataPoints, _i, cycles_2, cycle, _a, _b, log, value;
            if (days === void 0) { days = 30; }
            if (threshold === void 0) { threshold = 2.5; }
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0: return [4 /*yield*/, this.computeMetricTrends(metricName, userId, days)];
                    case 1:
                        analysis = _c.sent();
                        endDate = (0, date_fns_1.endOfDay)(new Date());
                        startDate = (0, date_fns_1.startOfDay)((0, date_fns_1.subDays)(endDate, days));
                        return [4 /*yield*/, this.prisma.medicationCycle.findMany({
                                where: {
                                    userId: userId
                                },
                                include: {
                                    metricLogs: {
                                        where: {
                                            kind: metricName.toUpperCase(),
                                            date: {
                                                gte: startDate,
                                                lte: endDate
                                            }
                                        }
                                    }
                                }
                            })];
                    case 2:
                        cycles = _c.sent();
                        dataPoints = [];
                        for (_i = 0, cycles_2 = cycles; _i < cycles_2.length; _i++) {
                            cycle = cycles_2[_i];
                            for (_a = 0, _b = cycle.metricLogs; _a < _b.length; _a++) {
                                log = _b[_a];
                                value = log.valueFloat || parseFloat(log.valueText || '0');
                                if (!isNaN(value)) {
                                    dataPoints.push({
                                        timestamp: log.date,
                                        value: value
                                    });
                                }
                            }
                        }
                        return [2 /*return*/, (0, utils_1.detectAnomalies)(dataPoints, analysis.baseline.mean, analysis.baseline.standardDeviation, threshold)];
                }
            });
        });
    };
    /**
     * Get all metrics trends for a user
     */
    TrendsEngine.prototype.getAllMetricsTrends = function (userId_1) {
        return __awaiter(this, arguments, void 0, function (userId, days) {
            var cycles, metricTypes, _i, cycles_3, cycle, _a, _b, log, trendsMap, _c, metricTypes_1, metricType, analysis, error_1;
            if (days === void 0) { days = 30; }
            return __generator(this, function (_d) {
                switch (_d.label) {
                    case 0: return [4 /*yield*/, this.prisma.medicationCycle.findMany({
                            where: {
                                userId: userId
                            },
                            include: {
                                metricLogs: {
                                    distinct: ['kind'],
                                    select: {
                                        kind: true
                                    }
                                }
                            }
                        })];
                    case 1:
                        cycles = _d.sent();
                        metricTypes = new Set();
                        for (_i = 0, cycles_3 = cycles; _i < cycles_3.length; _i++) {
                            cycle = cycles_3[_i];
                            for (_a = 0, _b = cycle.metricLogs; _a < _b.length; _a++) {
                                log = _b[_a];
                                if (log.kind) {
                                    metricTypes.add(log.kind);
                                }
                            }
                        }
                        trendsMap = new Map();
                        _c = 0, metricTypes_1 = metricTypes;
                        _d.label = 2;
                    case 2:
                        if (!(_c < metricTypes_1.length)) return [3 /*break*/, 7];
                        metricType = metricTypes_1[_c];
                        _d.label = 3;
                    case 3:
                        _d.trys.push([3, 5, , 6]);
                        return [4 /*yield*/, this.computeMetricTrends(metricType, userId, days)];
                    case 4:
                        analysis = _d.sent();
                        trendsMap.set(metricType, analysis);
                        return [3 /*break*/, 6];
                    case 5:
                        error_1 = _d.sent();
                        // Skip metrics with no data
                        console.warn("No data for metric: ".concat(metricType));
                        return [3 /*break*/, 6];
                    case 6:
                        _c++;
                        return [3 /*break*/, 2];
                    case 7: return [2 /*return*/, trendsMap];
                }
            });
        });
    };
    return TrendsEngine;
}());
exports.TrendsEngine = TrendsEngine;

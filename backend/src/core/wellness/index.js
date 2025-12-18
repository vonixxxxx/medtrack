"use strict";
/**
 * Wellness Score Engine
 *
 * Composite score combining adherence, metrics, stability, and energy/sleep
 * Inspired by Oura/Whoop's readiness score
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
exports.WellnessEngine = void 0;
var date_fns_1 = require("date-fns");
var adherence_1 = require("../adherence");
var trends_1 = require("../trends");
var utils_1 = require("./utils");
var WellnessEngine = /** @class */ (function () {
    function WellnessEngine(prisma) {
        this.prisma = prisma;
        this.adherenceEngine = new adherence_1.AdherenceEngine(prisma);
        this.trendsEngine = new trends_1.TrendsEngine(prisma);
    }
    /**
     * Calculate overall wellness score
     */
    WellnessEngine.prototype.calculateWellnessScore = function (userId_1) {
        return __awaiter(this, arguments, void 0, function (userId, days) {
            var adherenceData, averageAdherence, adherenceScore, metricsTrends, metricScore, stabilityScore, energyOrSleepScore, weights, overallScore;
            if (days === void 0) { days = 30; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.adherenceEngine.getAllMedicationsAdherence(userId)];
                    case 1:
                        adherenceData = _a.sent();
                        averageAdherence = adherenceData.length > 0
                            ? adherenceData.reduce(function (sum, med) { return sum + med.adherencePercentage; }, 0) / adherenceData.length
                            : 100;
                        adherenceScore = averageAdherence;
                        return [4 /*yield*/, this.trendsEngine.getAllMetricsTrends(userId, days)];
                    case 2:
                        metricsTrends = _a.sent();
                        metricScore = this.calculateMetricsScore(metricsTrends);
                        stabilityScore = this.calculateStabilityScore(metricsTrends);
                        return [4 /*yield*/, this.getEnergyOrSleepScore(userId, days)];
                    case 3:
                        energyOrSleepScore = _a.sent();
                        weights = {
                            adherence: 0.3,
                            metrics: 0.4,
                            stability: 0.2,
                            energyOrSleep: 0.1
                        };
                        overallScore = (0, utils_1.calculateWeightedAverage)([
                            { value: adherenceScore, weight: weights.adherence },
                            { value: metricScore, weight: weights.metrics },
                            { value: stabilityScore, weight: weights.stability },
                            { value: energyOrSleepScore, weight: weights.energyOrSleep }
                        ]);
                        return [2 /*return*/, {
                                overallScore: Math.round(overallScore * 100) / 100,
                                breakdown: {
                                    adherenceScore: Math.round(adherenceScore * 100) / 100,
                                    metricScore: Math.round(metricScore * 100) / 100,
                                    stabilityScore: Math.round(stabilityScore * 100) / 100,
                                    energyOrSleepScore: Math.round(energyOrSleepScore * 100) / 100
                                },
                                weights: weights,
                                baselineAdjusted: true,
                                timestamp: new Date()
                            }];
                }
            });
        });
    };
    /**
     * Calculate metrics score from normalized metrics
     */
    WellnessEngine.prototype.calculateMetricsScore = function (metricsTrends) {
        if (metricsTrends.size === 0)
            return 50; // Default if no metrics
        var normalizedScores = [];
        for (var _i = 0, _a = metricsTrends.entries(); _i < _a.length; _i++) {
            var _b = _a[_i], metricName = _b[0], analysis = _b[1];
            // Skip energy/sleep metrics (handled separately)
            if (['ENERGY', 'SLEEP', 'SLEEP_HOURS', 'ENERGY_LEVEL'].includes(metricName.toUpperCase())) {
                continue;
            }
            var current = analysis.trajectory.current;
            var baseline = analysis.baseline.mean;
            var stdDev = analysis.baseline.standardDeviation;
            // Normalize based on trend direction
            // For metrics where higher is better (e.g., mood, energy)
            // For metrics where lower is better (e.g., pain, blood pressure), we'll need to invert
            var isHigherBetter = this.isHigherBetterMetric(metricName);
            var normalizedValue = void 0;
            if (stdDev === 0) {
                normalizedValue = current === baseline ? 50 : (isHigherBetter ? (current > baseline ? 75 : 25) : (current < baseline ? 75 : 25));
            }
            else {
                // Use z-score normalization, then convert to 0-100
                var zScore = (current - baseline) / stdDev;
                if (isHigherBetter) {
                    normalizedValue = (0, utils_1.normalizeToScore)(zScore, -3, 3);
                }
                else {
                    // Invert for lower-is-better metrics
                    normalizedValue = (0, utils_1.normalizeToScore)(-zScore, -3, 3);
                }
            }
            normalizedScores.push(normalizedValue);
        }
        return normalizedScores.length > 0
            ? normalizedScores.reduce(function (a, b) { return a + b; }, 0) / normalizedScores.length
            : 50;
    };
    /**
     * Determine if higher values are better for a metric
     */
    WellnessEngine.prototype.isHigherBetterMetric = function (metricName) {
        var higherIsBetter = [
            'MOOD', 'ENERGY', 'SLEEP_QUALITY', 'SLEEP_HOURS', 'ENERGY_LEVEL',
            'HAPPINESS', 'VITALITY', 'WELLBEING'
        ];
        var lowerIsBetter = [
            'PAIN', 'BLOOD_PRESSURE', 'HEART_RATE', 'STRESS', 'ANXIETY',
            'WEIGHT', 'BMI', 'BLOOD_SUGAR'
        ];
        var upperName = metricName.toUpperCase();
        if (higherIsBetter.some(function (m) { return upperName.includes(m); }))
            return true;
        if (lowerIsBetter.some(function (m) { return upperName.includes(m); }))
            return false;
        // Default: assume higher is better
        return true;
    };
    /**
     * Calculate stability score from variability
     */
    WellnessEngine.prototype.calculateStabilityScore = function (metricsTrends) {
        if (metricsTrends.size === 0)
            return 50;
        var variabilityScores = [];
        for (var _i = 0, _a = metricsTrends.values(); _i < _a.length; _i++) {
            var analysis = _a[_i];
            var cv = analysis.variabilityScore.coefficientOfVariation;
            var stability = (0, utils_1.calculateInverseVariabilityScore)(cv);
            variabilityScores.push(stability);
        }
        return variabilityScores.length > 0
            ? variabilityScores.reduce(function (a, b) { return a + b; }, 0) / variabilityScores.length
            : 50;
    };
    /**
     * Get energy or sleep score
     */
    WellnessEngine.prototype.getEnergyOrSleepScore = function (userId, days) {
        return __awaiter(this, void 0, void 0, function () {
            var endDate, startDate, cycles, energyOrSleepValues, _i, cycles_1, cycle, _a, _b, log, value, avgValue;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        endDate = (0, date_fns_1.endOfDay)(new Date());
                        startDate = (0, date_fns_1.subDays)(endDate, days);
                        return [4 /*yield*/, this.prisma.medicationCycle.findMany({
                                where: {
                                    userId: userId
                                },
                                include: {
                                    metricLogs: {
                                        where: {
                                            kind: {
                                                in: ['ENERGY', 'SLEEP', 'SLEEP_HOURS', 'ENERGY_LEVEL']
                                            },
                                            date: {
                                                gte: startDate,
                                                lte: endDate
                                            }
                                        },
                                        orderBy: {
                                            date: 'desc'
                                        },
                                        take: 30
                                    }
                                }
                            })];
                    case 1:
                        cycles = _c.sent();
                        energyOrSleepValues = [];
                        for (_i = 0, cycles_1 = cycles; _i < cycles_1.length; _i++) {
                            cycle = cycles_1[_i];
                            for (_a = 0, _b = cycle.metricLogs; _a < _b.length; _a++) {
                                log = _b[_a];
                                value = log.valueFloat || parseFloat(log.valueText || '0');
                                if (!isNaN(value)) {
                                    energyOrSleepValues.push(value);
                                }
                            }
                        }
                        if (energyOrSleepValues.length === 0) {
                            return [2 /*return*/, 50]; // Default if no energy/sleep data
                        }
                        avgValue = energyOrSleepValues.reduce(function (a, b) { return a + b; }, 0) / energyOrSleepValues.length;
                        // Normalize to 0-100 (assuming 0-10 scale for energy, or 0-12 for sleep hours)
                        return [2 /*return*/, (0, utils_1.normalizeToScore)(avgValue, 0, 10)];
                }
            });
        });
    };
    /**
     * Get wellness breakdown
     */
    WellnessEngine.prototype.getWellnessBreakdown = function (userId_1) {
        return __awaiter(this, arguments, void 0, function (userId, days) {
            var adherenceData, metricsTrends, averageAdherence, normalizedMetrics, _i, _a, _b, metricName, analysis, current, baseline, stdDev, isHigherBetter, normalizedValue, zScore, metricScore, variabilityScores, _c, _d, analysis, cv, stabilityScore, energyOrSleepData;
            if (days === void 0) { days = 30; }
            return __generator(this, function (_e) {
                switch (_e.label) {
                    case 0: return [4 /*yield*/, this.adherenceEngine.getAllMedicationsAdherence(userId)];
                    case 1:
                        adherenceData = _e.sent();
                        return [4 /*yield*/, this.trendsEngine.getAllMetricsTrends(userId, days)];
                    case 2:
                        metricsTrends = _e.sent();
                        averageAdherence = adherenceData.length > 0
                            ? adherenceData.reduce(function (sum, med) { return sum + med.adherencePercentage; }, 0) / adherenceData.length
                            : 100;
                        normalizedMetrics = [];
                        for (_i = 0, _a = metricsTrends.entries(); _i < _a.length; _i++) {
                            _b = _a[_i], metricName = _b[0], analysis = _b[1];
                            if (['ENERGY', 'SLEEP', 'SLEEP_HOURS', 'ENERGY_LEVEL'].includes(metricName.toUpperCase())) {
                                continue;
                            }
                            current = analysis.trajectory.current;
                            baseline = analysis.baseline.mean;
                            stdDev = analysis.baseline.standardDeviation;
                            isHigherBetter = this.isHigherBetterMetric(metricName);
                            normalizedValue = void 0;
                            if (stdDev === 0) {
                                normalizedValue = 50;
                            }
                            else {
                                zScore = (current - baseline) / stdDev;
                                normalizedValue = isHigherBetter
                                    ? (0, utils_1.normalizeToScore)(zScore, -3, 3)
                                    : (0, utils_1.normalizeToScore)(-zScore, -3, 3);
                            }
                            normalizedMetrics.push({
                                name: metricName,
                                normalizedValue: normalizedValue,
                                trend: analysis.trendClassification.trend
                            });
                        }
                        metricScore = normalizedMetrics.length > 0
                            ? normalizedMetrics.reduce(function (sum, m) { return sum + m.normalizedValue; }, 0) / normalizedMetrics.length
                            : 50;
                        variabilityScores = [];
                        for (_c = 0, _d = metricsTrends.values(); _c < _d.length; _c++) {
                            analysis = _d[_c];
                            cv = analysis.variabilityScore.coefficientOfVariation;
                            variabilityScores.push((0, utils_1.calculateInverseVariabilityScore)(cv));
                        }
                        stabilityScore = variabilityScores.length > 0
                            ? variabilityScores.reduce(function (a, b) { return a + b; }, 0) / variabilityScores.length
                            : 50;
                        return [4 /*yield*/, this.getEnergyOrSleepData(userId, days)];
                    case 3:
                        energyOrSleepData = _e.sent();
                        return [2 /*return*/, {
                                adherence: {
                                    score: averageAdherence,
                                    averageAdherence: averageAdherence,
                                    medicationsCount: adherenceData.length
                                },
                                metrics: {
                                    score: metricScore,
                                    normalizedMetrics: normalizedMetrics
                                },
                                stability: {
                                    score: stabilityScore,
                                    averageVariability: variabilityScores.length > 0
                                        ? variabilityScores.reduce(function (a, b) { return a + b; }, 0) / variabilityScores.length
                                        : 0,
                                    metricsCount: metricsTrends.size
                                },
                                energyOrSleep: energyOrSleepData
                            }];
                }
            });
        });
    };
    /**
     * Get energy or sleep data
     */
    WellnessEngine.prototype.getEnergyOrSleepData = function (userId, days) {
        return __awaiter(this, void 0, void 0, function () {
            var endDate, startDate, cycles, _i, cycles_2, cycle, _a, _b, log, value;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        endDate = (0, date_fns_1.endOfDay)(new Date());
                        startDate = (0, date_fns_1.subDays)(endDate, days);
                        return [4 /*yield*/, this.prisma.medicationCycle.findMany({
                                where: {
                                    userId: userId
                                },
                                include: {
                                    metricLogs: {
                                        where: {
                                            kind: {
                                                in: ['ENERGY', 'SLEEP', 'SLEEP_HOURS', 'ENERGY_LEVEL']
                                            },
                                            date: {
                                                gte: startDate,
                                                lte: endDate
                                            }
                                        },
                                        orderBy: {
                                            date: 'desc'
                                        },
                                        take: 1
                                    }
                                }
                            })];
                    case 1:
                        cycles = _c.sent();
                        for (_i = 0, cycles_2 = cycles; _i < cycles_2.length; _i++) {
                            cycle = cycles_2[_i];
                            for (_a = 0, _b = cycle.metricLogs; _a < _b.length; _a++) {
                                log = _b[_a];
                                value = log.valueFloat || parseFloat(log.valueText || '0');
                                if (!isNaN(value)) {
                                    return [2 /*return*/, {
                                            score: (0, utils_1.normalizeToScore)(value, 0, 10),
                                            metricName: log.kind,
                                            value: value,
                                            available: true
                                        }];
                                }
                            }
                        }
                        return [2 /*return*/, {
                                score: 50,
                                metricName: 'NONE',
                                value: 0,
                                available: false
                            }];
                }
            });
        });
    };
    /**
     * Compute baseline-adjusted metrics
     */
    WellnessEngine.prototype.computeBaselineAdjustedMetrics = function (userId_1) {
        return __awaiter(this, arguments, void 0, function (userId, days) {
            var metricsTrends, adjustedMetrics, _i, _a, _b, metricName, analysis, current, baseline, stdDev, deviation, isHigherBetter, normalizedScore, zScore, trend;
            if (days === void 0) { days = 30; }
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0: return [4 /*yield*/, this.trendsEngine.getAllMetricsTrends(userId, days)];
                    case 1:
                        metricsTrends = _c.sent();
                        adjustedMetrics = [];
                        for (_i = 0, _a = metricsTrends.entries(); _i < _a.length; _i++) {
                            _b = _a[_i], metricName = _b[0], analysis = _b[1];
                            current = analysis.trajectory.current;
                            baseline = analysis.baseline.mean;
                            stdDev = analysis.baseline.standardDeviation;
                            deviation = (0, utils_1.calculateBaselineDeviation)(current, baseline, stdDev);
                            isHigherBetter = this.isHigherBetterMetric(metricName);
                            normalizedScore = void 0;
                            if (stdDev === 0) {
                                normalizedScore = current === baseline ? 50 : (isHigherBetter ? (current > baseline ? 75 : 25) : (current < baseline ? 75 : 25));
                            }
                            else {
                                zScore = deviation;
                                normalizedScore = isHigherBetter
                                    ? (0, utils_1.normalizeToScore)(zScore, -3, 3)
                                    : (0, utils_1.normalizeToScore)(-zScore, -3, 3);
                            }
                            trend = void 0;
                            if (Math.abs(deviation) < 0.5) {
                                trend = 'at_baseline';
                            }
                            else if (deviation > 0) {
                                trend = isHigherBetter ? 'above_baseline' : 'below_baseline';
                            }
                            else {
                                trend = isHigherBetter ? 'below_baseline' : 'above_baseline';
                            }
                            adjustedMetrics.push({
                                metricName: metricName,
                                currentValue: current,
                                baselineValue: baseline,
                                deviation: deviation,
                                normalizedScore: normalizedScore,
                                trend: trend
                            });
                        }
                        return [2 /*return*/, adjustedMetrics];
                }
            });
        });
    };
    return WellnessEngine;
}());
exports.WellnessEngine = WellnessEngine;

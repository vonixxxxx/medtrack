"use strict";
/**
 * Health Report Generator
 *
 * Combines adherence, trends, wellness, and summarizer into comprehensive reports
 * Inspired by MediLog / Levels style summarization
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
exports.HealthReportGenerator = void 0;
var adherence_1 = require("../adherence");
var trends_1 = require("../trends");
var wellness_1 = require("../wellness");
var summarizer_1 = require("../summarizer");
var HealthReportGenerator = /** @class */ (function () {
    function HealthReportGenerator(prisma, summarizerConfig) {
        this.prisma = prisma;
        this.adherenceEngine = new adherence_1.AdherenceEngine(prisma);
        this.trendsEngine = new trends_1.TrendsEngine(prisma);
        this.wellnessEngine = new wellness_1.WellnessEngine(prisma);
        this.summarizer = new summarizer_1.SummarizerService(summarizerConfig);
    }
    /**
     * Generate comprehensive health report
     */
    HealthReportGenerator.prototype.generateHealthReport = function (userId_1) {
        return __awaiter(this, arguments, void 0, function (userId, timeframe) {
            var days, _a, adherenceData, metricsTrends, wellnessScore, wellnessBreakdown, overallAdherence, adherencePatterns, adherenceSummary, streaks, _loop_1, _i, streaks_1, streak, metricTrendSummaries, allAnomalies, _b, _c, _d, metricName, analysis, _e, _f, anomaly, correlationCandidates, narrativeSummary;
            var _this = this;
            if (timeframe === void 0) { timeframe = '30d'; }
            return __generator(this, function (_g) {
                switch (_g.label) {
                    case 0:
                        days = this.parseTimeframe(timeframe);
                        return [4 /*yield*/, Promise.all([
                                this.adherenceEngine.getAllMedicationsAdherence(userId, { startDate: undefined, endDate: undefined }),
                                this.trendsEngine.getAllMetricsTrends(userId, days),
                                this.wellnessEngine.calculateWellnessScore(userId, days),
                                this.wellnessEngine.getWellnessBreakdown(userId, days)
                            ])];
                    case 1:
                        _a = _g.sent(), adherenceData = _a[0], metricsTrends = _a[1], wellnessScore = _a[2], wellnessBreakdown = _a[3];
                        overallAdherence = adherenceData.length > 0
                            ? adherenceData.reduce(function (sum, med) { return sum + med.adherencePercentage; }, 0) / adherenceData.length
                            : 100;
                        return [4 /*yield*/, Promise.all(adherenceData.map(function (med) {
                                return _this.adherenceEngine.analyzeAdherencePatterns(med.medicationId, userId, days);
                            }))];
                    case 2:
                        adherencePatterns = _g.sent();
                        adherenceSummary = {
                            overallAdherence: overallAdherence,
                            medications: adherenceData.map(function (med, idx) { return ({
                                medicationId: med.medicationId,
                                name: med.medicationId, // Would need to fetch actual name from cycle
                                adherence: med.adherencePercentage,
                                pattern: adherencePatterns[idx].pattern,
                                streak: 0 // Will be populated below
                            }); }),
                            pattern: this.determineOverallPattern(adherencePatterns.map(function (p) { return p.pattern; }))
                        };
                        return [4 /*yield*/, Promise.all(adherenceData.map(function (med) { return __awaiter(_this, void 0, void 0, function () {
                                var streakData;
                                return __generator(this, function (_a) {
                                    switch (_a.label) {
                                        case 0: return [4 /*yield*/, this.adherenceEngine.getMedicationStreaks(med.medicationId, userId, days)];
                                        case 1:
                                            streakData = _a.sent();
                                            return [2 /*return*/, {
                                                    medicationId: med.medicationId,
                                                    currentStreak: streakData.currentStreak,
                                                    longestStreak: streakData.longestStreak
                                                }];
                                    }
                                });
                            }); }))];
                    case 3:
                        streaks = _g.sent();
                        _loop_1 = function (streak) {
                            var med = adherenceSummary.medications.find(function (m) { return m.medicationId === streak.medicationId; });
                            if (med) {
                                med.streak = streak.currentStreak;
                            }
                        };
                        // Update adherence summary with streaks
                        for (_i = 0, streaks_1 = streaks; _i < streaks_1.length; _i++) {
                            streak = streaks_1[_i];
                            _loop_1(streak);
                        }
                        metricTrendSummaries = Array.from(metricsTrends.entries()).map(function (_a) {
                            var metricName = _a[0], analysis = _a[1];
                            return ({
                                metricName: metricName,
                                trend: analysis.trendClassification.trend,
                                currentValue: analysis.trajectory.current,
                                changePercentage: analysis.trendClassification.changePercentage,
                                classification: "".concat(analysis.trendClassification.trend, " (confidence: ").concat((analysis.trendClassification.confidence * 100).toFixed(0), "%)")
                            });
                        });
                        allAnomalies = [];
                        for (_b = 0, _c = metricsTrends.entries(); _b < _c.length; _b++) {
                            _d = _c[_b], metricName = _d[0], analysis = _d[1];
                            for (_e = 0, _f = analysis.anomalies; _e < _f.length; _e++) {
                                anomaly = _f[_e];
                                allAnomalies.push({
                                    metricName: metricName,
                                    timestamp: anomaly.timestamp,
                                    value: anomaly.value,
                                    severity: anomaly.severity
                                });
                            }
                        }
                        correlationCandidates = this.calculateCorrelations(metricsTrends);
                        return [4 /*yield*/, this.summarizer.generateSummary({
                                adherenceSummary: adherenceSummary,
                                metricTrends: metricTrendSummaries,
                                anomalies: allAnomalies,
                                wellnessScore: {
                                    overallScore: wellnessScore.overallScore,
                                    breakdown: wellnessScore.breakdown
                                },
                                streaks: streaks,
                                correlationCandidates: correlationCandidates,
                                timeframe: timeframe
                            })];
                    case 4:
                        narrativeSummary = _g.sent();
                        return [2 /*return*/, {
                                timeframe: timeframe,
                                wellnessScore: {
                                    overallScore: wellnessScore.overallScore,
                                    breakdown: wellnessScore.breakdown
                                },
                                adherenceSummary: adherenceSummary,
                                metricTrendSummaries: metricTrendSummaries,
                                anomalies: allAnomalies,
                                narrativeSummary: narrativeSummary,
                                recommendations: narrativeSummary.recommendations,
                                generatedAt: new Date()
                            }];
                }
            });
        });
    };
    /**
     * Parse timeframe string to days
     */
    HealthReportGenerator.prototype.parseTimeframe = function (timeframe) {
        var match = timeframe.match(/(\d+)([dwmy])/);
        if (!match)
            return 30;
        var value = parseInt(match[1]);
        var unit = match[2];
        switch (unit) {
            case 'd': return value;
            case 'w': return value * 7;
            case 'm': return value * 30;
            case 'y': return value * 365;
            default: return 30;
        }
    };
    /**
     * Determine overall pattern from individual patterns
     */
    HealthReportGenerator.prototype.determineOverallPattern = function (patterns) {
        if (patterns.length === 0)
            return 'stable';
        var counts = {
            improving: 0,
            declining: 0,
            stable: 0,
            volatile: 0
        };
        for (var _i = 0, patterns_1 = patterns; _i < patterns_1.length; _i++) {
            var pattern = patterns_1[_i];
            counts[pattern]++;
        }
        // If volatile is most common, return volatile
        if (counts.volatile > patterns.length / 2)
            return 'volatile';
        // Otherwise, return the most common pattern
        var maxCount = Math.max(counts.improving, counts.declining, counts.stable);
        if (maxCount === counts.improving)
            return 'improving';
        if (maxCount === counts.declining)
            return 'declining';
        return 'stable';
    };
    /**
     * Calculate simple correlations between metrics
     */
    HealthReportGenerator.prototype.calculateCorrelations = function (metricsTrends) {
        var correlations = [];
        var metrics = Array.from(metricsTrends.entries());
        // Simple correlation calculation (Pearson correlation coefficient)
        for (var i = 0; i < metrics.length; i++) {
            for (var j = i + 1; j < metrics.length; j++) {
                var _a = metrics[i], name1 = _a[0], analysis1 = _a[1];
                var _b = metrics[j], name2 = _b[0], analysis2 = _b[1];
                // Use moving averages for correlation
                var values1 = analysis1.movingAverages.sevenDay.map(function (ma) { return ma.value; });
                var values2 = analysis2.movingAverages.sevenDay.map(function (ma) { return ma.value; });
                if (values1.length === values2.length && values1.length > 1) {
                    var correlation = this.calculatePearsonCorrelation(values1, values2);
                    // Only include moderate to strong correlations
                    if (Math.abs(correlation) > 0.3) {
                        correlations.push({
                            metric1: name1,
                            metric2: name2,
                            correlation: correlation
                        });
                    }
                }
            }
        }
        return correlations.sort(function (a, b) { return Math.abs(b.correlation) - Math.abs(a.correlation); });
    };
    /**
     * Calculate Pearson correlation coefficient
     */
    HealthReportGenerator.prototype.calculatePearsonCorrelation = function (x, y) {
        if (x.length !== y.length || x.length === 0)
            return 0;
        var n = x.length;
        var sumX = x.reduce(function (a, b) { return a + b; }, 0);
        var sumY = y.reduce(function (a, b) { return a + b; }, 0);
        var sumXY = x.reduce(function (acc, val, i) { return acc + val * y[i]; }, 0);
        var sumXX = x.reduce(function (acc, val) { return acc + val * val; }, 0);
        var sumYY = y.reduce(function (acc, val) { return acc + val * val; }, 0);
        var numerator = n * sumXY - sumX * sumY;
        var denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
        if (denominator === 0)
            return 0;
        return numerator / denominator;
    };
    return HealthReportGenerator;
}());
exports.HealthReportGenerator = HealthReportGenerator;

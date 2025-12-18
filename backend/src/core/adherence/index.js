"use strict";
/**
 * Medication Adherence Engine
 *
 * Computes adherence percentages, streaks, and patterns
 * Inspired by MyTherapy/Medisafe adherence models
 */
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
exports.AdherenceEngine = void 0;
var date_fns_1 = require("date-fns");
var utils_1 = require("./utils");
var AdherenceEngine = /** @class */ (function () {
    function AdherenceEngine(prisma) {
        this.prisma = prisma;
    }
    /**
     * Get daily adherence for a medication
     */
    AdherenceEngine.prototype.getDailyAdherence = function (medicationId, userId, date) {
        return __awaiter(this, void 0, void 0, function () {
            var targetDate, endDate, cycle, expectedDoses, actualDoses, adherencePercentage;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        targetDate = date ? (0, date_fns_1.startOfDay)(date) : (0, date_fns_1.startOfDay)(new Date());
                        endDate = (0, date_fns_1.endOfDay)(targetDate);
                        return [4 /*yield*/, this.prisma.medicationCycle.findFirst({
                                where: {
                                    id: medicationId,
                                    userId: userId
                                },
                                include: {
                                    doseLogs: {
                                        where: {
                                            date: {
                                                gte: targetDate,
                                                lte: endDate
                                            }
                                        }
                                    }
                                }
                            })];
                    case 1:
                        cycle = _a.sent();
                        if (!cycle) {
                            throw new Error('Medication cycle not found');
                        }
                        expectedDoses = cycle.dosesPerDay;
                        actualDoses = cycle.doseLogs.filter(function (log) { return log.taken; }).length;
                        adherencePercentage = (0, utils_1.calculateAdherence)(actualDoses, expectedDoses);
                        return [2 /*return*/, {
                                medicationId: medicationId,
                                expectedDosesPerDay: expectedDoses,
                                actualDosesTaken: cycle.doseLogs.map(function (log) { return ({
                                    timestamp: log.date,
                                    taken: log.taken
                                }); }),
                                adherencePercentage: adherencePercentage,
                                streakCount: 0, // Will be calculated separately
                                weeklyAdherenceHistory: []
                            }];
                }
            });
        });
    };
    /**
     * Get weekly adherence for a medication
     */
    AdherenceEngine.prototype.getWeeklyAdherence = function (medicationId_1, userId_1) {
        return __awaiter(this, arguments, void 0, function (medicationId, userId, weeks) {
            var endDate, startDate, cycle, weeklyHistory, dosesByDay, weekOffset, weekStart, weekEnd, dosesTaken, dosesExpected, datesInWeek, _i, datesInWeek_1, date, dayKey, dayData, weekAdherence, allDoses, totalTaken, totalExpected, adherencePercentage;
            if (weeks === void 0) { weeks = 4; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        endDate = (0, date_fns_1.endOfDay)(new Date());
                        startDate = (0, date_fns_1.subDays)(endDate, weeks * 7);
                        return [4 /*yield*/, this.prisma.medicationCycle.findFirst({
                                where: {
                                    id: medicationId,
                                    userId: userId
                                },
                                include: {
                                    doseLogs: {
                                        where: {
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
                        cycle = _a.sent();
                        if (!cycle) {
                            throw new Error('Medication cycle not found');
                        }
                        weeklyHistory = [];
                        dosesByDay = (0, utils_1.groupDosesByDay)(cycle.doseLogs.map(function (log) { return ({
                            timestamp: log.date,
                            taken: log.taken
                        }); }));
                        // Calculate weekly adherence
                        for (weekOffset = 0; weekOffset < weeks; weekOffset++) {
                            weekStart = (0, date_fns_1.startOfWeek)((0, date_fns_1.subDays)(endDate, weekOffset * 7), { weekStartsOn: 1 });
                            weekEnd = (0, date_fns_1.endOfWeek)(weekStart, { weekStartsOn: 1 });
                            dosesTaken = 0;
                            dosesExpected = 0;
                            datesInWeek = (0, utils_1.getDatesInRange)(weekStart, weekEnd);
                            for (_i = 0, datesInWeek_1 = datesInWeek; _i < datesInWeek_1.length; _i++) {
                                date = datesInWeek_1[_i];
                                dayKey = date.toISOString();
                                dayData = dosesByDay.get(dayKey);
                                if (dayData) {
                                    dosesTaken += dayData.taken;
                                    dosesExpected += dayData.expected;
                                }
                                else {
                                    // If no log exists, assume expected doses per day
                                    dosesExpected += cycle.dosesPerDay;
                                }
                            }
                            weekAdherence = (0, utils_1.calculateAdherence)(dosesTaken, dosesExpected);
                            weeklyHistory.push({
                                weekStart: weekStart,
                                weekEnd: weekEnd,
                                adherencePercentage: weekAdherence,
                                dosesTaken: dosesTaken,
                                dosesExpected: dosesExpected
                            });
                        }
                        allDoses = cycle.doseLogs;
                        totalTaken = allDoses.filter(function (log) { return log.taken; }).length;
                        totalExpected = allDoses.length || ((0, date_fns_1.differenceInDays)(endDate, startDate) + 1) * cycle.dosesPerDay;
                        adherencePercentage = (0, utils_1.calculateAdherence)(totalTaken, totalExpected);
                        return [2 /*return*/, {
                                medicationId: medicationId,
                                expectedDosesPerDay: cycle.dosesPerDay,
                                actualDosesTaken: allDoses.map(function (log) { return ({
                                    timestamp: log.date,
                                    taken: log.taken
                                }); }),
                                adherencePercentage: adherencePercentage,
                                streakCount: 0, // Will be calculated separately
                                weeklyAdherenceHistory: weeklyHistory.reverse() // Most recent first
                            }];
                }
            });
        });
    };
    /**
     * Get medication streaks
     */
    AdherenceEngine.prototype.getMedicationStreaks = function (medicationId_1, userId_1) {
        return __awaiter(this, arguments, void 0, function (medicationId, userId, days) {
            var endDate, startDate, cycle, dosesByDay, currentStreak, longestStreak, tempStreak, streakStartDate, lastMissedDate, consecutiveMissedDays, maxConsecutiveMissed, datesInRange, _i, datesInRange_1, date, dayKey, dayData, expectedDoses, takenDoses, isFullyAdherent;
            if (days === void 0) { days = 30; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        endDate = (0, date_fns_1.endOfDay)(new Date());
                        startDate = (0, date_fns_1.subDays)(endDate, days);
                        return [4 /*yield*/, this.prisma.medicationCycle.findFirst({
                                where: {
                                    id: medicationId,
                                    userId: userId
                                },
                                include: {
                                    doseLogs: {
                                        where: {
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
                        cycle = _a.sent();
                        if (!cycle) {
                            throw new Error('Medication cycle not found');
                        }
                        dosesByDay = (0, utils_1.groupDosesByDay)(cycle.doseLogs.map(function (log) { return ({
                            timestamp: log.date,
                            taken: log.taken
                        }); }));
                        currentStreak = 0;
                        longestStreak = 0;
                        tempStreak = 0;
                        streakStartDate = null;
                        lastMissedDate = null;
                        consecutiveMissedDays = 0;
                        maxConsecutiveMissed = 0;
                        datesInRange = (0, utils_1.getDatesInRange)(startDate, endDate);
                        for (_i = 0, datesInRange_1 = datesInRange; _i < datesInRange_1.length; _i++) {
                            date = datesInRange_1[_i];
                            dayKey = date.toISOString();
                            dayData = dosesByDay.get(dayKey);
                            expectedDoses = cycle.dosesPerDay;
                            takenDoses = (dayData === null || dayData === void 0 ? void 0 : dayData.taken) || 0;
                            isFullyAdherent = (0, utils_1.isDayFullyAdherent)(takenDoses, expectedDoses);
                            if (isFullyAdherent) {
                                tempStreak++;
                                consecutiveMissedDays = 0;
                                if (tempStreak > longestStreak) {
                                    longestStreak = tempStreak;
                                }
                            }
                            else {
                                if (tempStreak > currentStreak) {
                                    currentStreak = tempStreak;
                                    streakStartDate = (0, date_fns_1.subDays)(date, tempStreak);
                                }
                                tempStreak = 0;
                                consecutiveMissedDays++;
                                if (consecutiveMissedDays > maxConsecutiveMissed) {
                                    maxConsecutiveMissed = consecutiveMissedDays;
                                    lastMissedDate = date;
                                }
                            }
                        }
                        // Update current streak if still active
                        if (tempStreak > currentStreak) {
                            currentStreak = tempStreak;
                            streakStartDate = (0, date_fns_1.subDays)(endDate, tempStreak);
                        }
                        return [2 /*return*/, {
                                medicationId: medicationId,
                                currentStreak: currentStreak,
                                longestStreak: longestStreak,
                                streakStartDate: streakStartDate,
                                lastMissedDate: lastMissedDate,
                                consecutiveMissedDays: maxConsecutiveMissed
                            }];
                }
            });
        });
    };
    /**
     * Analyze adherence patterns
     */
    AdherenceEngine.prototype.analyzeAdherencePatterns = function (medicationId_1, userId_1) {
        return __awaiter(this, arguments, void 0, function (medicationId, userId, days) {
            var adherenceData, weeklyAdherence, averageAdherence, recentAdherence, midPoint, firstHalf, secondHalf, firstHalfAvg, secondHalfAvg, trend, variance, volatility, pattern;
            if (days === void 0) { days = 30; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.getWeeklyAdherence(medicationId, userId, Math.ceil(days / 7))];
                    case 1:
                        adherenceData = _a.sent();
                        if (adherenceData.weeklyAdherenceHistory.length < 2) {
                            return [2 /*return*/, {
                                    medicationId: medicationId,
                                    pattern: 'stable',
                                    trend: 0,
                                    volatility: 0,
                                    averageAdherence: adherenceData.adherencePercentage,
                                    recentAdherence: adherenceData.adherencePercentage
                                }];
                        }
                        weeklyAdherence = adherenceData.weeklyAdherenceHistory.map(function (w) { return w.adherencePercentage; });
                        averageAdherence = weeklyAdherence.reduce(function (a, b) { return a + b; }, 0) / weeklyAdherence.length;
                        recentAdherence = weeklyAdherence.slice(0, 2).reduce(function (a, b) { return a + b; }, 0) / Math.min(2, weeklyAdherence.length);
                        midPoint = Math.floor(weeklyAdherence.length / 2);
                        firstHalf = weeklyAdherence.slice(midPoint);
                        secondHalf = weeklyAdherence.slice(0, midPoint);
                        firstHalfAvg = firstHalf.reduce(function (a, b) { return a + b; }, 0) / firstHalf.length;
                        secondHalfAvg = secondHalf.reduce(function (a, b) { return a + b; }, 0) / secondHalf.length;
                        trend = (secondHalfAvg - firstHalfAvg) / 100;
                        variance = weeklyAdherence.reduce(function (acc, val) {
                            return acc + Math.pow(val - averageAdherence, 2);
                        }, 0) / weeklyAdherence.length;
                        volatility = Math.sqrt(variance) / 100;
                        pattern = 'stable';
                        if (volatility > 0.2) {
                            pattern = 'volatile';
                        }
                        else if (trend > 0.1) {
                            pattern = 'improving';
                        }
                        else if (trend < -0.1) {
                            pattern = 'declining';
                        }
                        return [2 /*return*/, {
                                medicationId: medicationId,
                                pattern: pattern,
                                trend: trend,
                                volatility: volatility,
                                averageAdherence: averageAdherence,
                                recentAdherence: recentAdherence
                            }];
                }
            });
        });
    };
    /**
     * Get adherence for all medications
     */
    AdherenceEngine.prototype.getAllMedicationsAdherence = function (userId_1) {
        return __awaiter(this, arguments, void 0, function (userId, options) {
            var cycles, adherencePromises;
            var _this = this;
            if (options === void 0) { options = {}; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.prisma.medicationCycle.findMany({
                            where: __assign({ userId: userId }, (options.startDate && options.endDate ? {
                                startDate: { lte: options.endDate },
                                OR: [
                                    { endDate: null },
                                    { endDate: { gte: options.startDate } }
                                ]
                            } : {}))
                        })];
                    case 1:
                        cycles = _a.sent();
                        adherencePromises = cycles.map(function (cycle) {
                            return _this.getWeeklyAdherence(cycle.id, userId, 4);
                        });
                        return [2 /*return*/, Promise.all(adherencePromises)];
                }
            });
        });
    };
    return AdherenceEngine;
}());
exports.AdherenceEngine = AdherenceEngine;

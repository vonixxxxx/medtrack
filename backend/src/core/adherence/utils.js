"use strict";
/**
 * Utility functions for adherence calculations
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.calculateAdherence = calculateAdherence;
exports.getDatesInRange = getDatesInRange;
exports.groupDosesByDay = groupDosesByDay;
exports.calculateMissedDosePenalty = calculateMissedDosePenalty;
exports.isDayFullyAdherent = isDayFullyAdherent;
var date_fns_1 = require("date-fns");
/**
 * Calculate adherence percentage
 */
function calculateAdherence(dosesTaken, dosesExpected, missedDosePenalty) {
    if (missedDosePenalty === void 0) { missedDosePenalty = 0; }
    if (dosesExpected === 0)
        return 100;
    var baseAdherence = (dosesTaken / dosesExpected) * 100;
    var penalty = missedDosePenalty * 100;
    return Math.max(0, Math.min(100, baseAdherence - penalty));
}
/**
 * Get all dates in a range
 */
function getDatesInRange(startDate, endDate) {
    var dates = [];
    var currentDate = (0, date_fns_1.startOfDay)(startDate);
    var end = (0, date_fns_1.startOfDay)(endDate);
    while (currentDate <= end) {
        dates.push(new Date(currentDate));
        currentDate = (0, date_fns_1.addDays)(currentDate, 1);
    }
    return dates;
}
/**
 * Group doses by day
 */
function groupDosesByDay(doses) {
    var grouped = new Map();
    for (var _i = 0, doses_1 = doses; _i < doses_1.length; _i++) {
        var dose = doses_1[_i];
        var dayKey = (0, date_fns_1.startOfDay)(dose.timestamp).toISOString();
        var existing = grouped.get(dayKey) || { taken: 0, expected: 0 };
        existing.expected += 1;
        if (dose.taken) {
            existing.taken += 1;
        }
        grouped.set(dayKey, existing);
    }
    return grouped;
}
/**
 * Calculate missed dose penalty
 */
function calculateMissedDosePenalty(missedDays, totalDays, penaltyWeight) {
    if (penaltyWeight === void 0) { penaltyWeight = 0.1; }
    if (totalDays === 0)
        return 0;
    return (missedDays / totalDays) * penaltyWeight;
}
/**
 * Check if a day is fully adherent (all expected doses taken)
 */
function isDayFullyAdherent(dosesTaken, dosesExpected) {
    return dosesTaken >= dosesExpected;
}

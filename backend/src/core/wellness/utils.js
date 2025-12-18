"use strict";
/**
 * Utility functions for wellness score calculations
 */
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
exports.normalizeToScore = normalizeToScore;
exports.calculateInverseVariabilityScore = calculateInverseVariabilityScore;
exports.calculateWeightedAverage = calculateWeightedAverage;
exports.calculateBaseline = calculateBaseline;
exports.calculateBaselineDeviation = calculateBaselineDeviation;
/**
 * Normalize a value to 0-100 scale
 */
function normalizeToScore(value, min, max) {
    if (max === min)
        return 50;
    return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
}
/**
 * Calculate inverse variability score (higher variability = lower score)
 */
function calculateInverseVariabilityScore(coefficientOfVariation, maxCV) {
    if (maxCV === void 0) { maxCV = 1.0; }
    // Normalize CV to 0-1, then invert
    var normalizedCV = Math.min(1, coefficientOfVariation / maxCV);
    return (1 - normalizedCV) * 100;
}
/**
 * Calculate weighted average
 */
function calculateWeightedAverage(values) {
    var totalWeight = values.reduce(function (sum, item) { return sum + item.weight; }, 0);
    if (totalWeight === 0)
        return 0;
    var weightedSum = values.reduce(function (sum, item) { return sum + (item.value * item.weight); }, 0);
    return weightedSum / totalWeight;
}
/**
 * Get baseline value (median of historical data)
 */
function calculateBaseline(values) {
    if (values.length === 0)
        return 0;
    var sorted = __spreadArray([], values, true).sort(function (a, b) { return a - b; });
    var mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];
}
/**
 * Calculate deviation from baseline in standard deviations
 */
function calculateBaselineDeviation(currentValue, baseline, standardDeviation) {
    if (standardDeviation === 0)
        return 0;
    return (currentValue - baseline) / standardDeviation;
}

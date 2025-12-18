"use strict";
/**
 * Utility functions for trend calculations
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
exports.calculateMovingAverage = calculateMovingAverage;
exports.calculateStandardDeviation = calculateStandardDeviation;
exports.calculateCoefficientOfVariation = calculateCoefficientOfVariation;
exports.detectAnomalies = detectAnomalies;
exports.calculateLinearProjection = calculateLinearProjection;
exports.normalizeMetricValue = normalizeMetricValue;
/**
 * Calculate moving average for a period
 */
function calculateMovingAverage(dataPoints, period, endDate) {
    if (endDate === void 0) { endDate = new Date(); }
    var averages = [];
    var sortedData = __spreadArray([], dataPoints, true).sort(function (a, b) {
        return a.timestamp.getTime() - b.timestamp.getTime();
    });
    for (var i = period - 1; i < sortedData.length; i++) {
        var window_1 = sortedData.slice(i - period + 1, i + 1);
        var sum = window_1.reduce(function (acc, point) { return acc + point.value; }, 0);
        var average = sum / window_1.length;
        averages.push({
            period: period,
            value: average,
            date: window_1[window_1.length - 1].timestamp
        });
    }
    return averages;
}
/**
 * Calculate standard deviation
 */
function calculateStandardDeviation(values) {
    if (values.length === 0)
        return 0;
    var mean = values.reduce(function (a, b) { return a + b; }, 0) / values.length;
    var variance = values.reduce(function (acc, val) { return acc + Math.pow(val - mean, 2); }, 0) / values.length;
    return Math.sqrt(variance);
}
/**
 * Calculate coefficient of variation
 */
function calculateCoefficientOfVariation(mean, standardDeviation) {
    if (mean === 0)
        return 0;
    return standardDeviation / mean;
}
/**
 * Detect anomalies using z-score method
 */
function detectAnomalies(dataPoints, mean, standardDeviation, threshold // Number of standard deviations
) {
    if (threshold === void 0) { threshold = 2.5; }
    var anomalies = [];
    for (var _i = 0, dataPoints_1 = dataPoints; _i < dataPoints_1.length; _i++) {
        var point = dataPoints_1[_i];
        var zScore = Math.abs((point.value - mean) / standardDeviation);
        if (zScore > threshold) {
            var expectedMin = mean - (threshold * standardDeviation);
            var expectedMax = mean + (threshold * standardDeviation);
            var severity = 'low';
            if (zScore > 3.5)
                severity = 'high';
            else if (zScore > 2.5)
                severity = 'medium';
            anomalies.push({
                timestamp: point.timestamp,
                value: point.value,
                expectedRange: { min: expectedMin, max: expectedMax },
                deviation: zScore,
                severity: severity
            });
        }
    }
    return anomalies.sort(function (a, b) { return b.deviation - a.deviation; });
}
/**
 * Calculate linear regression for projection
 */
function calculateLinearProjection(dataPoints, daysAhead) {
    var _a;
    if (daysAhead === void 0) { daysAhead = 7; }
    if (dataPoints.length < 2) {
        return ((_a = dataPoints[0]) === null || _a === void 0 ? void 0 : _a.value) || 0;
    }
    var sortedData = __spreadArray([], dataPoints, true).sort(function (a, b) {
        return a.timestamp.getTime() - b.timestamp.getTime();
    });
    var n = sortedData.length;
    var xValues = sortedData.map(function (_, i) { return i; });
    var yValues = sortedData.map(function (p) { return p.value; });
    var sumX = xValues.reduce(function (a, b) { return a + b; }, 0);
    var sumY = yValues.reduce(function (a, b) { return a + b; }, 0);
    var sumXY = xValues.reduce(function (acc, x, i) { return acc + x * yValues[i]; }, 0);
    var sumXX = xValues.reduce(function (acc, x) { return acc + x * x; }, 0);
    var slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    var intercept = (sumY - slope * sumX) / n;
    // Project forward
    return slope * (n + daysAhead - 1) + intercept;
}
/**
 * Normalize metric value to 0-100 scale
 */
function normalizeMetricValue(value, min, max) {
    if (max === min)
        return 50; // Default to middle if no range
    return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
}

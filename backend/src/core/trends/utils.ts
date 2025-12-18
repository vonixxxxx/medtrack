/**
 * Utility functions for trend calculations
 */

import { subDays, addDays, startOfDay } from 'date-fns';
import { MetricDataPoint, MovingAverage, Anomaly } from './types';

/**
 * Calculate moving average for a period
 */
export function calculateMovingAverage(
  dataPoints: MetricDataPoint[],
  period: number,
  endDate: Date = new Date()
): MovingAverage[] {
  const averages: MovingAverage[] = [];
  const sortedData = [...dataPoints].sort((a, b) => 
    a.timestamp.getTime() - b.timestamp.getTime()
  );

  for (let i = period - 1; i < sortedData.length; i++) {
    const window = sortedData.slice(i - period + 1, i + 1);
    const sum = window.reduce((acc, point) => acc + point.value, 0);
    const average = sum / window.length;

    averages.push({
      period,
      value: average,
      date: window[window.length - 1].timestamp
    });
  }

  return averages;
}

/**
 * Calculate standard deviation
 */
export function calculateStandardDeviation(values: number[]): number {
  if (values.length === 0) return 0;
  
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
  
  return Math.sqrt(variance);
}

/**
 * Calculate coefficient of variation
 */
export function calculateCoefficientOfVariation(mean: number, standardDeviation: number): number {
  if (mean === 0) return 0;
  return standardDeviation / mean;
}

/**
 * Detect anomalies using z-score method
 */
export function detectAnomalies(
  dataPoints: MetricDataPoint[],
  mean: number,
  standardDeviation: number,
  threshold: number = 2.5 // Number of standard deviations
): Anomaly[] {
  const anomalies: Anomaly[] = [];

  for (const point of dataPoints) {
    const zScore = Math.abs((point.value - mean) / standardDeviation);
    
    if (zScore > threshold) {
      const expectedMin = mean - (threshold * standardDeviation);
      const expectedMax = mean + (threshold * standardDeviation);
      
      let severity: 'low' | 'medium' | 'high' = 'low';
      if (zScore > 3.5) severity = 'high';
      else if (zScore > 2.5) severity = 'medium';

      anomalies.push({
        timestamp: point.timestamp,
        value: point.value,
        expectedRange: { min: expectedMin, max: expectedMax },
        deviation: zScore,
        severity
      });
    }
  }

  return anomalies.sort((a, b) => b.deviation - a.deviation);
}

/**
 * Calculate linear regression for projection
 */
export function calculateLinearProjection(
  dataPoints: MetricDataPoint[],
  daysAhead: number = 7
): number {
  if (dataPoints.length < 2) {
    return dataPoints[0]?.value || 0;
  }

  const sortedData = [...dataPoints].sort((a, b) => 
    a.timestamp.getTime() - b.timestamp.getTime()
  );

  const n = sortedData.length;
  const xValues = sortedData.map((_, i) => i);
  const yValues = sortedData.map(p => p.value);

  const sumX = xValues.reduce((a, b) => a + b, 0);
  const sumY = yValues.reduce((a, b) => a + b, 0);
  const sumXY = xValues.reduce((acc, x, i) => acc + x * yValues[i], 0);
  const sumXX = xValues.reduce((acc, x) => acc + x * x, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  // Project forward
  return slope * (n + daysAhead - 1) + intercept;
}

/**
 * Normalize metric value to 0-100 scale
 */
export function normalizeMetricValue(
  value: number,
  min: number,
  max: number
): number {
  if (max === min) return 50; // Default to middle if no range
  return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
}








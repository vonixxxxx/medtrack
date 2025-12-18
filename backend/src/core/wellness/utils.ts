/**
 * Utility functions for wellness score calculations
 */

/**
 * Normalize a value to 0-100 scale
 */
export function normalizeToScore(value: number, min: number, max: number): number {
  if (max === min) return 50;
  return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
}

/**
 * Calculate inverse variability score (higher variability = lower score)
 */
export function calculateInverseVariabilityScore(
  coefficientOfVariation: number,
  maxCV: number = 1.0
): number {
  // Normalize CV to 0-1, then invert
  const normalizedCV = Math.min(1, coefficientOfVariation / maxCV);
  return (1 - normalizedCV) * 100;
}

/**
 * Calculate weighted average
 */
export function calculateWeightedAverage(
  values: Array<{ value: number; weight: number }>
): number {
  const totalWeight = values.reduce((sum, item) => sum + item.weight, 0);
  if (totalWeight === 0) return 0;
  
  const weightedSum = values.reduce((sum, item) => sum + (item.value * item.weight), 0);
  return weightedSum / totalWeight;
}

/**
 * Get baseline value (median of historical data)
 */
export function calculateBaseline(values: number[]): number {
  if (values.length === 0) return 0;
  
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/**
 * Calculate deviation from baseline in standard deviations
 */
export function calculateBaselineDeviation(
  currentValue: number,
  baseline: number,
  standardDeviation: number
): number {
  if (standardDeviation === 0) return 0;
  return (currentValue - baseline) / standardDeviation;
}








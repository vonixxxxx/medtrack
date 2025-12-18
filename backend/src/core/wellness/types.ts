/**
 * Types for Wellness Score Engine
 */

export interface WellnessScore {
  overallScore: number; // 0-100
  breakdown: {
    adherenceScore: number; // 0-100, weight: 30%
    metricScore: number; // 0-100, weight: 40%
    stabilityScore: number; // 0-100, weight: 20%
    energyOrSleepScore: number; // 0-100, weight: 10%
  };
  weights: {
    adherence: number;
    metrics: number;
    stability: number;
    energyOrSleep: number;
  };
  baselineAdjusted: boolean;
  timestamp: Date;
}

export interface WellnessBreakdown {
  adherence: {
    score: number;
    averageAdherence: number;
    medicationsCount: number;
  };
  metrics: {
    score: number;
    normalizedMetrics: Array<{
      name: string;
      normalizedValue: number;
      trend: 'improving' | 'declining' | 'stable' | 'volatile';
    }>;
  };
  stability: {
    score: number;
    averageVariability: number;
    metricsCount: number;
  };
  energyOrSleep: {
    score: number;
    metricName: string;
    value: number;
    available: boolean;
  };
}

export interface BaselineAdjustedMetrics {
  metricName: string;
  currentValue: number;
  baselineValue: number;
  deviation: number; // Standard deviations from baseline
  normalizedScore: number; // 0-100
  trend: 'above_baseline' | 'below_baseline' | 'at_baseline';
}








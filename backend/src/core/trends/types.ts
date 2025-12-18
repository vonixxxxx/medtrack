/**
 * Types for Progress-over-Time Engine
 */

export interface MetricDataPoint {
  timestamp: Date;
  value: number;
  unit?: string;
}

export interface MovingAverage {
  period: number; // 7, 14, or 30 days
  value: number;
  date: Date;
}

export interface TrendClassification {
  trend: 'improving' | 'declining' | 'stable' | 'volatile';
  confidence: number; // 0-1
  changePercentage: number; // Percentage change from baseline
  direction: number; // -1 to 1, negative = declining, positive = improving
}

export interface VariabilityScore {
  standardDeviation: number;
  coefficientOfVariation: number; // CV = SD / mean
  score: number; // 0-100, normalized
}

export interface Anomaly {
  timestamp: Date;
  value: number;
  expectedRange: { min: number; max: number };
  deviation: number; // How many standard deviations from mean
  severity: 'low' | 'medium' | 'high';
}

export interface MetricTrendAnalysis {
  metricName: string;
  movingAverages: {
    sevenDay: MovingAverage[];
    fourteenDay: MovingAverage[];
    thirtyDay: MovingAverage[];
  };
  trendClassification: TrendClassification;
  variabilityScore: VariabilityScore;
  anomalies: Anomaly[];
  baseline: {
    mean: number;
    median: number;
    standardDeviation: number;
  };
  trajectory: {
    current: number;
    projected: number; // Simple linear projection
    changeFromBaseline: number;
  };
}








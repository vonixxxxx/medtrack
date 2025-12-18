/**
 * Types for AI Summarizer
 */

export interface AdherenceSummary {
  overallAdherence: number;
  medications: Array<{
    medicationId: string;
    name: string;
    adherence: number;
    pattern: 'improving' | 'declining' | 'stable' | 'volatile';
    streak: number;
  }>;
  pattern: 'improving' | 'declining' | 'stable' | 'volatile';
}

export interface MetricTrendSummary {
  metricName: string;
  trend: 'improving' | 'declining' | 'stable' | 'volatile';
  currentValue: number;
  changePercentage: number;
  classification: string;
}

export interface SummarizerInput {
  adherenceSummary: AdherenceSummary;
  metricTrends: MetricTrendSummary[];
  anomalies: Array<{
    metricName: string;
    timestamp: Date;
    value: number;
    severity: 'low' | 'medium' | 'high';
  }>;
  wellnessScore: {
    overallScore: number;
    breakdown: {
      adherenceScore: number;
      metricScore: number;
      stabilityScore: number;
      energyOrSleepScore: number;
    };
  };
  streaks: Array<{
    medicationId: string;
    currentStreak: number;
    longestStreak: number;
  }>;
  correlationCandidates: Array<{
    metric1: string;
    metric2: string;
    correlation: number;
  }>;
  timeframe: string; // e.g., "30d", "7d"
}

export interface SummarizerOutput {
  overallStatus: string;
  progress: string;
  medicationAdherence: string;
  metricTrends: string;
  notableEvents: string;
  wellnessScoreInterpretation: string;
  recommendations: string[];
}








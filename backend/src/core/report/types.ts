/**
 * Types for Health Report Generator
 */

export interface HealthReport {
  timeframe: string;
  wellnessScore: {
    overallScore: number;
    breakdown: {
      adherenceScore: number;
      metricScore: number;
      stabilityScore: number;
      energyOrSleepScore: number;
    };
  };
  adherenceSummary: {
    overallAdherence: number;
    medications: Array<{
      medicationId: string;
      name: string;
      adherence: number;
      pattern: 'improving' | 'declining' | 'stable' | 'volatile';
      streak: number;
    }>;
    pattern: 'improving' | 'declining' | 'stable' | 'volatile';
  };
  metricTrendSummaries: Array<{
    metricName: string;
    trend: 'improving' | 'declining' | 'stable' | 'volatile';
    currentValue: number;
    changePercentage: number;
    classification: string;
  }>;
  anomalies: Array<{
    metricName: string;
    timestamp: Date;
    value: number;
    severity: 'low' | 'medium' | 'high';
  }>;
  narrativeSummary: {
    overallStatus: string;
    progress: string;
    medicationAdherence: string;
    metricTrends: string;
    notableEvents: string;
    wellnessScoreInterpretation: string;
    recommendations: string[];
  };
  recommendations: string[];
  generatedAt: Date;
}








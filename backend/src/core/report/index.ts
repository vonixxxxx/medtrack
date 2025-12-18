/**
 * Health Report Generator
 * 
 * Combines adherence, trends, wellness, and summarizer into comprehensive reports
 * Inspired by MediLog / Levels style summarization
 */

import { PrismaClient } from '@prisma/client';
import { AdherenceEngine } from '../adherence';
import { TrendsEngine } from '../trends';
import { WellnessEngine } from '../wellness';
import { SummarizerService } from '../summarizer';
import { HealthReport } from './types';

export class HealthReportGenerator {
  private adherenceEngine: AdherenceEngine;
  private trendsEngine: TrendsEngine;
  private wellnessEngine: WellnessEngine;
  private summarizer: SummarizerService;

  constructor(
    private prisma: PrismaClient,
    summarizerConfig?: {
      provider?: 'openai' | 'anthropic' | 'ollama' | 'local';
      apiKey?: string;
      apiUrl?: string;
      model?: string;
    }
  ) {
    this.adherenceEngine = new AdherenceEngine(prisma);
    this.trendsEngine = new TrendsEngine(prisma);
    this.wellnessEngine = new WellnessEngine(prisma);
    this.summarizer = new SummarizerService(summarizerConfig);
  }

  /**
   * Generate comprehensive health report
   */
  async generateHealthReport(
    userId: string,
    timeframe: string = '30d'
  ): Promise<HealthReport> {
    const days = this.parseTimeframe(timeframe);

    // Get all data in parallel
    const [adherenceData, metricsTrends, wellnessScore, wellnessBreakdown] = await Promise.all([
      this.adherenceEngine.getAllMedicationsAdherence(userId, { startDate: undefined, endDate: undefined }),
      this.trendsEngine.getAllMetricsTrends(userId, days),
      this.wellnessEngine.calculateWellnessScore(userId, days),
      this.wellnessEngine.getWellnessBreakdown(userId, days)
    ]);

    // Build adherence summary
    const overallAdherence = adherenceData.length > 0
      ? adherenceData.reduce((sum, med) => sum + med.adherencePercentage, 0) / adherenceData.length
      : 100;

    const adherencePatterns = await Promise.all(
      adherenceData.map(med => 
        this.adherenceEngine.analyzeAdherencePatterns(med.medicationId, userId, days)
      )
    );

    const adherenceSummary = {
      overallAdherence,
      medications: adherenceData.map((med, idx) => ({
        medicationId: med.medicationId,
        name: med.medicationId, // Would need to fetch actual name from cycle
        adherence: med.adherencePercentage,
        pattern: adherencePatterns[idx].pattern,
        streak: 0 // Will be populated below
      })),
      pattern: this.determineOverallPattern(adherencePatterns.map(p => p.pattern))
    };

    // Get streaks
    const streaks = await Promise.all(
      adherenceData.map(async (med) => {
        const streakData = await this.adherenceEngine.getMedicationStreaks(med.medicationId, userId, days);
        return {
          medicationId: med.medicationId,
          currentStreak: streakData.currentStreak,
          longestStreak: streakData.longestStreak
        };
      })
    );

    // Update adherence summary with streaks
    for (const streak of streaks) {
      const med = adherenceSummary.medications.find(m => m.medicationId === streak.medicationId);
      if (med) {
        med.streak = streak.currentStreak;
      }
    }

    // Build metric trend summaries
    const metricTrendSummaries = Array.from(metricsTrends.entries()).map(([metricName, analysis]) => ({
      metricName,
      trend: analysis.trendClassification.trend,
      currentValue: analysis.trajectory.current,
      changePercentage: analysis.trendClassification.changePercentage,
      classification: `${analysis.trendClassification.trend} (confidence: ${(analysis.trendClassification.confidence * 100).toFixed(0)}%)`
    }));

    // Collect anomalies
    const allAnomalies: HealthReport['anomalies'] = [];
    for (const [metricName, analysis] of metricsTrends.entries()) {
      for (const anomaly of analysis.anomalies) {
        allAnomalies.push({
          metricName,
          timestamp: anomaly.timestamp,
          value: anomaly.value,
          severity: anomaly.severity
        });
      }
    }

    // Calculate correlations (simple correlation between metrics)
    const correlationCandidates = this.calculateCorrelations(metricsTrends);

    // Generate narrative summary using AI
    const narrativeSummary = await this.summarizer.generateSummary({
      adherenceSummary,
      metricTrends: metricTrendSummaries,
      anomalies: allAnomalies,
      wellnessScore: {
        overallScore: wellnessScore.overallScore,
        breakdown: wellnessScore.breakdown
      },
      streaks,
      correlationCandidates,
      timeframe
    });

    return {
      timeframe,
      wellnessScore: {
        overallScore: wellnessScore.overallScore,
        breakdown: wellnessScore.breakdown
      },
      adherenceSummary,
      metricTrendSummaries,
      anomalies: allAnomalies,
      narrativeSummary,
      recommendations: narrativeSummary.recommendations,
      generatedAt: new Date()
    };
  }

  /**
   * Parse timeframe string to days
   */
  private parseTimeframe(timeframe: string): number {
    const match = timeframe.match(/(\d+)([dwmy])/);
    if (!match) return 30;

    const value = parseInt(match[1]);
    const unit = match[2];

    switch (unit) {
      case 'd': return value;
      case 'w': return value * 7;
      case 'm': return value * 30;
      case 'y': return value * 365;
      default: return 30;
    }
  }

  /**
   * Determine overall pattern from individual patterns
   */
  private determineOverallPattern(
    patterns: Array<'improving' | 'declining' | 'stable' | 'volatile'>
  ): 'improving' | 'declining' | 'stable' | 'volatile' {
    if (patterns.length === 0) return 'stable';

    const counts = {
      improving: 0,
      declining: 0,
      stable: 0,
      volatile: 0
    };

    for (const pattern of patterns) {
      counts[pattern]++;
    }

    // If volatile is most common, return volatile
    if (counts.volatile > patterns.length / 2) return 'volatile';

    // Otherwise, return the most common pattern
    const maxCount = Math.max(counts.improving, counts.declining, counts.stable);
    if (maxCount === counts.improving) return 'improving';
    if (maxCount === counts.declining) return 'declining';
    return 'stable';
  }

  /**
   * Calculate simple correlations between metrics
   */
  private calculateCorrelations(
    metricsTrends: Map<string, import('../trends').MetricTrendAnalysis>
  ): Array<{ metric1: string; metric2: string; correlation: number }> {
    const correlations: Array<{ metric1: string; metric2: string; correlation: number }> = [];
    const metrics = Array.from(metricsTrends.entries());

    // Simple correlation calculation (Pearson correlation coefficient)
    for (let i = 0; i < metrics.length; i++) {
      for (let j = i + 1; j < metrics.length; j++) {
        const [name1, analysis1] = metrics[i];
        const [name2, analysis2] = metrics[j];

        // Use moving averages for correlation
        const values1 = analysis1.movingAverages.sevenDay.map(ma => ma.value);
        const values2 = analysis2.movingAverages.sevenDay.map(ma => ma.value);

        if (values1.length === values2.length && values1.length > 1) {
          const correlation = this.calculatePearsonCorrelation(values1, values2);
          
          // Only include moderate to strong correlations
          if (Math.abs(correlation) > 0.3) {
            correlations.push({
              metric1: name1,
              metric2: name2,
              correlation
            });
          }
        }
      }
    }

    return correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  private calculatePearsonCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;

    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, val, i) => acc + val * y[i], 0);
    const sumXX = x.reduce((acc, val) => acc + val * val, 0);
    const sumYY = y.reduce((acc, val) => acc + val * val, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));

    if (denominator === 0) return 0;
    return numerator / denominator;
  }
}








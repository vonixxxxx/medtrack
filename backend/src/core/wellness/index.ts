/**
 * Wellness Score Engine
 * 
 * Composite score combining adherence, metrics, stability, and energy/sleep
 * Inspired by Oura/Whoop's readiness score
 */

import { PrismaClient } from '@prisma/client';
import { subDays, endOfDay } from 'date-fns';
import { AdherenceEngine } from '../adherence';
import { TrendsEngine } from '../trends';
import {
  WellnessScore,
  WellnessBreakdown,
  BaselineAdjustedMetrics
} from './types';
import {
  normalizeToScore,
  calculateInverseVariabilityScore,
  calculateWeightedAverage,
  calculateBaseline,
  calculateBaselineDeviation
} from './utils';

export class WellnessEngine {
  private adherenceEngine: AdherenceEngine;
  private trendsEngine: TrendsEngine;

  constructor(private prisma: PrismaClient) {
    this.adherenceEngine = new AdherenceEngine(prisma);
    this.trendsEngine = new TrendsEngine(prisma);
  }

  /**
   * Calculate overall wellness score
   */
  async calculateWellnessScore(
    userId: string,
    days: number = 30
  ): Promise<WellnessScore> {
    // Get adherence score (weight: 30%)
    const adherenceData = await this.adherenceEngine.getAllMedicationsAdherence(userId);
    const averageAdherence = adherenceData.length > 0
      ? adherenceData.reduce((sum, med) => sum + med.adherencePercentage, 0) / adherenceData.length
      : 100;
    const adherenceScore = averageAdherence;

    // Get metrics score (weight: 40%)
    const metricsTrends = await this.trendsEngine.getAllMetricsTrends(userId, days);
    const metricScore = this.calculateMetricsScore(metricsTrends);

    // Get stability score (weight: 20%)
    const stabilityScore = this.calculateStabilityScore(metricsTrends);

    // Get energy or sleep score (weight: 10%)
    const energyOrSleepScore = await this.getEnergyOrSleepScore(userId, days);

    // Calculate weighted wellness score
    const weights = {
      adherence: 0.3,
      metrics: 0.4,
      stability: 0.2,
      energyOrSleep: 0.1
    };

    const overallScore = calculateWeightedAverage([
      { value: adherenceScore, weight: weights.adherence },
      { value: metricScore, weight: weights.metrics },
      { value: stabilityScore, weight: weights.stability },
      { value: energyOrSleepScore, weight: weights.energyOrSleep }
    ]);

    return {
      overallScore: Math.round(overallScore * 100) / 100,
      breakdown: {
        adherenceScore: Math.round(adherenceScore * 100) / 100,
        metricScore: Math.round(metricScore * 100) / 100,
        stabilityScore: Math.round(stabilityScore * 100) / 100,
        energyOrSleepScore: Math.round(energyOrSleepScore * 100) / 100
      },
      weights,
      baselineAdjusted: true,
      timestamp: new Date()
    };
  }

  /**
   * Calculate metrics score from normalized metrics
   */
  private calculateMetricsScore(
    metricsTrends: Map<string, import('../trends').MetricTrendAnalysis>
  ): number {
    if (metricsTrends.size === 0) return 50; // Default if no metrics

    const normalizedScores: number[] = [];

    for (const [metricName, analysis] of metricsTrends.entries()) {
      // Skip energy/sleep metrics (handled separately)
      if (['ENERGY', 'SLEEP', 'SLEEP_HOURS', 'ENERGY_LEVEL'].includes(metricName.toUpperCase())) {
        continue;
      }

      const current = analysis.trajectory.current;
      const baseline = analysis.baseline.mean;
      const stdDev = analysis.baseline.standardDeviation;

      // Normalize based on trend direction
      // For metrics where higher is better (e.g., mood, energy)
      // For metrics where lower is better (e.g., pain, blood pressure), we'll need to invert
      const isHigherBetter = this.isHigherBetterMetric(metricName);
      
      let normalizedValue: number;
      if (stdDev === 0) {
        normalizedValue = current === baseline ? 50 : (isHigherBetter ? (current > baseline ? 75 : 25) : (current < baseline ? 75 : 25));
      } else {
        // Use z-score normalization, then convert to 0-100
        const zScore = (current - baseline) / stdDev;
        if (isHigherBetter) {
          normalizedValue = normalizeToScore(zScore, -3, 3);
        } else {
          // Invert for lower-is-better metrics
          normalizedValue = normalizeToScore(-zScore, -3, 3);
        }
      }

      normalizedScores.push(normalizedValue);
    }

    return normalizedScores.length > 0
      ? normalizedScores.reduce((a, b) => a + b, 0) / normalizedScores.length
      : 50;
  }

  /**
   * Determine if higher values are better for a metric
   */
  private isHigherBetterMetric(metricName: string): boolean {
    const higherIsBetter = [
      'MOOD', 'ENERGY', 'SLEEP_QUALITY', 'SLEEP_HOURS', 'ENERGY_LEVEL',
      'HAPPINESS', 'VITALITY', 'WELLBEING'
    ];
    
    const lowerIsBetter = [
      'PAIN', 'BLOOD_PRESSURE', 'HEART_RATE', 'STRESS', 'ANXIETY',
      'WEIGHT', 'BMI', 'BLOOD_SUGAR'
    ];

    const upperName = metricName.toUpperCase();
    if (higherIsBetter.some(m => upperName.includes(m))) return true;
    if (lowerIsBetter.some(m => upperName.includes(m))) return false;
    
    // Default: assume higher is better
    return true;
  }

  /**
   * Calculate stability score from variability
   */
  private calculateStabilityScore(
    metricsTrends: Map<string, import('../trends').MetricTrendAnalysis>
  ): number {
    if (metricsTrends.size === 0) return 50;

    const variabilityScores: number[] = [];

    for (const analysis of metricsTrends.values()) {
      const cv = analysis.variabilityScore.coefficientOfVariation;
      const stability = calculateInverseVariabilityScore(cv);
      variabilityScores.push(stability);
    }

    return variabilityScores.length > 0
      ? variabilityScores.reduce((a, b) => a + b, 0) / variabilityScores.length
      : 50;
  }

  /**
   * Get energy or sleep score
   */
  private async getEnergyOrSleepScore(
    userId: string,
    days: number
  ): Promise<number> {
    const endDate = endOfDay(new Date());
    const startDate = subDays(endDate, days);

    // Try to find energy or sleep metrics
    const cycles = await this.prisma.medicationCycle.findMany({
      where: {
        userId: userId
      },
      include: {
        metricLogs: {
          where: {
            kind: {
              in: ['ENERGY', 'SLEEP', 'SLEEP_HOURS', 'ENERGY_LEVEL']
            },
            date: {
              gte: startDate,
              lte: endDate
            }
          },
          orderBy: {
            date: 'desc'
          },
          take: 30
        }
      }
    });

    const energyOrSleepValues: number[] = [];
    for (const cycle of cycles) {
      for (const log of cycle.metricLogs) {
        const value = log.valueFloat || parseFloat(log.valueText || '0');
        if (!isNaN(value)) {
          energyOrSleepValues.push(value);
        }
      }
    }

    if (energyOrSleepValues.length === 0) {
      return 50; // Default if no energy/sleep data
    }

    // Normalize energy/sleep values
    // Assume typical ranges: energy 1-10, sleep 0-12 hours
    const avgValue = energyOrSleepValues.reduce((a, b) => a + b, 0) / energyOrSleepValues.length;
    
    // Normalize to 0-100 (assuming 0-10 scale for energy, or 0-12 for sleep hours)
    return normalizeToScore(avgValue, 0, 10);
  }

  /**
   * Get wellness breakdown
   */
  async getWellnessBreakdown(
    userId: string,
    days: number = 30
  ): Promise<WellnessBreakdown> {
    const adherenceData = await this.adherenceEngine.getAllMedicationsAdherence(userId);
    const metricsTrends = await this.trendsEngine.getAllMetricsTrends(userId, days);

    // Adherence breakdown
    const averageAdherence = adherenceData.length > 0
      ? adherenceData.reduce((sum, med) => sum + med.adherencePercentage, 0) / adherenceData.length
      : 100;

    // Metrics breakdown
    const normalizedMetrics: WellnessBreakdown['metrics']['normalizedMetrics'] = [];
    for (const [metricName, analysis] of metricsTrends.entries()) {
      if (['ENERGY', 'SLEEP', 'SLEEP_HOURS', 'ENERGY_LEVEL'].includes(metricName.toUpperCase())) {
        continue;
      }

      const current = analysis.trajectory.current;
      const baseline = analysis.baseline.mean;
      const stdDev = analysis.baseline.standardDeviation;
      const isHigherBetter = this.isHigherBetterMetric(metricName);

      let normalizedValue: number;
      if (stdDev === 0) {
        normalizedValue = 50;
      } else {
        const zScore = (current - baseline) / stdDev;
        normalizedValue = isHigherBetter
          ? normalizeToScore(zScore, -3, 3)
          : normalizeToScore(-zScore, -3, 3);
      }

      normalizedMetrics.push({
        name: metricName,
        normalizedValue,
        trend: analysis.trendClassification.trend
      });
    }

    const metricScore = normalizedMetrics.length > 0
      ? normalizedMetrics.reduce((sum, m) => sum + m.normalizedValue, 0) / normalizedMetrics.length
      : 50;

    // Stability breakdown
    const variabilityScores: number[] = [];
    for (const analysis of metricsTrends.values()) {
      const cv = analysis.variabilityScore.coefficientOfVariation;
      variabilityScores.push(calculateInverseVariabilityScore(cv));
    }

    const stabilityScore = variabilityScores.length > 0
      ? variabilityScores.reduce((a, b) => a + b, 0) / variabilityScores.length
      : 50;

    // Energy/Sleep breakdown
    const energyOrSleepData = await this.getEnergyOrSleepData(userId, days);

    return {
      adherence: {
        score: averageAdherence,
        averageAdherence,
        medicationsCount: adherenceData.length
      },
      metrics: {
        score: metricScore,
        normalizedMetrics
      },
      stability: {
        score: stabilityScore,
        averageVariability: variabilityScores.length > 0
          ? variabilityScores.reduce((a, b) => a + b, 0) / variabilityScores.length
          : 0,
        metricsCount: metricsTrends.size
      },
      energyOrSleep: energyOrSleepData
    };
  }

  /**
   * Get energy or sleep data
   */
  private async getEnergyOrSleepData(
    userId: string,
    days: number
  ): Promise<WellnessBreakdown['energyOrSleep']> {
    const endDate = endOfDay(new Date());
    const startDate = subDays(endDate, days);

    const cycles = await this.prisma.medicationCycle.findMany({
      where: {
        userId: userId
      },
      include: {
        metricLogs: {
          where: {
            kind: {
              in: ['ENERGY', 'SLEEP', 'SLEEP_HOURS', 'ENERGY_LEVEL']
            },
            date: {
              gte: startDate,
              lte: endDate
            }
          },
          orderBy: {
            date: 'desc'
          },
          take: 1
        }
      }
    });

    for (const cycle of cycles) {
      for (const log of cycle.metricLogs) {
        const value = log.valueFloat || parseFloat(log.valueText || '0');
        if (!isNaN(value)) {
          return {
            score: normalizeToScore(value, 0, 10),
            metricName: log.kind,
            value,
            available: true
          };
        }
      }
    }

    return {
      score: 50,
      metricName: 'NONE',
      value: 0,
      available: false
    };
  }

  /**
   * Compute baseline-adjusted metrics
   */
  async computeBaselineAdjustedMetrics(
    userId: string,
    days: number = 30
  ): Promise<BaselineAdjustedMetrics[]> {
    const metricsTrends = await this.trendsEngine.getAllMetricsTrends(userId, days);
    const adjustedMetrics: BaselineAdjustedMetrics[] = [];

    for (const [metricName, analysis] of metricsTrends.entries()) {
      const current = analysis.trajectory.current;
      const baseline = analysis.baseline.mean;
      const stdDev = analysis.baseline.standardDeviation;
      const deviation = calculateBaselineDeviation(current, baseline, stdDev);

      const isHigherBetter = this.isHigherBetterMetric(metricName);
      let normalizedScore: number;
      
      if (stdDev === 0) {
        normalizedScore = current === baseline ? 50 : (isHigherBetter ? (current > baseline ? 75 : 25) : (current < baseline ? 75 : 25));
      } else {
        const zScore = deviation;
        normalizedScore = isHigherBetter
          ? normalizeToScore(zScore, -3, 3)
          : normalizeToScore(-zScore, -3, 3);
      }

      let trend: BaselineAdjustedMetrics['trend'];
      if (Math.abs(deviation) < 0.5) {
        trend = 'at_baseline';
      } else if (deviation > 0) {
        trend = isHigherBetter ? 'above_baseline' : 'below_baseline';
      } else {
        trend = isHigherBetter ? 'below_baseline' : 'above_baseline';
      }

      adjustedMetrics.push({
        metricName,
        currentValue: current,
        baselineValue: baseline,
        deviation,
        normalizedScore,
        trend
      });
    }

    return adjustedMetrics;
  }
}








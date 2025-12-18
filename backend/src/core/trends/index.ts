/**
 * Progress-over-Time Engine for Metrics
 * 
 * Computes moving averages, trends, variability, and anomalies
 * Inspired by Daylio, Bearable, Levels, Tidepool
 */

import { PrismaClient } from '@prisma/client';
import { subDays, startOfDay, endOfDay } from 'date-fns';
import {
  MetricTrendAnalysis,
  MetricDataPoint,
  TrendClassification,
  VariabilityScore,
  Anomaly
} from './types';
import {
  calculateMovingAverage,
  calculateStandardDeviation,
  calculateCoefficientOfVariation,
  detectAnomalies,
  calculateLinearProjection,
  normalizeMetricValue
} from './utils';

export type { MetricTrendAnalysis } from './types';

export class TrendsEngine {
  constructor(private prisma: PrismaClient) {}

  /**
   * Compute metric trends
   */
  async computeMetricTrends(
    metricName: string,
    userId: string,
    days: number = 30
  ): Promise<MetricTrendAnalysis> {
    const endDate = endOfDay(new Date());
    const startDate = startOfDay(subDays(endDate, days));

    // Get metric logs from cycles
    const cycles = await this.prisma.medicationCycle.findMany({
      where: {
        userId: userId
      },
      include: {
        metricLogs: {
          where: {
            kind: metricName.toUpperCase(),
            date: {
              gte: startDate,
              lte: endDate
            }
          },
          orderBy: {
            date: 'asc'
          }
        }
      }
    });

    // Convert to data points
    const dataPoints: MetricDataPoint[] = [];
    for (const cycle of cycles) {
      for (const log of cycle.metricLogs) {
        const value = log.valueFloat || parseFloat(log.valueText || '0');
        if (!isNaN(value)) {
          dataPoints.push({
            timestamp: log.date,
            value: value,
            unit: log.notes || undefined
          });
        }
      }
    }

    if (dataPoints.length === 0) {
      throw new Error(`No data points found for metric: ${metricName}`);
    }

    // Calculate moving averages
    const sevenDayMA = calculateMovingAverage(dataPoints, 7, endDate);
    const fourteenDayMA = calculateMovingAverage(dataPoints, 14, endDate);
    const thirtyDayMA = calculateMovingAverage(dataPoints, Math.min(30, dataPoints.length), endDate);

    // Calculate baseline statistics
    const values = dataPoints.map(p => p.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sortedValues = [...values].sort((a, b) => a - b);
    const median = sortedValues.length % 2 === 0
      ? (sortedValues[sortedValues.length / 2 - 1] + sortedValues[sortedValues.length / 2]) / 2
      : sortedValues[Math.floor(sortedValues.length / 2)];
    const standardDeviation = calculateStandardDeviation(values);

    // Calculate trend classification
    const trendClassification = this.getTrendClassification(
      dataPoints,
      mean,
      standardDeviation
    );

    // Calculate variability score
    const coefficientOfVariation = calculateCoefficientOfVariation(mean, standardDeviation);
    const variabilityScore: VariabilityScore = {
      standardDeviation,
      coefficientOfVariation,
      score: Math.min(100, Math.max(0, 100 - (coefficientOfVariation * 100)))
    };

    // Detect anomalies
    const anomalies = detectAnomalies(dataPoints, mean, standardDeviation);

    // Calculate trajectory
    const current = dataPoints[dataPoints.length - 1]?.value || mean;
    const projected = calculateLinearProjection(dataPoints, 7);
    const changeFromBaseline = current - mean;

    return {
      metricName,
      movingAverages: {
        sevenDay: sevenDayMA,
        fourteenDay: fourteenDayMA,
        thirtyDay: thirtyDayMA
      },
      trendClassification,
      variabilityScore,
      anomalies,
      baseline: {
        mean,
        median,
        standardDeviation
      },
      trajectory: {
        current,
        projected,
        changeFromBaseline
      }
    };
  }

  /**
   * Get trend classification
   */
  getTrendClassification(
    dataPoints: MetricDataPoint[],
    baselineMean: number,
    baselineStdDev: number
  ): TrendClassification {
    if (dataPoints.length < 2) {
      return {
        trend: 'stable',
        confidence: 0,
        changePercentage: 0,
        direction: 0
      };
    }

    const sortedData = [...dataPoints].sort((a, b) => 
      a.timestamp.getTime() - b.timestamp.getTime()
    );

    // Compare first third to last third
    const thirdSize = Math.floor(sortedData.length / 3);
    const firstThird = sortedData.slice(0, thirdSize);
    const lastThird = sortedData.slice(-thirdSize);

    const firstThirdAvg = firstThird.reduce((acc, p) => acc + p.value, 0) / firstThird.length;
    const lastThirdAvg = lastThird.reduce((acc, p) => acc + p.value, 0) / lastThird.length;

    const change = lastThirdAvg - firstThirdAvg;
    const changePercentage = (change / baselineMean) * 100;
    const direction = change > 0 ? 1 : change < 0 ? -1 : 0;

    // Calculate volatility
    const allValues = sortedData.map(p => p.value);
    const volatility = calculateStandardDeviation(allValues) / baselineMean;

    // Determine trend
    let trend: TrendClassification['trend'] = 'stable';
    let confidence = 0.5;

    if (volatility > 0.3) {
      trend = 'volatile';
      confidence = 0.7;
    } else if (Math.abs(changePercentage) > 10) {
      trend = changePercentage > 0 ? 'improving' : 'declining';
      confidence = Math.min(0.95, 0.5 + Math.abs(changePercentage) / 50);
    } else {
      trend = 'stable';
      confidence = 0.8;
    }

    return {
      trend,
      confidence,
      changePercentage,
      direction
    };
  }

  /**
   * Get metric trajectory
   */
  async getMetricTrajectory(
    metricName: string,
    userId: string,
    days: number = 30
  ): Promise<MetricTrendAnalysis['trajectory']> {
    const analysis = await this.computeMetricTrends(metricName, userId, days);
    return analysis.trajectory;
  }

  /**
   * Detect metric anomalies
   */
  async detectMetricAnomalies(
    metricName: string,
    userId: string,
    days: number = 30,
    threshold: number = 2.5
  ): Promise<Anomaly[]> {
    const analysis = await this.computeMetricTrends(metricName, userId, days);
    
    // Re-detect with custom threshold
    const endDate = endOfDay(new Date());
    const startDate = startOfDay(subDays(endDate, days));

    const cycles = await this.prisma.medicationCycle.findMany({
      where: {
        userId: userId
      },
      include: {
        metricLogs: {
          where: {
            kind: metricName.toUpperCase(),
            date: {
              gte: startDate,
              lte: endDate
            }
          }
        }
      }
    });

    const dataPoints: MetricDataPoint[] = [];
    for (const cycle of cycles) {
      for (const log of cycle.metricLogs) {
        const value = log.valueFloat || parseFloat(log.valueText || '0');
        if (!isNaN(value)) {
          dataPoints.push({
            timestamp: log.date,
            value: value
          });
        }
      }
    }

    return detectAnomalies(
      dataPoints,
      analysis.baseline.mean,
      analysis.baseline.standardDeviation,
      threshold
    );
  }

  /**
   * Get all metrics trends for a user
   */
  async getAllMetricsTrends(
    userId: string,
    days: number = 30
  ): Promise<Map<string, MetricTrendAnalysis>> {
    // Get all unique metric types
    const cycles = await this.prisma.medicationCycle.findMany({
      where: {
        userId: userId
      },
      include: {
        metricLogs: {
          distinct: ['kind'],
          select: {
            kind: true
          }
        }
      }
    });

    const metricTypes = new Set<string>();
    for (const cycle of cycles) {
      for (const log of cycle.metricLogs) {
        if (log.kind) {
          metricTypes.add(log.kind);
        }
      }
    }

    const trendsMap = new Map<string, MetricTrendAnalysis>();
    
    for (const metricType of metricTypes) {
      try {
        const analysis = await this.computeMetricTrends(metricType, userId, days);
        trendsMap.set(metricType, analysis);
      } catch (error) {
        // Skip metrics with no data
        console.warn(`No data for metric: ${metricType}`);
      }
    }

    return trendsMap;
  }
}


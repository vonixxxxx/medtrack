/**
 * Trends Controller
 * Handles API requests for metric trends
 */

import { Request, Response } from 'express';
import { TrendsEngine } from '../core/trends';

export class TrendsController {
  private engine: TrendsEngine;

  constructor(prisma: any) {
    this.engine = new TrendsEngine(prisma);
  }

  /**
   * GET /api/metrics/trends
   * Get trends for all metrics
   */
  async getAll(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const days = parseInt(req.query.days as string) || 30;
      const trendsMap = await this.engine.getAllMetricsTrends(userId, days);

      // Convert Map to object
      const trends: Record<string, any> = {};
      for (const [metricName, analysis] of trendsMap.entries()) {
        trends[metricName] = analysis;
      }

      res.json({
        success: true,
        data: trends,
        timeframe: `${days} days`
      });
    } catch (error: any) {
      console.error('Trends controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to fetch trends'
      });
    }
  }

  /**
   * GET /api/metrics/trends/:metricName
   * Get trends for a specific metric
   */
  async getOne(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const { metricName } = req.params;
      const days = parseInt(req.query.days as string) || 30;

      const analysis = await this.engine.computeMetricTrends(metricName, userId, days);

      res.json({
        success: true,
        data: analysis
      });
    } catch (error: any) {
      console.error('Trends controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to fetch metric trends'
      });
    }
  }

  /**
   * GET /api/metrics/trends/:metricName/classification
   * Get trend classification
   */
  async getClassification(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const { metricName } = req.params;
      const days = parseInt(req.query.days as string) || 30;

      const analysis = await this.engine.computeMetricTrends(metricName, userId, days);

      res.json({
        success: true,
        data: analysis.trendClassification
      });
    } catch (error: any) {
      console.error('Trends controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to get trend classification'
      });
    }
  }

  /**
   * GET /api/metrics/trends/:metricName/anomalies
   * Detect anomalies for a metric
   */
  async getAnomalies(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const { metricName } = req.params;
      const days = parseInt(req.query.days as string) || 30;
      const threshold = parseFloat(req.query.threshold as string) || 2.5;

      const anomalies = await this.engine.detectMetricAnomalies(metricName, userId, days, threshold);

      res.json({
        success: true,
        data: anomalies
      });
    } catch (error: any) {
      console.error('Trends controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to detect anomalies'
      });
    }
  }

  /**
   * GET /api/metrics/trends/:metricName/trajectory
   * Get metric trajectory
   */
  async getTrajectory(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const { metricName } = req.params;
      const days = parseInt(req.query.days as string) || 30;

      const trajectory = await this.engine.getMetricTrajectory(metricName, userId, days);

      res.json({
        success: true,
        data: trajectory
      });
    } catch (error: any) {
      console.error('Trends controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to get trajectory'
      });
    }
  }
}








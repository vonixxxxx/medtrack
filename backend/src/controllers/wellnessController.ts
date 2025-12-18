/**
 * Wellness Controller
 * Handles API requests for wellness scores
 */

import { Request, Response } from 'express';
import { WellnessEngine } from '../core/wellness';

export class WellnessController {
  private engine: WellnessEngine;

  constructor(prisma: any) {
    this.engine = new WellnessEngine(prisma);
  }

  /**
   * GET /api/wellness
   * Get overall wellness score
   */
  async getScore(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const days = parseInt(req.query.days as string) || 30;
      const wellnessScore = await this.engine.calculateWellnessScore(userId, days);

      res.json({
        success: true,
        data: wellnessScore
      });
    } catch (error: any) {
      console.error('Wellness controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to calculate wellness score'
      });
    }
  }

  /**
   * GET /api/wellness/breakdown
   * Get detailed wellness breakdown
   */
  async getBreakdown(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const days = parseInt(req.query.days as string) || 30;
      const breakdown = await this.engine.getWellnessBreakdown(userId, days);

      res.json({
        success: true,
        data: breakdown
      });
    } catch (error: any) {
      console.error('Wellness controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to get wellness breakdown'
      });
    }
  }

  /**
   * GET /api/wellness/baseline-adjusted
   * Get baseline-adjusted metrics
   */
  async getBaselineAdjusted(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const days = parseInt(req.query.days as string) || 30;
      const adjustedMetrics = await this.engine.computeBaselineAdjustedMetrics(userId, days);

      res.json({
        success: true,
        data: adjustedMetrics
      });
    } catch (error: any) {
      console.error('Wellness controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to get baseline-adjusted metrics'
      });
    }
  }
}








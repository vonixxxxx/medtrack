/**
 * Adherence Controller
 * Handles API requests for medication adherence
 */

import { Request, Response } from 'express';
import { AdherenceEngine } from '../core/adherence';

export class AdherenceController {
  private engine: AdherenceEngine;

  constructor(prisma: any) {
    this.engine = new AdherenceEngine(prisma);
  }

  /**
   * GET /api/adherence
   * Get adherence for all medications
   */
  async getAll(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const days = parseInt(req.query.days as string) || 30;
      const adherenceData = await this.engine.getAllMedicationsAdherence(userId, {
        startDate: req.query.startDate ? new Date(req.query.startDate as string) : undefined,
        endDate: req.query.endDate ? new Date(req.query.endDate as string) : undefined
      });

      res.json({
        success: true,
        data: adherenceData,
        timeframe: `${days} days`
      });
    } catch (error: any) {
      console.error('Adherence controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to fetch adherence data'
      });
    }
  }

  /**
   * GET /api/adherence/:medicationId
   * Get adherence for a specific medication
   */
  async getOne(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const { medicationId } = req.params;
      const period = (req.query.period as string) || 'weekly';

      let adherenceData;
      if (period === 'daily') {
        adherenceData = await this.engine.getDailyAdherence(
          medicationId,
          userId,
          req.query.date ? new Date(req.query.date as string) : undefined
        );
      } else {
        const weeks = parseInt(req.query.weeks as string) || 4;
        adherenceData = await this.engine.getWeeklyAdherence(medicationId, userId, weeks);
      }

      res.json({
        success: true,
        data: adherenceData
      });
    } catch (error: any) {
      console.error('Adherence controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to fetch adherence data'
      });
    }
  }

  /**
   * GET /api/adherence/:medicationId/streaks
   * Get streak data for a medication
   */
  async getStreaks(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const { medicationId } = req.params;
      const days = parseInt(req.query.days as string) || 30;

      const streakData = await this.engine.getMedicationStreaks(medicationId, userId, days);

      res.json({
        success: true,
        data: streakData
      });
    } catch (error: any) {
      console.error('Adherence controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to fetch streak data'
      });
    }
  }

  /**
   * GET /api/adherence/:medicationId/patterns
   * Analyze adherence patterns
   */
  async getPatterns(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const { medicationId } = req.params;
      const days = parseInt(req.query.days as string) || 30;

      const patternData = await this.engine.analyzeAdherencePatterns(medicationId, userId, days);

      res.json({
        success: true,
        data: patternData
      });
    } catch (error: any) {
      console.error('Adherence controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to analyze patterns'
      });
    }
  }
}








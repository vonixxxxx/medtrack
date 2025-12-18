/**
 * Health Report Controller
 * Handles API requests for health reports
 */

import { Request, Response } from 'express';
import { HealthReportGenerator } from '../core/report';

export class HealthReportController {
  private generator: HealthReportGenerator;

  constructor(prisma: any) {
    this.generator = new HealthReportGenerator(prisma, {
      provider: (process.env.AI_PROVIDER as any) || 'local',
      apiKey: process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY,
      apiUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
      model: process.env.AI_MODEL || 'llama3.2'
    });
  }

  /**
   * GET /api/health-report
   * Generate comprehensive health report
   */
  async generate(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const timeframe = (req.query.timeframe as string) || '30d';
      const report = await this.generator.generateHealthReport(userId, timeframe);

      res.json({
        success: true,
        data: report
      });
    } catch (error: any) {
      console.error('Health report controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to generate health report'
      });
    }
  }

  /**
   * GET /api/health-report/download
   * Download health report as JSON
   */
  async download(req: Request, res: Response) {
    try {
      const userId = (req as any).user?.id;
      if (!userId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const timeframe = (req.query.timeframe as string) || '30d';
      const report = await this.generator.generateHealthReport(userId, timeframe);

      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Content-Disposition', `attachment; filename=health-report-${Date.now()}.json`);
      res.json(report);
    } catch (error: any) {
      console.error('Health report controller error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to generate health report'
      });
    }
  }
}








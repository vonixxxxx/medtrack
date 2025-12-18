/**
 * Wellness Routes
 */

import { Router } from 'express';
import { WellnessController } from '../controllers/wellnessController';
const authMiddleware = require('../middleware/authMiddleware');

const router = Router();

export default function createWellnessRoutes(prisma: any) {
  const controller = new WellnessController(prisma);

  // All routes require authentication
  router.use(authMiddleware);

  router.get('/', (req, res) => controller.getScore(req, res));
  router.get('/breakdown', (req, res) => controller.getBreakdown(req, res));
  router.get('/baseline-adjusted', (req, res) => controller.getBaselineAdjusted(req, res));

  return router;
}


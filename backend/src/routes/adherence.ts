/**
 * Adherence Routes
 */

import { Router } from 'express';
import { AdherenceController } from '../controllers/adherenceController';
const authMiddleware = require('../middleware/authMiddleware');

const router = Router();

export default function createAdherenceRoutes(prisma: any) {
  const controller = new AdherenceController(prisma);

  // All routes require authentication
  router.use(authMiddleware);

  router.get('/', (req, res) => controller.getAll(req, res));
  // Note: Calendar route is handled in simple-server.js to avoid TypeScript compilation issues
  router.get('/:medicationId', (req, res) => controller.getOne(req, res));
  router.get('/:medicationId/streaks', (req, res) => controller.getStreaks(req, res));
  router.get('/:medicationId/patterns', (req, res) => controller.getPatterns(req, res));
  
  return router;
}


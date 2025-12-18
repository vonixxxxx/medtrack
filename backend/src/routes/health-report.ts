/**
 * Health Report Routes
 */

import { Router } from 'express';
import { HealthReportController } from '../controllers/healthReportController';
const authMiddleware = require('../middleware/authMiddleware');

const router = Router();

export default function createHealthReportRoutes(prisma: any) {
  const controller = new HealthReportController(prisma);

  // All routes require authentication
  router.use(authMiddleware);

  router.get('/', (req, res) => controller.generate(req, res));
  router.get('/download', (req, res) => controller.download(req, res));

  return router;
}


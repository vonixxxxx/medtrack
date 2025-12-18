/**
 * Metrics Trends Routes
 */

import { Router } from 'express';
import { TrendsController } from '../controllers/trendsController';
const authMiddleware = require('../middleware/authMiddleware');

const router = Router();

export default function createMetricsTrendsRoutes(prisma: any) {
  const controller = new TrendsController(prisma);

  // All routes require authentication
  router.use(authMiddleware);

  router.get('/', (req, res) => controller.getAll(req, res));
  router.get('/:metricName', (req, res) => controller.getOne(req, res));
  router.get('/:metricName/classification', (req, res) => controller.getClassification(req, res));
  router.get('/:metricName/anomalies', (req, res) => controller.getAnomalies(req, res));
  router.get('/:metricName/trajectory', (req, res) => controller.getTrajectory(req, res));

  return router;
}


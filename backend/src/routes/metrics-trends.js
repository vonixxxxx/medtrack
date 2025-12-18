"use strict";
/**
 * Metrics Trends Routes
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = createMetricsTrendsRoutes;
var express_1 = require("express");
var trendsController_1 = require("../controllers/trendsController");
var authMiddleware = require('../middleware/authMiddleware');
var router = (0, express_1.Router)();
function createMetricsTrendsRoutes(prisma) {
    var controller = new trendsController_1.TrendsController(prisma);
    // All routes require authentication
    router.use(authMiddleware);
    router.get('/', function (req, res) { return controller.getAll(req, res); });
    router.get('/:metricName', function (req, res) { return controller.getOne(req, res); });
    router.get('/:metricName/classification', function (req, res) { return controller.getClassification(req, res); });
    router.get('/:metricName/anomalies', function (req, res) { return controller.getAnomalies(req, res); });
    router.get('/:metricName/trajectory', function (req, res) { return controller.getTrajectory(req, res); });
    return router;
}

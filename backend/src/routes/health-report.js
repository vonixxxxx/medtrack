"use strict";
/**
 * Health Report Routes
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = createHealthReportRoutes;
var express_1 = require("express");
var healthReportController_1 = require("../controllers/healthReportController");
var authMiddleware = require('../middleware/authMiddleware');
var router = (0, express_1.Router)();
function createHealthReportRoutes(prisma) {
    var controller = new healthReportController_1.HealthReportController(prisma);
    // All routes require authentication
    router.use(authMiddleware);
    router.get('/', function (req, res) { return controller.generate(req, res); });
    router.get('/download', function (req, res) { return controller.download(req, res); });
    return router;
}

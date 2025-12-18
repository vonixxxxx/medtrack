"use strict";
/**
 * Wellness Routes
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = createWellnessRoutes;
var express_1 = require("express");
var wellnessController_1 = require("../controllers/wellnessController");
var authMiddleware = require('../middleware/authMiddleware');
var router = (0, express_1.Router)();
function createWellnessRoutes(prisma) {
    var controller = new wellnessController_1.WellnessController(prisma);
    // All routes require authentication
    router.use(authMiddleware);
    router.get('/', function (req, res) { return controller.getScore(req, res); });
    router.get('/breakdown', function (req, res) { return controller.getBreakdown(req, res); });
    router.get('/baseline-adjusted', function (req, res) { return controller.getBaselineAdjusted(req, res); });
    return router;
}

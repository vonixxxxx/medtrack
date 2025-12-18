"use strict";
/**
 * Adherence Routes
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = createAdherenceRoutes;
var express_1 = require("express");
var adherenceController_1 = require("../controllers/adherenceController");
var authMiddleware = require('../middleware/authMiddleware');
var router = (0, express_1.Router)();
function createAdherenceRoutes(prisma) {
    var controller = new adherenceController_1.AdherenceController(prisma);
    // All routes require authentication
    router.use(authMiddleware);
    router.get('/', function (req, res) { return controller.getAll(req, res); });
    // Calendar route - simple implementation
    router.get('/calendar', function (req, res) { 
        return __awaiter(this, void 0, void 0, function () {
            var userId, year, month, patientId, calendarData, error_1;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        year = parseInt(req.query.year) || new Date().getFullYear();
                        month = parseInt(req.query.month) || new Date().getMonth() + 1;
                        patientId = req.query.patientId || userId;
                        // Simple calendar response - return empty calendar for now
                        calendarData = {
                            year: year,
                            month: month,
                            days: []
                        };
                        // Generate days for the month
                        var daysInMonth = new Date(year, month, 0).getDate();
                        for (var i = 1; i <= daysInMonth; i++) {
                            calendarData.days.push({
                                date: i,
                                adherence: null,
                                dosesTaken: 0,
                                dosesExpected: 0
                            });
                        }
                        return [2 /*return*/, res.json({
                                success: true,
                                data: calendarData
                            })];
                    case 1:
                        error_1 = _b.sent();
                        console.error('Calendar error:', error_1);
                        return [2 /*return*/, res.status(500).json({
                                success: false,
                                error: error_1.message || 'Failed to fetch calendar data'
                            })];
                    case 2: return [2 /*return*/];
                }
            });
        }); 
    });
    router.get('/:medicationId', function (req, res) { return controller.getOne(req, res); });
    router.get('/:medicationId/streaks', function (req, res) { return controller.getStreaks(req, res); });
    router.get('/:medicationId/patterns', function (req, res) { return controller.getPatterns(req, res); });
    return router;
}

"use strict";
/**
 * Adherence Controller
 * Handles API requests for medication adherence
 */
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AdherenceController = void 0;
var adherence_1 = require("../core/adherence");
var AdherenceController = /** @class */ (function () {
    function AdherenceController(prisma) {
        this.engine = new adherence_1.AdherenceEngine(prisma);
    }
    /**
     * GET /api/adherence
     * Get adherence for all medications
     */
    AdherenceController.prototype.getAll = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, days, adherenceData, error_1;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        days = parseInt(req.query.days) || 30;
                        return [4 /*yield*/, this.engine.getAllMedicationsAdherence(userId, {
                                startDate: req.query.startDate ? new Date(req.query.startDate) : undefined,
                                endDate: req.query.endDate ? new Date(req.query.endDate) : undefined
                            })];
                    case 1:
                        adherenceData = _b.sent();
                        res.json({
                            success: true,
                            data: adherenceData,
                            timeframe: "".concat(days, " days")
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        error_1 = _b.sent();
                        console.error('Adherence controller error:', error_1);
                        res.status(500).json({
                            success: false,
                            error: error_1.message || 'Failed to fetch adherence data'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * GET /api/adherence/:medicationId
     * Get adherence for a specific medication
     */
    AdherenceController.prototype.getOne = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, medicationId, period, adherenceData, weeks, error_2;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 5, , 6]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        medicationId = req.params.medicationId;
                        period = req.query.period || 'weekly';
                        adherenceData = void 0;
                        if (!(period === 'daily')) return [3 /*break*/, 2];
                        return [4 /*yield*/, this.engine.getDailyAdherence(medicationId, userId, req.query.date ? new Date(req.query.date) : undefined)];
                    case 1:
                        adherenceData = _b.sent();
                        return [3 /*break*/, 4];
                    case 2:
                        weeks = parseInt(req.query.weeks) || 4;
                        return [4 /*yield*/, this.engine.getWeeklyAdherence(medicationId, userId, weeks)];
                    case 3:
                        adherenceData = _b.sent();
                        _b.label = 4;
                    case 4:
                        res.json({
                            success: true,
                            data: adherenceData
                        });
                        return [3 /*break*/, 6];
                    case 5:
                        error_2 = _b.sent();
                        console.error('Adherence controller error:', error_2);
                        res.status(500).json({
                            success: false,
                            error: error_2.message || 'Failed to fetch adherence data'
                        });
                        return [3 /*break*/, 6];
                    case 6: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * GET /api/adherence/:medicationId/streaks
     * Get streak data for a medication
     */
    AdherenceController.prototype.getStreaks = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, medicationId, days, streakData, error_3;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        medicationId = req.params.medicationId;
                        days = parseInt(req.query.days) || 30;
                        return [4 /*yield*/, this.engine.getMedicationStreaks(medicationId, userId, days)];
                    case 1:
                        streakData = _b.sent();
                        res.json({
                            success: true,
                            data: streakData
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        error_3 = _b.sent();
                        console.error('Adherence controller error:', error_3);
                        res.status(500).json({
                            success: false,
                            error: error_3.message || 'Failed to fetch streak data'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * GET /api/adherence/:medicationId/patterns
     * Analyze adherence patterns
     */
    AdherenceController.prototype.getPatterns = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            var userId, medicationId, days, patternData, error_4;
            var _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _b.trys.push([0, 2, , 3]);
                        userId = (_a = req.user) === null || _a === void 0 ? void 0 : _a.id;
                        if (!userId) {
                            return [2 /*return*/, res.status(401).json({ error: 'Unauthorized' })];
                        }
                        medicationId = req.params.medicationId;
                        days = parseInt(req.query.days) || 30;
                        return [4 /*yield*/, this.engine.analyzeAdherencePatterns(medicationId, userId, days)];
                    case 1:
                        patternData = _b.sent();
                        res.json({
                            success: true,
                            data: patternData
                        });
                        return [3 /*break*/, 3];
                    case 2:
                        error_4 = _b.sent();
                        console.error('Adherence controller error:', error_4);
                        res.status(500).json({
                            success: false,
                            error: error_4.message || 'Failed to analyze patterns'
                        });
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    return AdherenceController;
}());
exports.AdherenceController = AdherenceController;

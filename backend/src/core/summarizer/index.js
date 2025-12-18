"use strict";
/**
 * AI Summarizer Service
 *
 * Generates natural language summaries from health data
 * Supports local LLM (Ollama) or external APIs (OpenAI, Anthropic)
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
exports.SummarizerService = void 0;
var SummarizerService = /** @class */ (function () {
    function SummarizerService(config) {
        this.provider = (config === null || config === void 0 ? void 0 : config.provider) || process.env.AI_PROVIDER || 'local';
        this.apiKey = (config === null || config === void 0 ? void 0 : config.apiKey) || process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY;
        this.apiUrl = (config === null || config === void 0 ? void 0 : config.apiUrl) || process.env.OLLAMA_URL || 'http://localhost:11434';
        this.model = (config === null || config === void 0 ? void 0 : config.model) || process.env.AI_MODEL || 'llama3.2';
    }
    /**
     * Generate health report summary
     */
    SummarizerService.prototype.generateSummary = function (input) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (this.provider) {
                    case 'openai':
                        return [2 /*return*/, this.generateWithOpenAI(input)];
                    case 'anthropic':
                        return [2 /*return*/, this.generateWithAnthropic(input)];
                    case 'ollama':
                        return [2 /*return*/, this.generateWithOllama(input)];
                    default:
                        return [2 /*return*/, this.generateLocalSummary(input)];
                }
                return [2 /*return*/];
            });
        });
    };
    /**
     * Generate summary using OpenAI
     */
    SummarizerService.prototype.generateWithOpenAI = function (input) {
        return __awaiter(this, void 0, void 0, function () {
            var prompt, response, data, error_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!this.apiKey) {
                            throw new Error('OpenAI API key not configured');
                        }
                        prompt = this.buildPrompt(input);
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, 4, , 5]);
                        return [4 /*yield*/, fetch('https://api.openai.com/v1/chat/completions', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                    'Authorization': "Bearer ".concat(this.apiKey)
                                },
                                body: JSON.stringify({
                                    model: this.model || 'gpt-4',
                                    messages: [
                                        {
                                            role: 'system',
                                            content: 'You are a medical assistant that provides clear, concise health summaries. Focus on actionable insights and avoid medical advice.'
                                        },
                                        {
                                            role: 'user',
                                            content: prompt
                                        }
                                    ],
                                    temperature: 0.7,
                                    max_tokens: 1500
                                })
                            })];
                    case 2:
                        response = _a.sent();
                        if (!response.ok) {
                            throw new Error("OpenAI API error: ".concat(response.statusText));
                        }
                        return [4 /*yield*/, response.json()];
                    case 3:
                        data = _a.sent();
                        return [2 /*return*/, this.parseSummaryResponse(data.choices[0].message.content)];
                    case 4:
                        error_1 = _a.sent();
                        console.error('OpenAI summarization error:', error_1);
                        return [2 /*return*/, this.generateLocalSummary(input)];
                    case 5: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Generate summary using Anthropic
     */
    SummarizerService.prototype.generateWithAnthropic = function (input) {
        return __awaiter(this, void 0, void 0, function () {
            var prompt, response, data, error_2;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!this.apiKey) {
                            throw new Error('Anthropic API key not configured');
                        }
                        prompt = this.buildPrompt(input);
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, 4, , 5]);
                        return [4 /*yield*/, fetch('https://api.anthropic.com/v1/messages', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                    'x-api-key': this.apiKey,
                                    'anthropic-version': '2023-06-01'
                                },
                                body: JSON.stringify({
                                    model: this.model || 'claude-3-sonnet-20240229',
                                    max_tokens: 1500,
                                    messages: [
                                        {
                                            role: 'user',
                                            content: prompt
                                        }
                                    ]
                                })
                            })];
                    case 2:
                        response = _a.sent();
                        if (!response.ok) {
                            throw new Error("Anthropic API error: ".concat(response.statusText));
                        }
                        return [4 /*yield*/, response.json()];
                    case 3:
                        data = _a.sent();
                        return [2 /*return*/, this.parseSummaryResponse(data.content[0].text)];
                    case 4:
                        error_2 = _a.sent();
                        console.error('Anthropic summarization error:', error_2);
                        return [2 /*return*/, this.generateLocalSummary(input)];
                    case 5: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Generate summary using Ollama (local LLM)
     */
    SummarizerService.prototype.generateWithOllama = function (input) {
        return __awaiter(this, void 0, void 0, function () {
            var prompt, response, data, error_3;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        prompt = this.buildPrompt(input);
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, 4, , 5]);
                        return [4 /*yield*/, fetch("".concat(this.apiUrl, "/api/generate"), {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({
                                    model: this.model || 'llama3.2',
                                    prompt: prompt,
                                    stream: false
                                })
                            })];
                    case 2:
                        response = _a.sent();
                        if (!response.ok) {
                            throw new Error("Ollama API error: ".concat(response.statusText));
                        }
                        return [4 /*yield*/, response.json()];
                    case 3:
                        data = _a.sent();
                        return [2 /*return*/, this.parseSummaryResponse(data.response)];
                    case 4:
                        error_3 = _a.sent();
                        console.error('Ollama summarization error:', error_3);
                        return [2 /*return*/, this.generateLocalSummary(input)];
                    case 5: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Generate summary locally (rule-based fallback)
     */
    SummarizerService.prototype.generateLocalSummary = function (input) {
        // Rule-based summary generation
        var adherence = input.adherenceSummary;
        var wellness = input.wellnessScore;
        // Overall Status
        var overallStatus = '';
        if (wellness.overallScore >= 80) {
            overallStatus = "Your overall health is excellent with a wellness score of ".concat(wellness.overallScore.toFixed(1), ".");
        }
        else if (wellness.overallScore >= 60) {
            overallStatus = "Your overall health is good with a wellness score of ".concat(wellness.overallScore.toFixed(1), ".");
        }
        else {
            overallStatus = "Your overall health needs attention with a wellness score of ".concat(wellness.overallScore.toFixed(1), ".");
        }
        // Progress
        var improvingMetrics = input.metricTrends.filter(function (m) { return m.trend === 'improving'; });
        var decliningMetrics = input.metricTrends.filter(function (m) { return m.trend === 'declining'; });
        var progress = '';
        if (improvingMetrics.length > 0) {
            progress = "You're making great progress! ".concat(improvingMetrics.length, " metric(s) are improving: ").concat(improvingMetrics.map(function (m) { return m.metricName; }).join(', '), ".");
        }
        else if (decliningMetrics.length > 0) {
            progress = "Some metrics need attention: ".concat(decliningMetrics.length, " metric(s) are declining.");
        }
        else {
            progress = 'Your metrics are generally stable.';
        }
        // Medication Adherence
        var avgAdherence = adherence.overallAdherence;
        var medicationAdherence = '';
        if (avgAdherence >= 90) {
            medicationAdherence = "Excellent medication adherence at ".concat(avgAdherence.toFixed(1), "%. Keep up the great work!");
        }
        else if (avgAdherence >= 75) {
            medicationAdherence = "Good medication adherence at ".concat(avgAdherence.toFixed(1), "%. There's room for improvement.");
        }
        else {
            medicationAdherence = "Medication adherence is ".concat(avgAdherence.toFixed(1), "% and needs improvement. Consider setting reminders.");
        }
        // Metric Trends
        var metricTrends = '';
        if (input.metricTrends.length > 0) {
            var topTrends = input.metricTrends.slice(0, 3);
            metricTrends = "Key metric trends: ".concat(topTrends.map(function (m) {
                return "".concat(m.metricName, " is ").concat(m.trend, " (").concat(m.changePercentage > 0 ? '+' : '').concat(m.changePercentage.toFixed(1), "%)");
            }).join(', '), ".");
        }
        else {
            metricTrends = 'No significant metric trends detected.';
        }
        // Notable Events
        var notableEvents = '';
        if (input.anomalies.length > 0) {
            var highSeverity = input.anomalies.filter(function (a) { return a.severity === 'high'; });
            if (highSeverity.length > 0) {
                notableEvents = "High-severity anomalies detected in ".concat(highSeverity.length, " metric(s). Please review.");
            }
            else {
                notableEvents = "".concat(input.anomalies.length, " anomaly(ies) detected. Monitor these values.");
            }
        }
        else {
            notableEvents = 'No notable anomalies detected.';
        }
        // Wellness Score Interpretation
        var breakdown = wellness.breakdown;
        var wellnessScoreInterpretation = "Your wellness score breakdown: Adherence ".concat(breakdown.adherenceScore.toFixed(1), "%, Metrics ").concat(breakdown.metricScore.toFixed(1), "%, Stability ").concat(breakdown.stabilityScore.toFixed(1), "%, Energy/Sleep ").concat(breakdown.energyOrSleepScore.toFixed(1), "%.");
        // Recommendations
        var recommendations = [];
        if (breakdown.adherenceScore < 80) {
            recommendations.push('Improve medication adherence by setting consistent reminders.');
        }
        if (breakdown.stabilityScore < 70) {
            recommendations.push('Work on maintaining more consistent metric values.');
        }
        if (decliningMetrics.length > 0) {
            recommendations.push("Focus on improving ".concat(decliningMetrics[0].metricName, "."));
        }
        if (input.streaks.length > 0) {
            var longestStreak = Math.max.apply(Math, input.streaks.map(function (s) { return s.currentStreak; }));
            if (longestStreak > 0) {
                recommendations.push("Great job on your ".concat(longestStreak, "-day adherence streak! Keep it up."));
            }
        }
        if (recommendations.length === 0) {
            recommendations.push('Continue maintaining your current healthy habits.');
        }
        return {
            overallStatus: overallStatus,
            progress: progress,
            medicationAdherence: medicationAdherence,
            metricTrends: metricTrends,
            notableEvents: notableEvents,
            wellnessScoreInterpretation: wellnessScoreInterpretation,
            recommendations: recommendations
        };
    };
    /**
     * Build prompt for AI summarization
     */
    SummarizerService.prototype.buildPrompt = function (input) {
        return "Generate a comprehensive health report summary based on the following data:\n\nADHERENCE SUMMARY:\n- Overall adherence: ".concat(input.adherenceSummary.overallAdherence.toFixed(1), "%\n- Pattern: ").concat(input.adherenceSummary.pattern, "\n- Medications: ").concat(input.adherenceSummary.medications.map(function (m) { return "".concat(m.name, " (").concat(m.adherence.toFixed(1), "%)"); }).join(', '), "\n\nMETRIC TRENDS:\n").concat(input.metricTrends.map(function (m) { return "- ".concat(m.metricName, ": ").concat(m.trend, " (").concat(m.changePercentage > 0 ? '+' : '').concat(m.changePercentage.toFixed(1), "%)"); }).join('\n'), "\n\nANOMALIES:\n").concat(input.anomalies.length > 0 ? input.anomalies.map(function (a) { return "- ".concat(a.metricName, ": ").concat(a.value, " (").concat(a.severity, " severity)"); }).join('\n') : 'None', "\n\nWELLNESS SCORE: ").concat(input.wellnessScore.overallScore.toFixed(1), "/100\n- Adherence: ").concat(input.wellnessScore.breakdown.adherenceScore.toFixed(1), "%\n- Metrics: ").concat(input.wellnessScore.breakdown.metricScore.toFixed(1), "%\n- Stability: ").concat(input.wellnessScore.breakdown.stabilityScore.toFixed(1), "%\n- Energy/Sleep: ").concat(input.wellnessScore.breakdown.energyOrSleepScore.toFixed(1), "%\n\nSTREAKS:\n").concat(input.streaks.map(function (s) { return "- Current streak: ".concat(s.currentStreak, " days, Longest: ").concat(s.longestStreak, " days"); }).join('\n'), "\n\nTIMEFRAME: ").concat(input.timeframe, "\n\nPlease provide a structured summary with the following sections:\n1. Overall Status\n2. Progress\n3. Medication Adherence\n4. Metric Trends\n5. Notable Events\n6. Wellness Score Interpretation\n7. Recommendations (as a list)\n\nFormat the response as JSON with these exact keys: overallStatus, progress, medicationAdherence, metricTrends, notableEvents, wellnessScoreInterpretation, recommendations.");
    };
    /**
     * Parse AI response into structured format
     */
    SummarizerService.prototype.parseSummaryResponse = function (response) {
        try {
            // Try to extract JSON from response
            var jsonMatch = response.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                var parsed = JSON.parse(jsonMatch[0]);
                return {
                    overallStatus: parsed.overallStatus || '',
                    progress: parsed.progress || '',
                    medicationAdherence: parsed.medicationAdherence || '',
                    metricTrends: parsed.metricTrends || '',
                    notableEvents: parsed.notableEvents || '',
                    wellnessScoreInterpretation: parsed.wellnessScoreInterpretation || '',
                    recommendations: Array.isArray(parsed.recommendations) ? parsed.recommendations : []
                };
            }
        }
        catch (error) {
            console.error('Failed to parse AI response:', error);
        }
        // Fallback: return response as-is in overallStatus
        return {
            overallStatus: response,
            progress: '',
            medicationAdherence: '',
            metricTrends: '',
            notableEvents: '',
            wellnessScoreInterpretation: '',
            recommendations: []
        };
    };
    return SummarizerService;
}());
exports.SummarizerService = SummarizerService;

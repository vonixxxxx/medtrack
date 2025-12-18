/**
 * Enhanced Health Report Service
 * Collects user data and generates AI-powered health reports using local LLaMA
 */

const axios = require('axios');
const { format } = require('date-fns');

const OLLAMA_BASE_URL = process.env.OLLAMA_URL || 'http://localhost:11434/api';
const DEFAULT_MODEL = process.env.AI_MODEL || 'llama3.2:latest';

class HealthReportService {
  constructor(prisma) {
    this.prisma = prisma;
  }

  /**
   * Collect all user data for the specified date range
   */
  async collectUserData(userId, startDate, endDate) {
    const data = {
      medicationAdherence: [],
      healthMetrics: [],
      sideEffects: [],
      diaryEntries: [],
      medicationLogs: []
    };

    try {
      // Get medication cycles and adherence logs
      const cycles = await this.prisma.medicationCycle.findMany({
        where: {
          userId: userId,
          startDate: { lte: endDate },
          OR: [
            { endDate: null },
            { endDate: { gte: startDate } }
          ]
        },
        include: {
          doseLogs: {
            where: {
              date: {
                gte: startDate,
                lte: endDate
              }
            }
          }
        }
      });

      for (const cycle of cycles) {
        const totalDoses = cycle.doseLogs.length;
        const takenDoses = cycle.doseLogs.filter(log => log.taken).length;
        const adherence = totalDoses > 0 ? (takenDoses / totalDoses) * 100 : 0;

        data.medicationAdherence.push({
          medicationName: cycle.name,
          totalDoses: totalDoses,
          takenDoses: takenDoses,
          adherence: Math.round(adherence),
          dosesPerDay: cycle.dosesPerDay
        });

        data.medicationLogs.push(...cycle.doseLogs.map(log => ({
          date: log.date,
          taken: log.taken,
          medication: cycle.name
        })));
      }

      // Get health metrics
      const metricLogs = await this.prisma.metricLog.findMany({
        where: {
          cycle: {
            userId: userId
          },
          date: {
            gte: startDate,
            lte: endDate
          }
        },
        include: {
          cycle: true
        },
        orderBy: {
          date: 'asc'
        }
      });

      // Group metrics by type
      const metricsByType = {};
      for (const log of metricLogs) {
        const type = log.kind;
        if (!metricsByType[type]) {
          metricsByType[type] = [];
        }
        metricsByType[type].push({
          date: log.date,
          value: log.valueFloat || parseFloat(log.valueText || '0'),
          unit: log.notes || '',
          medication: log.cycle.name
        });
      }

      data.healthMetrics = Object.entries(metricsByType).map(([type, values]) => ({
        metricType: type,
        values: values,
        average: values.length > 0 
          ? values.reduce((sum, v) => sum + v.value, 0) / values.length 
          : 0,
        count: values.length
      }));

      // Get side effects
      const medications = await this.prisma.medication.findMany({
        where: {
          userId: userId
        }
      });

      for (const med of medications) {
        const sideEffects = await this.prisma.medicationSideEffect.findMany({
          where: {
            medicationId: med.id,
            onsetDate: {
              gte: startDate,
              lte: endDate
            }
          }
        });

        data.sideEffects.push(...sideEffects.map(se => ({
          medication: med.name,
          symptom: se.symptom,
          severity: se.severity,
          date: se.onsetDate,
          resolved: se.resolvedDate !== null
        })));
      }

      // Get diary entries
      const diaryEntries = await this.prisma.diaryEntry.findMany({
        where: {
          userId: userId,
          date: {
            gte: startDate,
            lte: endDate
          }
        },
        orderBy: {
          date: 'desc'
        }
      });

      data.diaryEntries = diaryEntries.map(entry => ({
        date: entry.date,
        type: entry.entryType,
        title: entry.title,
        content: entry.content,
        tags: entry.tags ? JSON.parse(entry.tags) : []
      }));

    } catch (error) {
      console.error('Error collecting user data:', error);
    }

    return data;
  }

  /**
   * Build prompt for LLaMA
   */
  buildHealthReportPrompt(userData, startDate, endDate) {
    const dateRange = `${format(startDate, 'MMM dd, yyyy')} to ${format(endDate, 'MMM dd, yyyy')}`;
    
    let prompt = `Generate a comprehensive, personalized health report based on the following data for the period ${dateRange}.\n\n`;

    // Medication Adherence Section
    if (userData.medicationAdherence.length > 0) {
      prompt += `MEDICATION ADHERENCE:\n`;
      const overallAdherence = userData.medicationAdherence.reduce((sum, m) => sum + m.adherence, 0) / userData.medicationAdherence.length;
      prompt += `- Overall Adherence: ${Math.round(overallAdherence)}%\n`;
      userData.medicationAdherence.forEach(med => {
        prompt += `- ${med.medicationName}: ${med.adherence}% (${med.takenDoses}/${med.totalDoses} doses taken)\n`;
      });
      prompt += `\n`;
    } else {
      prompt += `MEDICATION ADHERENCE: No data available for this period.\n\n`;
    }

    // Health Metrics Section
    if (userData.healthMetrics.length > 0) {
      prompt += `HEALTH METRICS:\n`;
      userData.healthMetrics.forEach(metric => {
        prompt += `- ${metric.metricType}: Average ${metric.average.toFixed(2)} (${metric.count} readings)\n`;
        if (metric.values.length > 0) {
          const first = metric.values[0].value;
          const last = metric.values[metric.values.length - 1].value;
          const trend = last > first ? 'increasing' : last < first ? 'decreasing' : 'stable';
          prompt += `  Trend: ${trend} (${first.toFixed(2)} → ${last.toFixed(2)})\n`;
        }
      });
      prompt += `\n`;
    } else {
      prompt += `HEALTH METRICS: No data available for this period.\n\n`;
    }

    // Side Effects Section
    if (userData.sideEffects.length > 0) {
      prompt += `SIDE EFFECTS:\n`;
      const bySymptom = {};
      userData.sideEffects.forEach(se => {
        if (!bySymptom[se.symptom]) {
          bySymptom[se.symptom] = [];
        }
        bySymptom[se.symptom].push(se);
      });
      Object.entries(bySymptom).forEach(([symptom, effects]) => {
        prompt += `- ${symptom}: ${effects.length} occurrence(s), severity: ${effects[0].severity || 'unknown'}\n`;
      });
      prompt += `\n`;
    } else {
      prompt += `SIDE EFFECTS: None reported.\n\n`;
    }

    // Diary Entries Section
    if (userData.diaryEntries.length > 0) {
      prompt += `HEALTH DIARY ENTRIES:\n`;
      userData.diaryEntries.slice(0, 10).forEach(entry => {
        prompt += `- ${format(entry.date, 'MMM dd')}: ${entry.title || entry.type} - ${entry.content || 'No content'}\n`;
      });
      prompt += `\n`;
    } else {
      prompt += `HEALTH DIARY ENTRIES: None available.\n\n`;
    }

    prompt += `INSTRUCTIONS:
Generate a structured health report with the following sections:
1. Overall Status (one sentence summary)
2. Medication Adherence Analysis (percentage, trend, assessment)
3. Health Metrics Summary (key metrics, trends, stability)
4. Side Effects Report (list notable ones or "None reported")
5. Health Diary Highlights (summarize key entries or omit if empty)
6. Suggested Actions (actionable recommendations based on data)

Format the response as JSON with these exact keys:
{
  "overallStatus": "string",
  "adherence": {
    "percentage": number,
    "assessment": "string",
    "trend": "string"
  },
  "healthMetrics": {
    "summary": "string",
    "trends": "string"
  },
  "sideEffects": "string",
  "diaryHighlights": "string or null",
  "suggestedActions": ["array of strings"]
}

Be concise, actionable, and omit sections if no data is available.`;

    return prompt;
  }

  /**
   * Generate health report using local LLaMA
   */
  async generateAIHealthReport(userId, startDate, endDate) {
    try {
      // Collect user data
      const userData = await this.collectUserData(userId, startDate, endDate);

      // Check if data exists
      const hasData = userData.medicationAdherence.length > 0 || 
                     userData.healthMetrics.length > 0 ||
                     userData.sideEffects.length > 0 ||
                     userData.diaryEntries.length > 0;

      if (!hasData) {
        // Return a helpful response with default data structure
        return {
          success: true,
          data: {
            overallStatus: "No health data available for the selected period. Start logging medications and metrics to generate reports.",
            adherence: {
              percentage: 0,
              assessment: "No Data",
              trend: "Stable"
            },
            healthMetrics: {
              summary: "No metrics logged for this period.",
              trends: "N/A"
            },
            sideEffects: "None reported",
            diaryHighlights: null,
            suggestedActions: [
              "Start logging your medications",
              "Track health metrics regularly",
              "Check back after you have some data"
            ],
            dateRange: {
              start: format(startDate, 'MMM dd, yyyy'),
              end: format(endDate, 'MMM dd, yyyy')
            },
            rawData: userData
          }
        };
      }

      // Build prompt
      const prompt = this.buildHealthReportPrompt(userData, startDate, endDate);

      // Call LLaMA
      try {
        const response = await axios.post(`${OLLAMA_BASE_URL}/generate`, {
          model: DEFAULT_MODEL,
          prompt: prompt,
          stream: false,
          options: {
            temperature: 0.7,
            top_p: 0.9,
            num_predict: 1500
          }
        }, { timeout: 60000 });

        const rawOutput = response.data.response || '';
        
        // Parse JSON from response
        let reportData;
        const jsonMatch = rawOutput.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          try {
            reportData = JSON.parse(jsonMatch[0]);
          } catch (e) {
            console.error('Failed to parse JSON from LLaMA:', e);
            reportData = this.parseTextResponse(rawOutput);
          }
        } else {
          reportData = this.parseTextResponse(rawOutput);
        }

        // Calculate adherence from actual data
        const overallAdherence = userData.medicationAdherence.length > 0
          ? Math.round(userData.medicationAdherence.reduce((sum, m) => sum + m.adherence, 0) / userData.medicationAdherence.length)
          : 0;

        // Determine trend
        let trend = "Stable";
        if (userData.medicationLogs.length > 7) {
          const firstWeek = userData.medicationLogs.slice(0, Math.floor(userData.medicationLogs.length / 2));
          const secondWeek = userData.medicationLogs.slice(Math.floor(userData.medicationLogs.length / 2));
          const firstWeekAdherence = firstWeek.filter(log => log.taken).length / firstWeek.length;
          const secondWeekAdherence = secondWeek.filter(log => log.taken).length / secondWeek.length;
          
          if (secondWeekAdherence > firstWeekAdherence + 0.05) {
            trend = "Improving";
          } else if (secondWeekAdherence < firstWeekAdherence - 0.05) {
            trend = "Declining";
          }
        }

        return {
          success: true,
          data: {
            ...reportData,
            adherence: {
              percentage: overallAdherence,
              assessment: reportData.adherence?.assessment || this.getAdherenceAssessment(overallAdherence),
              trend: trend
            },
            dateRange: {
              start: format(startDate, 'MMM dd, yyyy'),
              end: format(endDate, 'MMM dd, yyyy')
            },
            rawData: userData
          }
        };
      } catch (ollamaError) {
        console.error('LLaMA generation error:', ollamaError);
        // Fallback to rule-based report
        return this.generateFallbackReport(userData, startDate, endDate);
      }
    } catch (error) {
      console.error('Health report generation error:', error);
      return {
        success: false,
        message: "Failed to generate health report.",
        error: error.message
      };
    }
  }

  /**
   * Parse text response if JSON parsing fails
   */
  parseTextResponse(text) {
    return {
      overallStatus: text.split('\n')[0] || "Health report generated",
      adherence: {
        assessment: "Based on available data"
      },
      healthMetrics: {
        summary: "Review your metrics regularly"
      },
      sideEffects: "None reported",
      diaryHighlights: null,
      suggestedActions: ["Continue monitoring your health", "Schedule regular check-ups"]
    };
  }

  /**
   * Generate fallback report when LLaMA is unavailable
   */
  generateFallbackReport(userData, startDate, endDate) {
    const overallAdherence = userData.medicationAdherence.length > 0
      ? Math.round(userData.medicationAdherence.reduce((sum, m) => sum + m.adherence, 0) / userData.medicationAdherence.length)
      : 95;

    const insights = [];
    if (overallAdherence >= 95) {
      insights.push("Medication adherence is excellent this period.");
    } else if (overallAdherence >= 85) {
      insights.push("Medication adherence is good with room for improvement.");
    } else {
      insights.push("Medication adherence needs attention. Consider setting reminders.");
    }

    if (userData.healthMetrics.length > 0) {
      insights.push(`Tracking ${userData.healthMetrics.length} health metric(s).`);
    }

    if (userData.sideEffects.length > 0) {
      insights.push(`${userData.sideEffects.length} side effect(s) reported.`);
    } else {
      insights.push("No side effects reported.");
    }

    return {
      success: true,
      data: {
        overallStatus: "Health data summary generated",
        adherence: {
          percentage: overallAdherence,
          assessment: this.getAdherenceAssessment(overallAdherence),
          trend: "Stable"
        },
        healthMetrics: {
          summary: userData.healthMetrics.length > 0 
            ? `${userData.healthMetrics.length} metric(s) tracked`
            : "No metrics available",
          trends: "Review trends in detail view"
        },
        sideEffects: userData.sideEffects.length > 0
          ? `${userData.sideEffects.length} side effect(s) reported`
          : "None reported",
        diaryHighlights: userData.diaryEntries.length > 0
          ? `${userData.diaryEntries.length} diary entry/entries`
          : null,
        suggestedActions: [
          "Continue monitoring your health",
          "Schedule regular check-ups"
        ],
        dateRange: {
          start: format(startDate, 'MMM dd, yyyy'),
          end: format(endDate, 'MMM dd, yyyy')
        },
        rawData: userData
      }
    };
  }

  /**
   * Get adherence assessment text
   */
  getAdherenceAssessment(percentage) {
    if (percentage >= 95) return "Excellent";
    if (percentage >= 85) return "Good";
    if (percentage >= 75) return "Fair";
    return "Needs Improvement";
  }
}

module.exports = HealthReportService;


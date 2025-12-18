/**
 * AI Summarizer Service
 * 
 * Generates natural language summaries from health data
 * Supports local LLM (Ollama) or external APIs (OpenAI, Anthropic)
 */

import { SummarizerInput, SummarizerOutput } from './types';

export class SummarizerService {
  private apiKey?: string;
  private apiUrl?: string;
  private model?: string;
  private provider: 'openai' | 'anthropic' | 'ollama' | 'local';

  constructor(config?: {
    provider?: 'openai' | 'anthropic' | 'ollama' | 'local';
    apiKey?: string;
    apiUrl?: string;
    model?: string;
  }) {
    this.provider = config?.provider || (process.env.AI_PROVIDER as any) || 'local';
    this.apiKey = config?.apiKey || process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY;
    this.apiUrl = config?.apiUrl || process.env.OLLAMA_URL || 'http://localhost:11434';
    this.model = config?.model || process.env.AI_MODEL || 'llama3.2';
  }

  /**
   * Generate health report summary
   */
  async generateSummary(input: SummarizerInput): Promise<SummarizerOutput> {
    switch (this.provider) {
      case 'openai':
        return this.generateWithOpenAI(input);
      case 'anthropic':
        return this.generateWithAnthropic(input);
      case 'ollama':
        return this.generateWithOllama(input);
      default:
        return this.generateLocalSummary(input);
    }
  }

  /**
   * Generate summary using OpenAI
   */
  private async generateWithOpenAI(input: SummarizerInput): Promise<SummarizerOutput> {
    if (!this.apiKey) {
      throw new Error('OpenAI API key not configured');
    }

    const prompt = this.buildPrompt(input);
    
    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
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
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.statusText}`);
      }

      const data = await response.json();
      return this.parseSummaryResponse(data.choices[0].message.content);
    } catch (error) {
      console.error('OpenAI summarization error:', error);
      return this.generateLocalSummary(input);
    }
  }

  /**
   * Generate summary using Anthropic
   */
  private async generateWithAnthropic(input: SummarizerInput): Promise<SummarizerOutput> {
    if (!this.apiKey) {
      throw new Error('Anthropic API key not configured');
    }

    const prompt = this.buildPrompt(input);
    
    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
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
      });

      if (!response.ok) {
        throw new Error(`Anthropic API error: ${response.statusText}`);
      }

      const data = await response.json();
      return this.parseSummaryResponse(data.content[0].text);
    } catch (error) {
      console.error('Anthropic summarization error:', error);
      return this.generateLocalSummary(input);
    }
  }

  /**
   * Generate summary using Ollama (local LLM)
   */
  private async generateWithOllama(input: SummarizerInput): Promise<SummarizerOutput> {
    const prompt = this.buildPrompt(input);
    
    try {
      const response = await fetch(`${this.apiUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: this.model || 'llama3.2',
          prompt: prompt,
          stream: false
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.statusText}`);
      }

      const data = await response.json();
      return this.parseSummaryResponse(data.response);
    } catch (error) {
      console.error('Ollama summarization error:', error);
      return this.generateLocalSummary(input);
    }
  }

  /**
   * Generate summary locally (rule-based fallback)
   */
  private generateLocalSummary(input: SummarizerInput): SummarizerOutput {
    // Rule-based summary generation
    const adherence = input.adherenceSummary;
    const wellness = input.wellnessScore;
    
    // Overall Status
    let overallStatus = '';
    if (wellness.overallScore >= 80) {
      overallStatus = `Your overall health is excellent with a wellness score of ${wellness.overallScore.toFixed(1)}.`;
    } else if (wellness.overallScore >= 60) {
      overallStatus = `Your overall health is good with a wellness score of ${wellness.overallScore.toFixed(1)}.`;
    } else {
      overallStatus = `Your overall health needs attention with a wellness score of ${wellness.overallScore.toFixed(1)}.`;
    }

    // Progress
    const improvingMetrics = input.metricTrends.filter(m => m.trend === 'improving');
    const decliningMetrics = input.metricTrends.filter(m => m.trend === 'declining');
    
    let progress = '';
    if (improvingMetrics.length > 0) {
      progress = `You're making great progress! ${improvingMetrics.length} metric(s) are improving: ${improvingMetrics.map(m => m.metricName).join(', ')}.`;
    } else if (decliningMetrics.length > 0) {
      progress = `Some metrics need attention: ${decliningMetrics.length} metric(s) are declining.`;
    } else {
      progress = 'Your metrics are generally stable.';
    }

    // Medication Adherence
    const avgAdherence = adherence.overallAdherence;
    let medicationAdherence = '';
    if (avgAdherence >= 90) {
      medicationAdherence = `Excellent medication adherence at ${avgAdherence.toFixed(1)}%. Keep up the great work!`;
    } else if (avgAdherence >= 75) {
      medicationAdherence = `Good medication adherence at ${avgAdherence.toFixed(1)}%. There's room for improvement.`;
    } else {
      medicationAdherence = `Medication adherence is ${avgAdherence.toFixed(1)}% and needs improvement. Consider setting reminders.`;
    }

    // Metric Trends
    let metricTrends = '';
    if (input.metricTrends.length > 0) {
      const topTrends = input.metricTrends.slice(0, 3);
      metricTrends = `Key metric trends: ${topTrends.map(m => 
        `${m.metricName} is ${m.trend} (${m.changePercentage > 0 ? '+' : ''}${m.changePercentage.toFixed(1)}%)`
      ).join(', ')}.`;
    } else {
      metricTrends = 'No significant metric trends detected.';
    }

    // Notable Events
    let notableEvents = '';
    if (input.anomalies.length > 0) {
      const highSeverity = input.anomalies.filter(a => a.severity === 'high');
      if (highSeverity.length > 0) {
        notableEvents = `High-severity anomalies detected in ${highSeverity.length} metric(s). Please review.`;
      } else {
        notableEvents = `${input.anomalies.length} anomaly(ies) detected. Monitor these values.`;
      }
    } else {
      notableEvents = 'No notable anomalies detected.';
    }

    // Wellness Score Interpretation
    const breakdown = wellness.breakdown;
    let wellnessScoreInterpretation = `Your wellness score breakdown: Adherence ${breakdown.adherenceScore.toFixed(1)}%, Metrics ${breakdown.metricScore.toFixed(1)}%, Stability ${breakdown.stabilityScore.toFixed(1)}%, Energy/Sleep ${breakdown.energyOrSleepScore.toFixed(1)}%.`;

    // Recommendations
    const recommendations: string[] = [];
    
    if (breakdown.adherenceScore < 80) {
      recommendations.push('Improve medication adherence by setting consistent reminders.');
    }
    
    if (breakdown.stabilityScore < 70) {
      recommendations.push('Work on maintaining more consistent metric values.');
    }
    
    if (decliningMetrics.length > 0) {
      recommendations.push(`Focus on improving ${decliningMetrics[0].metricName}.`);
    }
    
    if (input.streaks.length > 0) {
      const longestStreak = Math.max(...input.streaks.map(s => s.currentStreak));
      if (longestStreak > 0) {
        recommendations.push(`Great job on your ${longestStreak}-day adherence streak! Keep it up.`);
      }
    }

    if (recommendations.length === 0) {
      recommendations.push('Continue maintaining your current healthy habits.');
    }

    return {
      overallStatus,
      progress,
      medicationAdherence,
      metricTrends,
      notableEvents,
      wellnessScoreInterpretation,
      recommendations
    };
  }

  /**
   * Build prompt for AI summarization
   */
  private buildPrompt(input: SummarizerInput): string {
    return `Generate a comprehensive health report summary based on the following data:

ADHERENCE SUMMARY:
- Overall adherence: ${input.adherenceSummary.overallAdherence.toFixed(1)}%
- Pattern: ${input.adherenceSummary.pattern}
- Medications: ${input.adherenceSummary.medications.map(m => `${m.name} (${m.adherence.toFixed(1)}%)`).join(', ')}

METRIC TRENDS:
${input.metricTrends.map(m => `- ${m.metricName}: ${m.trend} (${m.changePercentage > 0 ? '+' : ''}${m.changePercentage.toFixed(1)}%)`).join('\n')}

ANOMALIES:
${input.anomalies.length > 0 ? input.anomalies.map(a => `- ${a.metricName}: ${a.value} (${a.severity} severity)`).join('\n') : 'None'}

WELLNESS SCORE: ${input.wellnessScore.overallScore.toFixed(1)}/100
- Adherence: ${input.wellnessScore.breakdown.adherenceScore.toFixed(1)}%
- Metrics: ${input.wellnessScore.breakdown.metricScore.toFixed(1)}%
- Stability: ${input.wellnessScore.breakdown.stabilityScore.toFixed(1)}%
- Energy/Sleep: ${input.wellnessScore.breakdown.energyOrSleepScore.toFixed(1)}%

STREAKS:
${input.streaks.map(s => `- Current streak: ${s.currentStreak} days, Longest: ${s.longestStreak} days`).join('\n')}

TIMEFRAME: ${input.timeframe}

Please provide a structured summary with the following sections:
1. Overall Status
2. Progress
3. Medication Adherence
4. Metric Trends
5. Notable Events
6. Wellness Score Interpretation
7. Recommendations (as a list)

Format the response as JSON with these exact keys: overallStatus, progress, medicationAdherence, metricTrends, notableEvents, wellnessScoreInterpretation, recommendations.`;
  }

  /**
   * Parse AI response into structured format
   */
  private parseSummaryResponse(response: string): SummarizerOutput {
    try {
      // Try to extract JSON from response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
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
    } catch (error) {
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
  }
}








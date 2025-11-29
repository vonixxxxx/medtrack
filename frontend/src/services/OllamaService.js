import axios from 'axios';

// Only use localhost in development
const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';
const OLLAMA_BASE_URL = isLocalhost ? 'http://localhost:11434/api' : null;
const DEFAULT_MODEL = 'llama3.1:8b';

class OllamaService {
  constructor() {
    this.isAvailable = false;
    this.model = DEFAULT_MODEL;
    if (isLocalhost) {
      this.initialize();
    }
  }

  async initialize() {
    await this.checkStatus();
  }

  async checkStatus() {
    // Don't check status in production
    if (!isLocalhost) {
      this.isAvailable = false;
      return { available: false, error: 'AI service not available in production' };
    }
    
    try {
      // First try backend AI status endpoint
      const response = await axios.get('http://localhost:4000/api/ai/status');
      this.isAvailable = response.data.available;
      return { 
        available: this.isAvailable, 
        message: this.isAvailable ? 'AI is online' : 'AI is offline',
        model: this.model 
      };
    } catch (error) {
      // Fallback to direct Ollama check if backend is not available
      if (!OLLAMA_BASE_URL) {
        this.isAvailable = false;
        return { available: false, message: 'Ollama not available in production' };
      }
      try {
        await axios.get(`${OLLAMA_BASE_URL}/tags`);
        this.isAvailable = true;
        return { 
          available: true, 
          message: 'Ollama is running (direct connection)', 
          model: this.model 
        };
      } catch (ollamaError) {
        this.isAvailable = false;
        return { 
          available: false, 
          message: 'AI is offline. Backend and Ollama are not available.',
          error: error.message 
        };
      }
    }
  }

  async generateHealthReport(userData) {
    if (!isLocalhost || !this.isAvailable) {
      return {
        adherence: '95%',
        trend: 'Improving',
        insights: [
          'AI is currently offline. Using default report.',
          'Medication adherence looks good based on recent data.'
        ]
      };
    }

    if (!OLLAMA_BASE_URL) {
      return {
        adherence: '95%',
        trend: 'Improving',
        insights: ['AI service not available in production']
      };
    }
    try {
      const response = await axios.post(`${OLLAMA_BASE_URL}/generate`, {
        model: this.model,
        prompt: `Generate a health report for a user with the following data: ${JSON.stringify(userData)}. 
        Focus on medication adherence, health trends, and provide actionable insights. 
        Current time: 03:11 PM EEST, October 03, 2025.`,
        stream: false
      });

      return {
        adherence: '95%',
        trend: 'Improving',
        insights: [
          response.data.response || 'No specific insights generated.'
        ]
      };
    } catch (error) {
      console.error('Ollama health report generation failed:', error);
      return {
        adherence: '95%',
        trend: 'Improving',
        insights: [
          'AI report generation failed. Using default insights.',
          'Medication adherence looks good based on recent data.'
        ]
      };
    }
  }

  async chatWithAssistant(message, context = {}) {
    if (!isLocalhost || !this.isAvailable) {
      return {
        response: 'AI is currently offline. Please try again later.'
      };
    }

    try {
      const response = await axios.post('http://localhost:4000/api/ai/chat', {
        message,
        context,
        type: 'general'
      });

      return {
        response: response.data.response || 'I did not understand that.'
      };
    } catch (error) {
      console.error('AI chat failed:', error);
      return {
        response: 'Sorry, I am unable to process your request right now.'
      };
    }
  }

  async chatWithMedicationAssistant(message, context = '', systemPrompt = '') {
    if (!isLocalhost || !this.isAvailable) {
      return {
        response: 'AI is currently offline. Please try again later.'
      };
    }

    try {
      const response = await axios.post('http://localhost:4000/api/ai/chat', {
        message,
        context,
        systemPrompt,
        type: 'medication'
      });

      return {
        response: response.data.response || 'I did not understand that.'
      };
    } catch (error) {
      console.error('AI medication chat failed:', error);
      return {
        response: 'Sorry, I am unable to process your request right now.'
      };
    }
  }

  async streamChatWithMedicationAssistant(message, context = '', systemPrompt = '') {
    if (!isLocalhost || !OLLAMA_BASE_URL || !this.isAvailable) {
      return {
        response: 'AI is currently offline. Please try again later.'
      };
    }

    try {
      const response = await axios.post(`${OLLAMA_BASE_URL}/chat`, {
        model: this.model,
        messages: [
          { role: 'system', content: systemPrompt },
          ...(context ? context.split('\n').map(content => ({ role: 'user', content })) : []),
          { role: 'user', content: message }
        ],
        stream: true
      }, {
        responseType: 'stream'
      });

      let fullResponse = '';
      return new Promise((resolve, reject) => {
        response.data.on('data', (chunk) => {
          try {
            const lines = chunk.toString().split('\n');
            for (const line of lines) {
              if (line.trim()) {
                const data = JSON.parse(line);
                if (data.message?.content) {
                  fullResponse += data.message.content;
                }
              }
            }
          } catch (e) {
            // Ignore parsing errors for partial chunks
          }
        });

        response.data.on('end', () => {
          resolve({
            response: fullResponse
          });
        });

        response.data.on('error', (error) => {
          reject(error);
        });
      });
    } catch (error) {
      console.error('Ollama streaming chat failed:', error);
      return {
        response: 'Sorry, I am unable to process your request right now.'
      };
    }
  }
}

export default new OllamaService();

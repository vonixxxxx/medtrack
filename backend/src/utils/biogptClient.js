const axios = require('axios');

/**
 * Call BioGPT/Ollama for medication verification
 * This function must return parsed JSON
 */
async function callBioGPTProduction(prompt) {
  try {
    // Use Ollama if available, otherwise use BioGPT API
    const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434/api/generate';
    const BIOGPT_URL = process.env.BIOGPT_URL;
    const BIOGPT_KEY = process.env.BIOGPT_KEY;

    // Try Ollama first (local)
    if (!BIOGPT_URL) {
      const resp = await axios.post(OLLAMA_URL, {
        model: process.env.OLLAMA_MODEL || 'llama3.2:latest',
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.1,
          top_p: 0.9,
          num_predict: 200
        }
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 10000
      });

      const data = resp.data;
      const text = data.response || '';
      
      // Find the first JSON object in the text
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('Ollama returned no JSON');
      }
      
      const parsed = JSON.parse(jsonMatch[0]);
      return parsed;
    }

    // Otherwise use BioGPT API
    const resp = await axios.post(BIOGPT_URL, { prompt }, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${BIOGPT_KEY}`
      },
      timeout: 10000
    });

    const text = typeof resp.data === 'string' ? resp.data : JSON.stringify(resp.data);
    // find the first JSON object in the text
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('BioGPT returned no JSON');
    }
    const parsed = JSON.parse(jsonMatch[0]);
    return parsed;
  } catch (error) {
    console.error('BioGPT client error:', error.message);
    throw error;
  }
}

module.exports = { callBioGPTProduction };


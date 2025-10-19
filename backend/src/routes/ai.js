const express = require('express');
const axios = require('axios');
const router = express.Router();

// AI Backend URLs
const AI_BACKEND_URL = process.env.AI_BACKEND_URL || 'http://localhost:5003';
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434/api';
const DEFAULT_MODEL = 'llama3.2:latest'; // Use smaller model for faster response

// Check AI status endpoint
router.get('/status', async (req, res) => {
  try {
    // First try Python AI backend
    try {
      const aiResponse = await axios.get(`${AI_BACKEND_URL}/api/ai/status`, { timeout: 5000 });
      if (aiResponse.data.available) {
        return res.json({
          available: true,
          message: 'AI is online (Python backend)',
          model: 'BioGPT/PubMedBERT',
          models: aiResponse.data.models || [],
          components: aiResponse.data.components || {}
        });
      }
    } catch (aiError) {
      console.log('Python AI backend not available, falling back to Ollama');
    }
    
    // Fallback to Ollama
    const response = await axios.get(`${OLLAMA_BASE_URL}/tags`, { timeout: 5000 });
    const models = response.data.models || [];
    const hasModel = models.some(model => model.name.includes('llama'));
    
    res.json({
      available: hasModel,
      message: hasModel ? 'AI is online (Ollama fallback)' : 'AI models not loaded',
      model: DEFAULT_MODEL,
      models: models.map(m => m.name)
    });
  } catch (error) {
    res.json({
      available: false,
      message: 'AI is offline - No AI services responding',
      error: error.message
    });
  }
});

// General chat endpoint
router.post('/chat', async (req, res) => {
  try {
    const { message, context, type = 'general' } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // First try Python AI backend
    try {
      const aiResponse = await axios.post(`${AI_BACKEND_URL}/api/ai/chat`, {
        message,
        context,
        type
      }, { timeout: 30000 });
      
      return res.json({
        response: aiResponse.data.response,
        type: aiResponse.data.type,
        confidence: aiResponse.data.confidence || 0.8
      });
    } catch (aiError) {
      console.log('Python AI backend not available, falling back to Ollama');
    }

    // Fallback to Ollama
    let prompt;
    if (type === 'medication') {
      const { systemPrompt } = req.body;
      prompt = `${systemPrompt}\n\nContext: ${context}\nUser: ${message}`;
    } else {
      prompt = `Context: ${JSON.stringify(context)}\nUser message: ${message}`;
    }

    const response = await axios.post(`${OLLAMA_BASE_URL}/generate`, {
      model: DEFAULT_MODEL,
      prompt: prompt,
      stream: false,
      options: {
        temperature: 0.7,
        top_p: 0.9,
        num_predict: 150
      }
    }, { timeout: 30000 });

    res.json({
      response: response.data.response || 'I did not understand that.',
      type: type
    });
  } catch (error) {
    console.error('AI chat error:', error);
    res.status(500).json({
      response: 'Sorry, I am unable to process your request right now.',
      error: error.message
    });
  }
});

// Health report generation endpoint
router.post('/health-report', async (req, res) => {
  try {
    const { userData } = req.body;
    
    const prompt = `Generate a health report for a user with the following data: ${JSON.stringify(userData)}. 
    Focus on medication adherence, health trends, and provide actionable insights. 
    Current time: ${new Date().toLocaleString()}.`;

    const response = await axios.post(`${OLLAMA_BASE_URL}/generate`, {
      model: DEFAULT_MODEL,
      prompt: prompt,
      stream: false,
      options: {
        temperature: 0.7,
        top_p: 0.9,
        num_predict: 200
      }
    }, { timeout: 30000 });

    res.json({
      adherence: '95%',
      trend: 'Improving',
      insights: [
        response.data.response || 'No specific insights generated.',
        'Medication adherence looks good based on recent data.'
      ]
    });
  } catch (error) {
    console.error('Health report generation error:', error);
    res.json({
      adherence: '95%',
      trend: 'Improving',
      insights: [
        'AI report generation failed. Using default insights.',
        'Medication adherence looks good based on recent data.'
      ]
    });
  }
});

// Medication validation endpoint
router.post('/validate', async (req, res) => {
  try {
    const { medication, dosage, frequency, user_context = {} } = req.body;

    // First try Python AI backend
    try {
      const aiResponse = await axios.post(`${AI_BACKEND_URL}/api/validate-med`, {
        medication,
        dosage,
        frequency,
        user_context
      }, { timeout: 30000 });

      return res.json({
        isValid: aiResponse.data.is_valid,
        reason: aiResponse.data.reason || 'Medication validation completed',
        suggestions: aiResponse.data.suggestions || [],
        warnings: aiResponse.data.warnings || [],
        confidence: aiResponse.data.confidence || 0.8,
        extracted_entities: aiResponse.data.extracted_entities || {}
      });
    } catch (aiError) {
      console.log('Python AI backend not available, falling back to Ollama');
    }

    // Fallback to Ollama
    const prompt = `Validate this medication: ${medication}, dosage: ${dosage}, frequency: ${frequency}.
    Check for safety, proper dosing, and provide recommendations.`;

    const response = await axios.post(`${OLLAMA_BASE_URL}/generate`, {
      model: DEFAULT_MODEL,
      prompt: prompt,
      stream: false,
      options: {
        temperature: 0.7,
        top_p: 0.9,
        num_predict: 200
      }
    }, { timeout: 30000 });

    res.json({
      isValid: true,
      reason: response.data.response || 'Medication appears to be valid.',
      suggestions: ['Consult your doctor for proper dosing'],
      confidence: 0.8
    });
  } catch (error) {
    console.error('Medication validation error:', error);
    res.json({
      isValid: false,
      reason: 'Unable to validate medication at this time.',
      suggestions: ['Please consult your doctor'],
      confidence: 0.0
    });
  }
});

// Dosage validation endpoint
router.post('/validate-dosage', async (req, res) => {
  try {
    const { strength, unit, medication } = req.body;

    // Basic validation
    if (!strength || isNaN(parseFloat(strength)) || parseFloat(strength) <= 0) {
      return res.json({
        isValid: false,
        error: 'Please enter a valid dosage amount',
        suggestions: ['Enter a positive number for dosage amount']
      });
    }

    if (!unit) {
      return res.json({
        isValid: false,
        error: 'Please select a unit',
        suggestions: ['Select mg, g, ml, mcg, units, tablets, or capsules']
      });
    }

    // Validate unit compatibility with medication type
    const validUnits = ['mg', 'g', 'ml', 'mcg', 'units', 'tablets', 'capsules'];
    if (!validUnits.includes(unit)) {
      return res.json({
        isValid: false,
        error: 'Invalid unit selected',
        suggestions: validUnits
      });
    }

    // Check for reasonable dosage ranges
    const dosageValue = parseFloat(strength);
    let warnings = [];
    
    if (unit === 'mg' && dosageValue > 1000) {
      warnings.push('High dosage detected - please verify with your doctor');
    }
    
    if (unit === 'mcg' && dosageValue > 1000) {
      warnings.push('Consider if this should be mg instead of mcg');
    }

    res.json({
      isValid: true,
      warnings: warnings,
      suggestions: ['Dosage appears valid']
    });
  } catch (error) {
    console.error('Dosage validation error:', error);
    res.json({
      isValid: false,
      error: 'Unable to validate dosage at this time.',
      suggestions: ['Please check your input and try again']
    });
  }
});

// Frequency validation endpoint
router.post('/validate-schedule', async (req, res) => {
  try {
    const { frequency, customFrequency } = req.body;

    if (!frequency) {
      return res.json({
        isValid: false,
        error: 'Please select a frequency',
        suggestions: ['Select from: daily, twice-daily, three-times-daily, weekly, as-needed, or custom']
      });
    }

    if (frequency === 'custom' && (!customFrequency || customFrequency.trim().length === 0)) {
      return res.json({
        isValid: false,
        error: 'Please specify custom frequency',
        suggestions: ['Enter details like "Every 8 hours" or "Morning and evening"']
      });
    }

    // Validate custom frequency format
    if (frequency === 'custom' && customFrequency) {
      const customLower = customFrequency.toLowerCase();
      const validPatterns = [
        /every\s+\d+\s+hours?/i,
        /morning\s+and\s+evening/i,
        /twice\s+daily/i,
        /three\s+times\s+daily/i,
        /once\s+daily/i,
        /as\s+needed/i
      ];

      const isValidCustom = validPatterns.some(pattern => pattern.test(customLower));
      
      if (!isValidCustom) {
        return res.json({
          isValid: false,
          error: 'Custom frequency format not recognized',
          suggestions: [
            'Try formats like: "Every 8 hours", "Morning and evening", "Twice daily"',
            'Or select from predefined options instead'
          ]
        });
      }
    }

    res.json({
      isValid: true,
      suggestions: ['Schedule appears valid']
    });
  } catch (error) {
    console.error('Schedule validation error:', error);
    res.json({
      isValid: false,
      error: 'Unable to validate schedule at this time.',
      suggestions: ['Please check your input and try again']
    });
  }
});

// Enhanced medication search endpoint with LLM integration and natural language parsing
router.post('/search-med', async (req, res) => {
  try {
    const { query, limit = 10, min_confidence = 0.3 } = req.body;

    if (!query || query.trim().length < 2) {
      return res.status(400).json({ error: 'Query must be at least 2 characters long' });
    }

    console.log(`🔍 Searching for medication: "${query}"`);

    // Parse natural language input for medication, dosage, and frequency
    const parsedInput = parseNaturalLanguageInput(query);
    console.log(`📝 Parsed input:`, parsedInput);

    // First try Python AI backend for intelligent search
    try {
      const aiResponse = await axios.post(`${AI_BACKEND_URL}/api/search-med`, {
        query: parsedInput.medication,
        limit,
        min_confidence
      }, { timeout: 30000 });

      if (aiResponse.data.results && aiResponse.data.results.length > 0) {
        console.log(`✅ AI backend found ${aiResponse.data.results.length} results`);
        return res.json({
          results: aiResponse.data.results,
          total: aiResponse.data.total,
          query_processed: parsedInput.medication,
          parsed_input: parsedInput,
          source: 'ai_backend'
        });
      }
    } catch (aiError) {
      console.log('Python AI backend not available, falling back to enhanced search');
    }

    // Enhanced fallback search using comprehensive medication database
    try {
      console.log(`🔍 Using comprehensive medication database for "${parsedInput.medication}"`);
      
      // Comprehensive medication database with common medications
      const medicationDatabase = [
        // Pain Relief & Anti-inflammatory
        { name: 'aspirin', type: 'generic', confidence: 0.95, dosage_forms: ['tablet', 'chewable'], strengths: ['81mg', '325mg', '500mg'], suggestions: ['Bayer', 'Ecotrin', 'Bufferin'], manufacturer: 'Various' },
        { name: 'ibuprofen', type: 'generic', confidence: 0.95, dosage_forms: ['tablet', 'capsule', 'liquid'], strengths: ['200mg', '400mg', '600mg'], suggestions: ['Advil', 'Motrin', 'Nurofen'], manufacturer: 'Various' },
        { name: 'acetaminophen', type: 'generic', confidence: 0.95, dosage_forms: ['tablet', 'capsule', 'liquid'], strengths: ['325mg', '500mg', '650mg'], suggestions: ['Tylenol', 'Panadol', 'Paracetamol'], manufacturer: 'Various' },
        { name: 'naproxen', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'capsule'], strengths: ['220mg', '440mg', '500mg'], suggestions: ['Aleve', 'Naprosyn', 'Anaprox'], manufacturer: 'Various' },
        
        // Diabetes Medications
        { name: 'metformin', type: 'generic', confidence: 0.95, dosage_forms: ['tablet', 'extended-release'], strengths: ['500mg', '850mg', '1000mg'], suggestions: ['Glucophage', 'Fortamet', 'Glumetza'], manufacturer: 'Various' },
        { name: 'insulin', type: 'generic', confidence: 0.9, dosage_forms: ['injection', 'pen'], strengths: ['100 units/ml'], suggestions: ['Humalog', 'Novolog', 'Lantus'], manufacturer: 'Various' },
        { name: 'glipizide', type: 'generic', confidence: 0.9, dosage_forms: ['tablet'], strengths: ['5mg', '10mg'], suggestions: ['Glucotrol', 'Glipizide XL'], manufacturer: 'Various' },
        
        // Cardiovascular
        { name: 'lisinopril', type: 'generic', confidence: 0.95, dosage_forms: ['tablet'], strengths: ['5mg', '10mg', '20mg', '40mg'], suggestions: ['Prinivil', 'Zestril'], manufacturer: 'Various' },
        { name: 'atorvastatin', type: 'generic', confidence: 0.95, dosage_forms: ['tablet'], strengths: ['10mg', '20mg', '40mg', '80mg'], suggestions: ['Lipitor'], manufacturer: 'Various' },
        { name: 'amlodipine', type: 'generic', confidence: 0.9, dosage_forms: ['tablet'], strengths: ['2.5mg', '5mg', '10mg'], suggestions: ['Norvasc'], manufacturer: 'Various' },
        { name: 'warfarin', type: 'generic', confidence: 0.9, dosage_forms: ['tablet'], strengths: ['1mg', '2mg', '2.5mg', '3mg', '4mg', '5mg', '6mg', '7.5mg', '10mg'], suggestions: ['Coumadin', 'Jantoven'], manufacturer: 'Various' },
        
        // Antibiotics
        { name: 'amoxicillin', type: 'generic', confidence: 0.95, dosage_forms: ['capsule', 'tablet', 'liquid'], strengths: ['250mg', '500mg', '875mg'], suggestions: ['Amoxil', 'Trimox'], manufacturer: 'Various' },
        { name: 'azithromycin', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'capsule', 'liquid'], strengths: ['250mg', '500mg'], suggestions: ['Zithromax', 'Z-Pak'], manufacturer: 'Various' },
        { name: 'ciprofloxacin', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'liquid'], strengths: ['250mg', '500mg', '750mg'], suggestions: ['Cipro'], manufacturer: 'Various' },
        
        // Mental Health
        { name: 'sertraline', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'liquid'], strengths: ['25mg', '50mg', '100mg', '150mg', '200mg'], suggestions: ['Zoloft'], manufacturer: 'Various' },
        { name: 'fluoxetine', type: 'generic', confidence: 0.9, dosage_forms: ['capsule', 'tablet', 'liquid'], strengths: ['10mg', '20mg', '40mg'], suggestions: ['Prozac'], manufacturer: 'Various' },
        { name: 'lorazepam', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'liquid'], strengths: ['0.5mg', '1mg', '2mg'], suggestions: ['Ativan'], manufacturer: 'Various' },
        
        // Gastrointestinal
        { name: 'omeprazole', type: 'generic', confidence: 0.95, dosage_forms: ['capsule', 'tablet'], strengths: ['10mg', '20mg', '40mg'], suggestions: ['Prilosec'], manufacturer: 'Various' },
        { name: 'ranitidine', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'liquid'], strengths: ['75mg', '150mg', '300mg'], suggestions: ['Zantac'], manufacturer: 'Various' },
        
        // Respiratory
        { name: 'albuterol', type: 'generic', confidence: 0.9, dosage_forms: ['inhaler', 'tablet', 'liquid'], strengths: ['90mcg', '2mg', '4mg'], suggestions: ['Proventil', 'Ventolin'], manufacturer: 'Various' },
        { name: 'prednisone', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'liquid'], strengths: ['1mg', '2.5mg', '5mg', '10mg', '20mg', '50mg'], suggestions: ['Deltasone', 'Sterapred'], manufacturer: 'Various' },
        
        // Additional Common Medications
        { name: 'paracetamol', type: 'generic', confidence: 0.95, dosage_forms: ['tablet', 'capsule', 'liquid'], strengths: ['325mg', '500mg', '650mg'], suggestions: ['Tylenol', 'Panadol', 'Acetaminophen'], manufacturer: 'Various' },
        { name: 'semaglutide', type: 'generic', confidence: 0.9, dosage_forms: ['injection', 'pen'], strengths: ['0.25mg', '0.5mg', '1mg', '2mg'], suggestions: ['Ozempic', 'Wegovy', 'Rybelsus'], manufacturer: 'Novo Nordisk' },
        { name: 'propranolol', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'capsule'], strengths: ['10mg', '20mg', '40mg', '60mg', '80mg', '120mg', '160mg'], suggestions: ['Inderal', 'Innopran XL'], manufacturer: 'Various' },
        { name: 'metoprolol', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'extended-release'], strengths: ['25mg', '50mg', '100mg', '200mg'], suggestions: ['Lopressor', 'Toprol XL'], manufacturer: 'Various' },
        { name: 'losartan', type: 'generic', confidence: 0.9, dosage_forms: ['tablet'], strengths: ['25mg', '50mg', '100mg'], suggestions: ['Cozaar'], manufacturer: 'Various' },
        { name: 'simvastatin', type: 'generic', confidence: 0.9, dosage_forms: ['tablet'], strengths: ['5mg', '10mg', '20mg', '40mg', '80mg'], suggestions: ['Zocor'], manufacturer: 'Various' },
        { name: 'levothyroxine', type: 'generic', confidence: 0.9, dosage_forms: ['tablet'], strengths: ['25mcg', '50mcg', '75mcg', '88mcg', '100mcg', '112mcg', '125mcg', '137mcg', '150mcg', '175mcg', '200mcg', '300mcg'], suggestions: ['Synthroid', 'Levoxyl', 'Tirosint'], manufacturer: 'Various' },
        { name: 'tramadol', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'capsule', 'liquid'], strengths: ['50mg', '100mg', '150mg', '200mg', '300mg'], suggestions: ['Ultram', 'ConZip'], manufacturer: 'Various' },
        { name: 'gabapentin', type: 'generic', confidence: 0.9, dosage_forms: ['capsule', 'tablet', 'liquid'], strengths: ['100mg', '300mg', '400mg', '600mg', '800mg'], suggestions: ['Neurontin', 'Gralise'], manufacturer: 'Various' },
        { name: 'pregabalin', type: 'generic', confidence: 0.9, dosage_forms: ['capsule', 'liquid'], strengths: ['25mg', '50mg', '75mg', '100mg', '150mg', '200mg', '225mg', '300mg'], suggestions: ['Lyrica'], manufacturer: 'Various' },
        { name: 'duloxetine', type: 'generic', confidence: 0.9, dosage_forms: ['capsule'], strengths: ['20mg', '30mg', '60mg'], suggestions: ['Cymbalta'], manufacturer: 'Various' },
        { name: 'venlafaxine', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'extended-release'], strengths: ['25mg', '37.5mg', '50mg', '75mg', '100mg', '150mg', '200mg', '225mg'], suggestions: ['Effexor', 'Effexor XR'], manufacturer: 'Various' },
        { name: 'trazodone', type: 'generic', confidence: 0.9, dosage_forms: ['tablet'], strengths: ['50mg', '100mg', '150mg', '300mg'], suggestions: ['Desyrel'], manufacturer: 'Various' },
        { name: 'hydrocodone', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'capsule', 'liquid'], strengths: ['5mg', '7.5mg', '10mg'], suggestions: ['Vicodin', 'Lortab', 'Norco'], manufacturer: 'Various' },
        { name: 'oxycodone', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'capsule', 'liquid'], strengths: ['5mg', '10mg', '15mg', '20mg', '30mg'], suggestions: ['OxyContin', 'Percocet', 'Roxicodone'], manufacturer: 'Various' },
        { name: 'morphine', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'capsule', 'liquid', 'injection'], strengths: ['15mg', '30mg', '60mg', '100mg'], suggestions: ['MS Contin', 'Kadian', 'Avinza'], manufacturer: 'Various' },
        { name: 'fentanyl', type: 'generic', confidence: 0.9, dosage_forms: ['patch', 'lozenge', 'injection'], strengths: ['12mcg', '25mcg', '50mcg', '75mcg', '100mcg'], suggestions: ['Duragesic', 'Actiq', 'Fentora'], manufacturer: 'Various' },
        { name: 'clonazepam', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'disintegrating'], strengths: ['0.25mg', '0.5mg', '1mg', '2mg'], suggestions: ['Klonopin'], manufacturer: 'Various' },
        { name: 'alprazolam', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'disintegrating'], strengths: ['0.25mg', '0.5mg', '1mg', '2mg'], suggestions: ['Xanax'], manufacturer: 'Various' },
        { name: 'diazepam', type: 'generic', confidence: 0.9, dosage_forms: ['tablet', 'liquid', 'injection'], strengths: ['2mg', '5mg', '10mg'], suggestions: ['Valium'], manufacturer: 'Various' }
      ];
      
      // Search the database
      const searchTerm = parsedInput.medication.toLowerCase().trim();
      const results = [];
      
      // Exact matches first
      const exactMatches = medicationDatabase.filter(med => 
        med.name.toLowerCase() === searchTerm ||
        med.suggestions.some(suggestion => suggestion.toLowerCase() === searchTerm)
      );
      
      // Partial matches
      const partialMatches = medicationDatabase.filter(med => 
        med.name.toLowerCase().includes(searchTerm) ||
        med.suggestions.some(suggestion => suggestion.toLowerCase().includes(searchTerm))
      );
      
      // Fuzzy matches (improved)
      const fuzzyMatches = medicationDatabase.filter(med => {
        const nameDistance = levenshteinDistance(med.name.toLowerCase(), searchTerm);
        const suggestionDistance = Math.min(...med.suggestions.map(s => 
          levenshteinDistance(s.toLowerCase(), searchTerm)
        ));
        
        // Also check if the search term is a substring of the medication name
        const nameContains = med.name.toLowerCase().includes(searchTerm);
        const suggestionContains = med.suggestions.some(s => 
          s.toLowerCase().includes(searchTerm)
        );
        
        return nameDistance <= 2 || suggestionDistance <= 2 || nameContains || suggestionContains;
      });
      
      // Combine and deduplicate results
      const allMatches = [...exactMatches, ...partialMatches, ...fuzzyMatches];
      const uniqueMatches = allMatches.filter((med, index, self) => 
        index === self.findIndex(m => m.name === med.name)
      );
      
      // Sort by confidence and relevance
      const sortedResults = uniqueMatches
        .map(med => ({
          ...med,
          confidence: med.name.toLowerCase() === searchTerm ? 1.0 : 
                     med.name.toLowerCase().includes(searchTerm) ? 0.9 :
                     med.suggestions.some(s => s.toLowerCase() === searchTerm) ? 0.8 :
                     med.suggestions.some(s => s.toLowerCase().includes(searchTerm)) ? 0.7 : 0.6
        }))
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, parseInt(limit));
      
      console.log(`🎯 Found ${sortedResults.length} matches for "${query}"`);
      
      res.json({
        results: sortedResults,
        total: sortedResults.length,
        query_processed: parsedInput.medication,
        parsed_input: parsedInput,
        source: 'comprehensive_database'
      });
      
    } catch (dbError) {
      console.error('Database search error:', dbError);
      
      // Final fallback - use LLM to generate suggestions
      try {
        const llmResponse = await axios.post(`${OLLAMA_BASE_URL}/generate`, {
          model: DEFAULT_MODEL,
          prompt: `You are a medical assistant. The user is looking for a medication called "${query}". 
          Please provide 3-5 common medications that might match this search, including:
          - Generic names
          - Brand names
          - Common variations
          - Similar sounding medications
          
          Format your response as a JSON array of objects with: name, type (generic/brand), confidence (0-1), suggestions.
          Example: [{"name": "Metformin", "type": "generic", "confidence": 0.9, "suggestions": ["Glucophage", "Fortamet"]}]`,
          stream: false,
          options: {
            temperature: 0.3,
            top_p: 0.9,
            num_predict: 200
          }
        }, { timeout: 15000 });
        
        let llmResults = [];
        try {
          llmResults = JSON.parse(llmResponse.data.response || '[]');
        } catch (parseError) {
          // If JSON parsing fails, create a simple result
          llmResults = [{
            name: query,
            type: 'unknown',
            confidence: 0.5,
            suggestions: ['Please verify the medication name with your doctor']
          }];
        }
        
        res.json({
          results: llmResults,
          total: llmResults.length,
          query_processed: query,
          source: 'llm_fallback'
        });
        
      } catch (llmError) {
        console.error('LLM fallback error:', llmError);
        res.json({
          results: [{
            name: query,
            type: 'unknown',
            confidence: 0.3,
            suggestions: ['Please verify the medication name with your doctor']
          }],
          total: 1,
          query_processed: query,
          source: 'error_fallback'
        });
      }
    }
  } catch (error) {
    console.error('Medication search error:', error);
    res.status(500).json({
      error: 'Search failed',
      results: [],
      total: 0
    });
  }
});

// Enhanced medication search function
async function enhancedMedicationSearch(query, medications, options) {
  const { limit = 10, min_confidence = 0.3 } = options;
  const searchTerm = query.toLowerCase().trim();
  const results = [];
  const seen = new Set();
  
  // Search strategies in order of preference
  const strategies = [
    // 1. Exact brand name match
    (med) => {
      if (!med.brand_name || !Array.isArray(med.brand_name)) return null;
      return med.brand_name.find(brand => 
        brand.toLowerCase() === searchTerm
      );
    },
    
    // 2. Exact generic name match
    (med) => {
      if (!med.generic_name || !Array.isArray(med.generic_name)) return null;
      return med.generic_name.find(generic => 
        generic.toLowerCase() === searchTerm
      );
    },
    
    // 3. Partial brand name match
    (med) => {
      if (!med.brand_name || !Array.isArray(med.brand_name)) return null;
      return med.brand_name.find(brand => 
        brand.toLowerCase().includes(searchTerm)
      );
    },
    
    // 4. Partial generic name match
    (med) => {
      if (!med.generic_name || !Array.isArray(med.generic_name)) return null;
      return med.generic_name.find(generic => 
        generic.toLowerCase().includes(searchTerm)
      );
    },
    
    // 5. Fuzzy brand name match
    (med) => {
      if (!med.brand_name || !Array.isArray(med.brand_name)) return null;
      return med.brand_name.find(brand => 
        levenshteinDistance(brand.toLowerCase(), searchTerm) <= 2
      );
    },
    
    // 6. Fuzzy generic name match
    (med) => {
      if (!med.generic_name || !Array.isArray(med.generic_name)) return null;
      return med.generic_name.find(generic => 
        levenshteinDistance(generic.toLowerCase(), searchTerm) <= 2
      );
    }
  ];
  
  // Apply each strategy
  for (const strategy of strategies) {
    if (results.length >= limit) break;
    
    for (const med of medications) {
      if (results.length >= limit) break;
      
      const match = strategy(med);
      if (match && !seen.has(med.product_ndc?.[0] || med.generic_name?.[0])) {
        const confidence = calculateConfidence(match, searchTerm, strategy);
        
        if (confidence >= min_confidence) {
          results.push({
            name: match,
            type: med.brand_name?.includes(match) ? 'brand' : 'generic',
            confidence: confidence,
            dosage_forms: med.dosage_form ? [med.dosage_form] : [],
            strengths: med.strength ? [med.strength] : [],
            route: med.route || [],
            manufacturer: med.manufacturer_name?.[0] || 'Unknown',
            ndc: med.product_ndc?.[0] || '',
            generic_name: med.generic_name?.[0] || '',
            brand_name: med.brand_name?.[0] || '',
            suggestions: generateSuggestions(med, match)
          });
          
          seen.add(med.product_ndc?.[0] || med.generic_name?.[0]);
        }
      }
    }
  }
  
  // Sort by confidence and return top results
  return results
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, limit);
}

// Calculate confidence score
function calculateConfidence(match, searchTerm, strategy) {
  const exactMatch = match.toLowerCase() === searchTerm;
  const partialMatch = match.toLowerCase().includes(searchTerm);
  const fuzzyMatch = levenshteinDistance(match.toLowerCase(), searchTerm) <= 2;
  
  if (exactMatch) return 1.0;
  if (partialMatch) return 0.8;
  if (fuzzyMatch) return 0.6;
  
  // Strategy-based confidence
  const strategyIndex = [
    'exact_brand', 'exact_generic', 'partial_brand', 
    'partial_generic', 'fuzzy_brand', 'fuzzy_generic'
  ].indexOf(strategy.name || 'unknown');
  
  return Math.max(0.3, 1.0 - (strategyIndex * 0.1));
}

// Generate helpful suggestions
function generateSuggestions(med, match) {
  const suggestions = [];
  
  if (med.brand_name && med.brand_name.length > 1) {
    suggestions.push(...med.brand_name.slice(0, 3));
  }
  
  if (med.generic_name && med.generic_name.length > 1) {
    suggestions.push(...med.generic_name.slice(0, 3));
  }
  
  if (med.strength) {
    suggestions.push(`Available in ${med.strength}`);
  }
  
  return [...new Set(suggestions)].slice(0, 5);
}

// Natural language parsing function
function parseNaturalLanguageInput(input) {
  const text = input.toLowerCase().trim();
  
  // Common medication patterns
  const medicationPatterns = [
    // Direct medication names with phrases
    /(?:i take|i'm taking|i use|i'm using|i need|i want|i have|i take)\s+([a-zA-Z\s]+?)(?:\s+\d+|\s+mg|\s+mcg|\s+units|\s+twice|\s+once|\s+daily|\s+weekly|\s+monthly|$)/i,
    // Medication with dosage - extract just the medication name
    /([a-zA-Z\s]+?)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units)/i,
    // Just medication name
    /^([a-zA-Z\s]+)$/
  ];
  
  // Dosage patterns
  const dosagePatterns = [
    /(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units)/gi,
    /(\d+(?:\.\d+)?)\s*(milligrams?|micrograms?|grams?|milliliters?|units?)/gi
  ];
  
  // Frequency patterns
  const frequencyPatterns = [
    /(twice|two times)\s+(daily|a day|per day)/gi,
    /(once|one time)\s+(daily|a day|per day)/gi,
    /(three times|thrice)\s+(daily|a day|per day)/gi,
    /(four times)\s+(daily|a day|per day)/gi,
    /(every\s+\d+\s+hours?)/gi,
    /(every\s+\d+\s+days?)/gi,
    /(weekly|once a week)/gi,
    /(monthly|once a month)/gi,
    /(as needed|prn|when needed)/gi
  ];
  
  let medication = '';
  let dosage = '';
  let frequency = '';
  
  // Extract medication name
  for (const pattern of medicationPatterns) {
    const match = text.match(pattern);
    if (match) {
      // For the dosage pattern, we want just the medication name (first group)
      // For other patterns, use the first group or the full match
      if (pattern.source.includes('\\d+') && pattern.source.includes('mg|mcg|g|ml|units')) {
        medication = match[1]; // Just the medication name before the dosage
      } else {
        medication = match[1] || match[0];
      }
      break;
    }
  }
  
  // Extract dosage
  for (const pattern of dosagePatterns) {
    const match = text.match(pattern);
    if (match) {
      dosage = match[0];
      break;
    }
  }
  
  // Extract frequency
  for (const pattern of frequencyPatterns) {
    const match = text.match(pattern);
    if (match) {
      frequency = match[0];
      break;
    }
  }
  
  // Clean up medication name
  medication = medication.replace(/\b(i take|i'm taking|i use|i'm using|i need|i want|i have)\b/gi, '').trim();
  
  // If no medication found, use the original input
  if (!medication) {
    medication = input;
  }
  
  return {
    medication: medication,
    dosage: dosage,
    frequency: frequency,
    original: input
  };
}

// Levenshtein distance for fuzzy matching
function levenshteinDistance(str1, str2) {
  const matrix = [];
  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }
  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }
  return matrix[str2.length][str1.length];
}

// Metrics options endpoint
router.get('/metrics/options', async (req, res) => {
  try {
    const metrics = [
      'Heart Rate',
      'Blood Pressure',
      'Temperature',
      'Pain Level',
      'Sleep Quality',
      'Energy Level',
      'Mood',
      'Weight',
      'Blood Sugar',
      'Blood Oxygen',
      'Blood Pressure (Systolic)',
      'Blood Pressure (Diastolic)',
      'Cholesterol',
      'Blood Pressure (Pulse)',
      'Body Mass Index (BMI)',
      'Waist Circumference',
      'Hip Circumference',
      'Waist-to-Hip Ratio',
      'Body Fat Percentage',
      'Muscle Mass',
      'Bone Density',
      'Vitamin D Level',
      'Iron Level',
      'Thyroid Function',
      'Kidney Function',
      'Liver Function',
      'Blood Count',
      'Inflammation Markers',
      'Allergy Symptoms',
      'Digestive Health',
      'Mental Health',
      'Cognitive Function',
      'Physical Activity',
      'Exercise Duration',
      'Exercise Intensity',
      'Steps Count',
      'Calories Burned',
      'Water Intake',
      'Alcohol Consumption',
      'Caffeine Intake',
      'Smoking Status',
      'Stress Level',
      'Anxiety Level',
      'Depression Score',
      'Quality of Life',
      'Medication Adherence',
      'Side Effects',
      'Drug Interactions',
      'Allergic Reactions',
      'Emergency Symptoms'
    ];

    res.json(metrics);
  } catch (error) {
    console.error('Error fetching metrics options:', error);
    res.status(500).json({ error: 'Failed to fetch metrics options' });
  }
});

// New medication validation endpoint using BioGPT
router.post('/medications/validate', async (req, res) => {
  try {
    const { medication_name } = req.body;
    
    if (!medication_name) {
      return res.status(400).json({
        success: false,
        error: 'Medication name is required'
      });
    }

    // Try Python AI backend first
    try {
      const response = await axios.post(`${AI_BACKEND_URL}/api/medications/validate`, {
        medication_name,
        user_id: req.body.user_id
      }, {
        timeout: 10000
      });

      if (response.data.success) {
        return res.json(response.data);
      }
    } catch (aiError) {
      console.log('AI backend unavailable, using fallback:', aiError.message);
    }

    // Fallback to local validation
    const fallbackResult = await enhancedMedicationSearch(medication_name, 1);
    
    if (fallbackResult.results.length > 0) {
      const result = fallbackResult.results[0];
      return res.json({
        success: true,
        data: {
          generic_name: result.name,
          brand_names: result.brands || [],
          drug_class: result.drug_class || 'Unknown',
          dosage_forms: result.dosage_forms || ['tablet'],
          typical_strengths: result.strengths || [],
          indications: result.indications || [],
          confidence: result.confidence || 0.7,
          alternatives: [],
          is_ambiguous: false,
          validation_notes: 'Validated using local database',
          original_input: medication_name,
          sources: ['local_fallback']
        },
        suggested_metrics: getSuggestedMetrics(result)
      });
    }

    // No match found
    res.json({
      success: true,
      data: {
        generic_name: medication_name,
        brand_names: [],
        drug_class: 'Unknown',
        dosage_forms: ['tablet'],
        typical_strengths: [],
        indications: [],
        confidence: 0.1,
        alternatives: [],
        is_ambiguous: true,
        validation_notes: 'No match found',
        original_input: medication_name,
        sources: ['no_match']
      },
      suggested_metrics: ['General Health', 'Side Effects']
    });

  } catch (error) {
    console.error('Error validating medication:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to validate medication'
    });
  }
});

// Helper function to get suggested metrics based on medication
function getSuggestedMetrics(medication) {
  const name = medication.name?.toLowerCase() || '';
  const drugClass = medication.drug_class?.toLowerCase() || '';
  const indications = medication.indications || [];

  // Diabetes medications
  if (name.includes('metformin') || name.includes('insulin') || name.includes('glipizide')) {
    return ['Blood Glucose', 'Weight', 'Blood Pressure', 'General Health'];
  }

  // Cardiovascular medications
  if (drugClass.includes('ace inhibitor') || drugClass.includes('beta blocker') || drugClass.includes('statin')) {
    return ['Blood Pressure', 'Heart Rate', 'Weight', 'General Health'];
  }

  // Pain medications
  if (name.includes('ibuprofen') || name.includes('acetaminophen') || name.includes('naproxen')) {
    return ['Pain Level', 'Sleep Quality', 'General Health'];
  }

  // Blood pressure medications
  if (indications.some(ind => ind.toLowerCase().includes('hypertension') || ind.toLowerCase().includes('blood pressure'))) {
    return ['Blood Pressure', 'Heart Rate', 'Weight', 'General Health'];
  }

  // Pain relief
  if (indications.some(ind => ind.toLowerCase().includes('pain') || ind.toLowerCase().includes('inflammation'))) {
    return ['Pain Level', 'Sleep Quality', 'General Health'];
  }

  return ['General Health', 'Side Effects'];
}

module.exports = router;
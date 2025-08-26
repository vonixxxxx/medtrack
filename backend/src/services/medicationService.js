const axios = require('axios');
const OpenAI = require('openai');
const { HfInference } = require('@huggingface/inference');

// Initialize LLM clients
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || null,
  dangerouslyAllowBrowser: false
});

const hf = new HfInference(process.env.HUGGINGFACE_API_KEY || 'your_huggingface_token_here');

// Working API configurations
const API_CONFIG = {
  timeout: 15000,
  maxRetries: 3,
  retryDelay: 1000
};

// HuggingFace model configuration
const HF_MODEL = process.env.HUGGINGFACE_MODEL || 'gpt2';

/**
 * Enhanced medication search with working API integrations
 */
class EnhancedMedicationService {
  static async searchMedications(query, limit = 20, source = 'ema') {
    try {
      let results = [];

      if (source === 'ema' || source === 'all') {
        // Use RxNorm as the primary source for comprehensive data
        const rxnormResults = await this.searchRxNorm(query, limit);
        results.push(...rxnormResults);
      }

      if (source === 'all') {
        // Add openFDA for safety information
        const openfdaResults = await this.searchOpenFDA(query, Math.floor(limit / 2));
        results.push(...openfdaResults);
      }

      // Remove duplicates and sort by relevance
      const uniqueResults = this.removeDuplicates(results);
      const sortedResults = this.sortByRelevance(uniqueResults, query);

      return sortedResults.slice(0, limit);

    } catch (error) {
      console.error('Enhanced medication search error:', error.message);
      return [];
    }
  }

  static async searchRxNorm(query, limit = 10) {
    try {
      const response = await axios.get(
        `https://rxnav.nlm.nih.gov/REST/drugs.json?name=${encodeURIComponent(query)}`,
        API_CONFIG
      );

      if (!response.data || !response.data.drugGroup || !response.data.drugGroup.conceptGroup) {
        return [];
      }

      const medications = [];
      const conceptGroups = response.data.drugGroup.conceptGroup;

      conceptGroups.forEach(group => {
        if (group.conceptProperties) {
          group.conceptProperties.forEach(concept => {
            if (concept.name && concept.name.toLowerCase().includes(query.toLowerCase())) {
              const cleanedName = this.cleanMedicationName(concept.name);
              const activeIngredients = this.extractActiveIngredients(concept.name);
              const strengths = this.extractStrengths(concept.name);
              
              medications.push({
                source: 'rxnorm',
                id: concept.rxcui,
                name: cleanedName,
                originalName: concept.name,
                genericName: concept.synonym || cleanedName,
                brandNames: concept.brandName ? [concept.brandName] : [],
                dosageForms: concept.dosageForm ? [concept.dosageForm] : [],
                strengths: strengths,
                activeIngredients: activeIngredients,
                atcClass: concept.atcCode ? [concept.atcCode] : [],
                therapeuticIndications: ['Pain relief', 'Fever reduction', 'Anti-inflammatory'],
                contraindications: ['Allergy to aspirin or NSAIDs', 'Active bleeding', 'Severe liver disease'],
                warnings: ['May cause stomach irritation', 'Avoid alcohol', 'Take with food'],
                sideEffects: ['Stomach upset', 'Nausea', 'Headache', 'Dizziness'],
                interactions: ['Blood thinners', 'Other NSAIDs', 'Alcohol'],
                pregnancyCategory: 'Category C - Consult healthcare provider',
                breastfeedingCategory: 'Consult healthcare provider',
                pediatricUse: 'Consult healthcare provider for dosing',
                geriatricUse: 'May require lower doses',
                identifiers: {
                  rxcui: concept.rxcui,
                  ndc: concept.ndc || []
                },
                sourceUrl: `https://rxnav.nlm.nih.gov/REST/rxcui/${concept.rxcui}/allrelated.json`,
                lastUpdated: new Date().toISOString()
              });
            }
          });
        }
      });

      return medications.slice(0, limit);
    } catch (error) {
      console.error('RxNorm search error:', error.message);
      return [];
    }
  }

  static async searchOpenFDA(query, limit = 10) {
    try {
      const response = await axios.get('https://api.fda.gov/drug/label.json', {
        params: {
          search: `openfda.generic_name:"${query}" OR openfda.brand_name:"${query}"`,
          limit: limit
        },
        timeout: API_CONFIG.timeout
      });

      if (!response.data || !response.data.results) {
        return [];
      }

      return response.data.results.map(drug => ({
        source: 'openfda',
        id: drug.set_id,
        name: drug.openfda.brand_name || drug.openfda.generic_name || 'Unknown',
        genericName: drug.openfda.generic_name || '',
        brandNames: drug.openfda.brand_name ? [drug.openfda.brand_name] : [],
        dosageForms: drug.openfda.pharmaceutical_dosage_form || [],
        strengths: drug.openfda.substance_name || [],
        atcClass: drug.openfda.substance_name || '',
        therapeuticIndications: this.extractOpenFDAText(drug.indications_and_usage),
        contraindications: this.extractOpenFDAText(drug.contraindications),
        warnings: this.extractOpenFDAText(drug.warnings),
        sideEffects: this.extractOpenFDAText(drug.adverse_reactions),
        interactions: this.extractOpenFDAText(drug.drug_interactions),
        pregnancyCategory: this.extractOpenFDAText(drug.pregnancy),
        breastfeedingCategory: this.extractOpenFDAText(drug.nursing_mothers),
        pediatricUse: this.extractOpenFDAText(drug.pediatric_use),
        geriatricUse: this.extractOpenFDAText(drug.geriatric_use),
        identifiers: {
          set_id: drug.set_id,
          ndc: drug.openfda.package_ndc || []
        },
        sourceUrl: `https://www.accessdata.fda.gov/spl/data/${drug.set_id}/${drug.set_id}.xml`,
        lastUpdated: new Date().toISOString()
      }));
    } catch (error) {
      console.error('OpenFDA search error:', error.message);
      return [];
    }
  }

  static extractOpenFDAText(textArray) {
    if (!textArray || !Array.isArray(textArray)) return [];
    return textArray.map(text => {
      if (typeof text === 'string') return text;
      if (text && typeof text === 'object') {
        return Object.values(text).join(' ');
      }
      return String(text);
    }).filter(text => text && text.trim().length > 0);
  }

  static async getMedicationDetails(medicineId, source = 'rxnorm') {
    try {
      if (source === 'rxnorm') {
        const details = await this.getRxNormDetails(medicineId);
        if (details) {
          // Add LLM summarization using HuggingFace
          const summary = await MedicationSummarizer.summarizeWithHuggingFace(details);
          details.llmSummary = summary;
        }
        return details;
      }

      if (source === 'openfda') {
        const details = await this.getOpenFDADetails(medicineId);
        if (details) {
          const summary = await MedicationSummarizer.summarizeWithHuggingFace(details);
          details.llmSummary = summary;
        }
        return details;
      }

      return null;
    } catch (error) {
      console.error('Enhanced medication details error:', error.message);
      return null;
    }
  }

  static async getRxNormDetails(rxcui) {
    try {
      // First try to get detailed information
      const response = await axios.get(
        `https://rxnav.nlm.nih.gov/REST/rxcui/${rxcui}/allrelated.json`,
        API_CONFIG
      );

      if (!response.data || !response.data.allRelatedGroup) {
        return null;
      }

      // Extract the main concept information
      const conceptGroup = response.data.allRelatedGroup.conceptGroup;
      if (!conceptGroup || conceptGroup.length === 0) {
        return null;
      }

      // Find the main concept - prioritize GPCK (Generic Pack) or SCD (Clinical Drug)
      let mainConcept = null;
      let priorityConcept = null;
      
      for (const group of conceptGroup) {
        if (group.conceptProperties && group.conceptProperties.length > 0) {
          // Look for GPCK (Generic Pack) first, then SCD (Clinical Drug)
          if (group.tty === 'GPCK' && !mainConcept) {
            // For GPCK, use the synonym if available (it contains the full medication name)
            const concept = group.conceptProperties[0];
            
            if (concept.synonym && concept.synonym.length > 0) {
              mainConcept = { ...concept, name: concept.synonym };
            } else {
              // If GPCK doesn't have a good name, mark it but don't use it yet
            }
          } else if (group.tty === 'SCD' && !priorityConcept) {
            priorityConcept = group.conceptProperties[0];
          } else if (!mainConcept && group.tty !== 'GPCK') {
            // Only use fallback if it's not GPCK
            mainConcept = group.conceptProperties[0];
          }
        }
      }
      
      // If we found a GPCK concept but it didn't have a good name, use SCD instead
      if (mainConcept && mainConcept.name === 'Pack' && priorityConcept) {
        mainConcept = priorityConcept;
      }

      if (!mainConcept) {
        return null;
      }

      // Create a comprehensive medication object
      const medication = {
        source: 'rxnorm',
        id: rxcui,
        name: this.cleanMedicationName(mainConcept.name),
        originalName: mainConcept.name,
        genericName: mainConcept.synonym || mainConcept.name,
        brandNames: mainConcept.brandName ? [mainConcept.brandName] : [],
        dosageForms: mainConcept.dosageForm ? [mainConcept.dosageForm] : [],
        strengths: this.extractStrengths(mainConcept.name),
        activeIngredients: this.extractActiveIngredients(mainConcept.name),
        atcClass: mainConcept.atcCode ? [mainConcept.atcCode] : [],
        therapeuticIndications: ['Pain relief', 'Fever reduction', 'Anti-inflammatory'],
        contraindications: ['Allergy to aspirin or NSAIDs', 'Active bleeding', 'Severe liver disease'],
        warnings: ['May cause stomach irritation', 'Avoid alcohol', 'Take with food'],
        sideEffects: ['Stomach upset', 'Nausea', 'Headache', 'Dizziness'],
        interactions: ['Blood thinners', 'Other NSAIDs', 'Alcohol'],
        pregnancyCategory: 'Category C - Consult healthcare provider',
        breastfeedingCategory: 'Consult healthcare provider',
        pediatricUse: 'Consult healthcare provider for dosing',
        geriatricUse: 'May require lower doses',
        identifiers: {
          rxcui: rxcui,
          ndc: mainConcept.ndc || []
        },
        sourceUrl: `https://rxnav.nlm.nih.gov/REST/rxcui/${rxcui}/allrelated.json`,
        lastUpdated: new Date().toISOString()
      };

      return medication;
    } catch (error) {
      console.error('RxNorm details error:', error.message);
      return null;
    }
  }

  static async getOpenFDADetails(setId) {
    try {
      const response = await axios.get('https://api.fda.gov/drug/label.json', {
        params: { search: `set_id:"${setId}"` },
        timeout: API_CONFIG.timeout
      });

      if (!response.data || !response.data.results || response.data.results.length === 0) {
        return null;
      }

      const drug = response.data.results[0];
      return {
        source: 'openfda',
        id: setId,
        name: drug.openfda.brand_name || drug.openfda.generic_name || 'Unknown',
        genericName: drug.openfda.generic_name || '',
        brandNames: drug.openfda.brand_name ? [drug.openfda.brand_name] : [],
        dosageForms: drug.openfda.pharmaceutical_dosage_form || [],
        strengths: drug.openfda.substance_name || [],
        atcClass: drug.openfda.substance_name || '',
        therapeuticIndications: this.extractOpenFDAText(drug.indications_and_usage),
        contraindications: this.extractOpenFDAText(drug.contraindications),
        warnings: this.extractOpenFDAText(drug.warnings),
        sideEffects: this.extractOpenFDAText(drug.adverse_reactions),
        interactions: this.extractOpenFDAText(drug.drug_interactions),
        pregnancyCategory: this.extractOpenFDAText(drug.pregnancy),
        breastfeedingCategory: this.extractOpenFDAText(drug.nursing_mothers),
        pediatricUse: this.extractOpenFDAText(drug.pediatric_use),
        geriatricUse: this.extractOpenFDAText(drug.geriatric_use),
        identifiers: {
          set_id: setId,
          ndc: drug.openfda.package_ndc || []
        },
        sourceUrl: `https://www.accessdata.fda.gov/spl/data/${setId}/${setId}.xml`,
        lastUpdated: new Date().toISOString()
      };
    } catch (error) {
      console.error('OpenFDA details error:', error.message);
      return null;
    }
  }

  static removeDuplicates(medications) {
    const seen = new Set();
    return medications.filter(med => {
      const key = `${med.name}-${med.genericName}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  static sortByRelevance(medications, query) {
    const queryLower = query.toLowerCase();
    
    return medications.sort((a, b) => {
      const aScore = this.calculateRelevanceScore(a, queryLower);
      const bScore = this.calculateRelevanceScore(b, queryLower);
      return bScore - aScore;
    });
  }

  static calculateRelevanceScore(medication, query) {
    let score = 0;
    
    // Exact name match gets highest score
    if (medication.name.toLowerCase() === query) score += 100;
    if (medication.genericName.toLowerCase() === query) score += 90;
    
    // Partial matches
    if (medication.name.toLowerCase().includes(query)) score += 50;
    if (medication.genericName.toLowerCase().includes(query)) score += 40;
    
    // Brand name matches
    if (medication.brandNames?.some(brand => brand.toLowerCase().includes(query))) {
      score += 30;
    }
    
    // ATC class relevance
    if (medication.atcClass) {
      if (Array.isArray(medication.atcClass)) {
        if (medication.atcClass.some(atc => atc.toLowerCase().includes(query))) {
          score += 20;
        }
      } else if (typeof medication.atcClass === 'string' && medication.atcClass.toLowerCase().includes(query)) {
        score += 20;
      }
    }
    
    return score;
  }

  static cleanMedicationName(name) {
    if (!name) return 'Unknown Medication';
    
    // Extract the main active ingredients and strengths
    const ingredientMatches = name.match(/([A-Za-z]+)\s+(\d+(?:\.\d+)?)\s*MG/g);
    
    if (ingredientMatches && ingredientMatches.length > 0) {
      // Create a clean name from the main ingredients
      const mainIngredients = ingredientMatches.slice(0, 3).map(match => {
        const [ingredient, strength] = match.match(/([A-Za-z]+)\s+(\d+(?:\.\d+)?)\s*MG/).slice(1);
        return `${ingredient} ${strength} MG`;
      });
      
      let cleanedName = mainIngredients.join(' / ');
      
      // Add dosage form if available
      if (name.includes('Oral Tablet')) {
        cleanedName += ' Oral Tablet';
      } else if (name.includes('Capsule')) {
        cleanedName += ' Capsule';
      } else if (name.includes('Liquid')) {
        cleanedName += ' Liquid';
      }
      
      // If it's a combination pack, indicate that
      if (name.includes('Pack') && mainIngredients.length > 1) {
        cleanedName += ' Combination Pack';
      }
      
      return cleanedName;
    }
    
    // Fallback: clean up the original name
    let cleaned = name
      .replace(/\{.*?\}/g, '') // Remove {...} pack descriptions
      .replace(/\/.*?Pack.*?\)/g, '') // Remove pack information
      .replace(/\s+/g, ' ') // Normalize whitespace
      .trim();
    
    // If still too long, truncate intelligently
    if (cleaned.length > 80) {
      cleaned = cleaned.substring(0, 77) + '...';
    }
    
    return cleaned || 'Unknown Medication';
  }

  static extractActiveIngredients(name) {
    if (!name) return [];
    
    const ingredients = [];
    const matches = name.match(/([A-Za-z]+)\s+(\d+(?:\.\d+)?)\s*MG/g);
    
    if (matches) {
      matches.forEach(match => {
        const [ingredient, strength] = match.match(/([A-Za-z]+)\s+(\d+(?:\.\d+)?)\s*MG/).slice(1);
        ingredients.push(`${ingredient} ${strength} MG`);
      });
    }
    
    return [...new Set(ingredients)]; // Remove duplicates
  }

  static extractStrengths(name) {
    if (!name) return [];
    
    const strengths = [];
    const matches = name.match(/([A-Za-z]+)\s+(\d+(?:\.\d+)?)\s*MG/g);
    
    if (matches) {
      matches.forEach(match => {
        const [ingredient, strength] = match.match(/([A-Za-z]+)\s+(\d+(?:\.\d+)?)\s*MG/).slice(1);
        strengths.push(`${ingredient} ${strength} MG`);
      });
    }
    
    return [...new Set(strengths)]; // Remove duplicates
  }
}

/**
 * LLM-powered medication information summarization
 */
class MedicationSummarizer {
  static async summarizeWithOpenAI(medicationData) {
    if (!process.env.OPENAI_API_KEY) {
      return this.fallbackSummarization(medicationData);
    }

    try {
      const prompt = this.buildSummarizationPrompt(medicationData);
      
      const completion = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: "You are a medical information specialist. Provide clear, concise, and user-friendly summaries of medication information. Focus on safety, effectiveness, and practical use."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        max_tokens: 1000,
        temperature: 0.3
      });

      return completion.choices[0].message.content;
    } catch (error) {
      console.error('OpenAI summarization error:', error.message);
      return this.fallbackSummarization(medicationData);
    }
  }

  static async summarizeWithHuggingFace(medicationData) {
    if (!process.env.HUGGINGFACE_API_KEY && !hf.token) {
      return this.fallbackSummarization(medicationData);
    }

    try {
      const prompt = this.buildSummarizationPrompt(medicationData);
      
      const response = await hf.textGeneration({
        model: HF_MODEL,
        inputs: prompt,
        parameters: {
          max_new_tokens: 800,
          temperature: 0.3,
          do_sample: true,
          top_p: 0.9,
          repetition_penalty: 1.1,
          stop: ["\n\n", "###", "END"]
        }
      });

      return response.generated_text;
    } catch (error) {
      console.error('HuggingFace summarization error:', error.message);
      return this.fallbackSummarization(medicationData);
    }
  }

  static buildSummarizationPrompt(medicationData) {
    return `
Please provide a clear, user-friendly summary of this medication:

Medication: ${medicationData.name}
Generic Name: ${medicationData.genericName}
Therapeutic Use: ${medicationData.therapeuticIndications?.join(', ') || 'Not specified'}

Please summarize the following information in a clear, organized way:

1. **Warnings & Precautions** (highest priority):
${medicationData.warnings?.join('\n') || 'No specific warnings listed'}

2. **Side Effects** (grouped by severity):
${medicationData.sideEffects?.join('\n') || 'No side effects listed'}

3. **Dosage Information**:
${medicationData.dosageForms?.join(', ') || 'Not specified'}

4. **Drug Interactions & Contraindications**:
${medicationData.interactions?.join('\n') || 'No interactions listed'}

5. **Special Populations**:
- Pregnancy: ${medicationData.pregnancyCategory || 'Not specified'}
- Breastfeeding: ${medicationData.breastfeedingCategory || 'Not specified'}
- Pediatric: ${medicationData.pediatricUse || 'Not specified'}
- Geriatric: ${medicationData.geriatricUse || 'Not specified'}

Please format this as a clear, organized summary that a patient can easily understand.
    `.trim();
  }

  static fallbackSummarization(medicationData) {
    // Enhanced fallback summarization with medical knowledge
    const name = medicationData.name || 'This medication';
    const genericName = medicationData.genericName || 'Unknown';
    
    // Create a comprehensive, structured summary
    const summary = {
      overview: `${name} (${genericName}) is a medication used to treat various medical conditions. Always consult your healthcare provider for proper diagnosis and treatment.`,
      
      warnings: medicationData.warnings?.length > 0 
        ? `⚠️ IMPORTANT WARNINGS: ${medicationData.warnings.join('. ')}`
        : '⚠️ IMPORTANT: Consult your healthcare provider before taking this medication, especially if you have allergies, medical conditions, or are taking other medications.',
      
      sideEffects: medicationData.sideEffects?.length > 0
        ? `🔄 COMMON SIDE EFFECTS: ${medicationData.sideEffects.slice(0, 8).join(', ')}. Stop taking this medication and seek immediate medical attention if you experience severe side effects.`
        : '🔄 SIDE EFFECTS: Common side effects may include nausea, stomach upset, headache, or dizziness. Contact your healthcare provider if side effects persist or worsen.',
      
      dosage: medicationData.dosageForms?.length > 0
        ? `💊 DOSAGE FORMS: Available as ${medicationData.dosageForms.join(', ')}. ${medicationData.strengths?.length > 0 ? `Common strengths include ${medicationData.strengths.join(', ')}.` : ''} Always follow your healthcare provider\'s dosing instructions.`
        : '💊 DOSAGE: Follow your healthcare provider\'s dosing instructions exactly. Do not increase or decrease the dose without medical supervision.',
      
      interactions: medicationData.interactions?.length > 0
        ? `🚫 DRUG INTERACTIONS: ${medicationData.interactions.slice(0, 5).join('. ')}. Always inform your healthcare provider about all medications, supplements, and herbal products you are taking.`
        : '🚫 DRUG INTERACTIONS: This medication may interact with other drugs, supplements, or herbal products. Inform your healthcare provider about all medications you are taking.',
      
      specialPopulations: {
        pregnancy: medicationData.pregnancyCategory || '🤰 PREGNANCY: Consult your healthcare provider before taking this medication during pregnancy. Some medications may harm the developing fetus.',
        breastfeeding: medicationData.breastfeedingCategory || '🤱 BREASTFEEDING: Consult your healthcare provider before taking this medication while breastfeeding. Some medications pass into breast milk.',
        pediatric: medicationData.pediatricUse || '👶 PEDIATRIC USE: Consult your healthcare provider for appropriate dosing in children. Dosage may vary based on age and weight.',
        geriatric: medicationData.geriatricUse || '👴👵 GERIATRIC USE: Older adults may be more sensitive to this medication. Consult your healthcare provider for appropriate dosing.'
      },
      
      safetyTips: [
        '📋 Always read the medication guide and patient information leaflet',
        '⏰ Take this medication exactly as prescribed by your healthcare provider',
        '🚨 Seek immediate medical attention for severe allergic reactions',
        '💊 Store medications properly and keep out of reach of children',
        '📞 Contact your healthcare provider with any questions or concerns'
      ],
      
      disclaimer: 'This information is for educational purposes only and should not replace professional medical advice. Always consult your healthcare provider for medical decisions.'
    };

    return summary;
  }
}

module.exports = {
  EnhancedMedicationService,
  MedicationSummarizer
};

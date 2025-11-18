const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

/**
 * Parse medical history using local Ollama model with deterministic extraction
 * @param {string} medicalNotes - Raw medical history text
 * @returns {Object} - Structured medical data matching Patient schema
 */
async function runOllamaParser(medicalNotes) {
  const SYSTEM_PROMPT = `
You are a deterministic medical information extraction model.

Your role is to convert clinical notes into a strict JSON Patient Record with 0/1/null boolean values.

CRITICAL RULES:
1. NEVER guess or infer diagnoses. Only extract what is explicitly stated.
2. Negation ALWAYS sets boolean = 0:
   - "No CKD", "CKD: No", "denies CKD", "CKD absent" → "ckd": 0
   - "No T2DM", "T2DM: No", "denies type 2 diabetes" → "t2dm": 0
   - "No hypertension", "HTN: No", "denies hypertension" → "htn": 0, "hypertension": 0
   - "No ASCVD", "ASCVD: No", "denies cardiovascular disease" → "ascvd": 0
3. Borderline conditions ALWAYS = 0 (not 1):
   - "Hypertension: Borderline" → "hypertension": 0
   - "Borderline diabetes" → "t2dm": 0, "prediabetes": 1 (if explicitly stated as prediabetes)
4. Possible/risk conditions = 0 unless explicitly confirmed as diagnosis:
   - "Possible MASLD" → "masld": 0
   - "MASLD: Possible" → "masld": 0
   - "MASLD: Confirmed" → "masld": 1
5. Explicit diagnosis ALWAYS = 1:
   - "OSA: Mild" → "osa": 1
   - "T2DM: Yes" → "t2dm": 1
   - "Prediabetes: Present" → "prediabetes": 1
6. If a field does not appear in the notes, return null (not 0, not 1).
7. Preserve all numbers exactly as written. Do not round or convert units unless explicitly stated.
8. Conditions array must list ONLY positively diagnosed conditions (no negations, no borderline, no "possible").
9. Output ONLY valid JSON. Never explanations, markdown, or additional text.

BOOLEAN FIELD VALUES:
- 1 = condition explicitly present/confirmed
- 0 = condition explicitly absent/negated
- null = not mentioned in notes

OUTPUT SCHEMA (return ALL fields, use null if not found, 0/1 for booleans):
{
  "age": number | null,
  "sex": "Male" | "Female" | "Other" | null,
  "height": number | null,
  "baseline_weight": number | null,
  "baseline_bmi": number | null,
  "baseline_weight_date": string | null,
  "systolic_bp": number | null,
  "diastolic_bp": number | null,
  "baseline_hba1c": number | null,
  "baseline_hba1c_date": string | null,
  "hba1c_percent": number | null,
  "hba1c_mmol": number | null,
  "baseline_fasting_glucose": number | null,
  "random_glucose": number | null,
  "baseline_tc": number | null,
  "baseline_hdl": number | null,
  "baseline_ldl": number | null,
  "baseline_tg": number | null,
  "baseline_lipid_date": string | null,
  "t2dm": 0 | 1 | null,
  "prediabetes": 0 | 1 | null,
  "diabetes_type": "Type 1" | "Type 2" | null,
  "htn": 0 | 1 | null,
  "hypertension": 0 | 1 | null,
  "dyslipidaemia": 0 | 1 | null,
  "ascvd": 0 | 1 | null,
  "ckd": 0 | 1 | null,
  "osa": 0 | 1 | null,
  "sleep_studies": 0 | 1 | null,
  "cpap": 0 | 1 | null,
  "asthma": 0 | 1 | null,
  "ischaemic_heart_disease": 0 | 1 | null,
  "heart_failure": 0 | 1 | null,
  "cerebrovascular_disease": 0 | 1 | null,
  "pulmonary_hypertension": 0 | 1 | null,
  "dvt": 0 | 1 | null,
  "pe": 0 | 1 | null,
  "gord": 0 | 1 | null,
  "kidney_stones": 0 | 1 | null,
  "masld": 0 | 1 | null,
  "infertility": 0 | 1 | null,
  "pcos": 0 | 1 | null,
  "anxiety": 0 | 1 | null,
  "depression": 0 | 1 | null,
  "bipolar_disorder": 0 | 1 | null,
  "emotional_eating": 0 | 1 | null,
  "schizoaffective_disorder": 0 | 1 | null,
  "oa_knee": 0 | 1 | null,
  "oa_hip": 0 | 1 | null,
  "limited_mobility": 0 | 1 | null,
  "lymphoedema": 0 | 1 | null,
  "thyroid_disorder": 0 | 1 | null,
  "iih": 0 | 1 | null,
  "epilepsy": 0 | 1 | null,
  "functional_neurological_disorder": 0 | 1 | null,
  "cancer": 0 | 1 | null,
  "bariatric_gastric_band": 0 | 1 | null,
  "bariatric_sleeve": 0 | 1 | null,
  "bariatric_bypass": 0 | 1 | null,
  "bariatric_balloon": 0 | 1 | null,
  "lipid_lowering_treatment": string | null,
  "antihypertensive_medications": string | null,
  "total_qualifying_comorbidities": number | null,
  "notes": string | null,
  "conditions": string[]
}

EXAMPLES:
- "T2DM: No" → {"t2dm": 0}
- "CKD: No" → {"ckd": 0}
- "Hypertension: Borderline" → {"hypertension": 0, "htn": 0}
- "OSA: Mild" → {"osa": 1}
- "Prediabetes: Yes" → {"prediabetes": 1}
- "Possible MASLD" → {"masld": 0}
- "MASLD confirmed" → {"masld": 1}
- Not mentioned → {"masld": null}

CRITICAL: 
- ALL boolean fields must be 0, 1, or null (never true/false)
- Conditions array contains ONLY confirmed diagnoses (strings like "Prediabetes", "Dyslipidaemia", "Obstructive Sleep Apnea")
- Do NOT include negated, borderline, or "possible" conditions in the conditions array

OUTPUT REQUIREMENTS:
- Always output a single complete JSON object.
- No markdown. No code fences. No commentary. No explanation.
- Do not wrap the JSON in backticks.
- If a field is not mentioned, set it to null.
- Boolean fields must be 1 (true/diagnosed) or 0 (false/not present).
- Conditions array must include only confirmed diagnoses.
- Do not infer or guess.
`;

  try {
    console.log('🤖 Starting Ollama parsing with deterministic extraction...');
    
    // Construct the prompt with system instructions and medical notes
    const prompt = SYSTEM_PROMPT + "\n\nEXTRACT FROM THIS TEXT:\n\n" + medicalNotes;
    
    // Create AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000); // 45 second timeout
    
    let response;
    let data;
    let rawOutput;
    let jsonMatch;
    
    try {
      response = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
          model: "llama3.2:latest",
        prompt: prompt,
        stream: false,
        options: {
            temperature: 0.1,
            top_p: 0.9,
            num_predict: 2048
        }
      }),
        signal: controller.signal
    });
      
      clearTimeout(timeoutId);

    if (!response.ok) {
      console.error(`Ollama request failed: ${response.status} ${response.statusText}`);
      throw new Error(`Ollama request failed: ${response.status}`);
    }

      data = await response.json();
      rawOutput = data.response || "";
    
      console.log('🤖 Raw Ollama output (first 300 chars):', rawOutput.substring(0, 300) + '...');
    
    // Extract JSON from the response
      jsonMatch = rawOutput.match(/\{[\s\S]*\}/);
      
      // If no JSON found, try retry with fix instruction
      if (!jsonMatch) {
        console.log('⚠️ No JSON found in initial response, attempting retry...');
        const retryController = new AbortController();
        const retryTimeoutId = setTimeout(() => retryController.abort(), 20000); // 20 second timeout for retry
        
        try {
          const retryResponse = await fetch("http://localhost:11434/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model: "llama3.2:latest",
              prompt: "Fix JSON strictly. No additions. Return ONLY valid JSON:\n\n" + rawOutput,
              stream: false,
              options: {
                temperature: 0.1,
                top_p: 0.9,
                num_predict: 1024
              }
            }),
            signal: retryController.signal
          });
          
          clearTimeout(retryTimeoutId);
        
          if (retryResponse.ok) {
            const retryData = await retryResponse.json();
            const retryOutput = retryData.response || "";
            jsonMatch = retryOutput.match(/\{[\s\S]*\}/);
          }
        } catch (retryError) {
          clearTimeout(retryTimeoutId);
          if (retryError.name === 'AbortError') {
            console.error('❌ Ollama retry request timed out after 20 seconds');
            throw new Error('Ollama request timed out during JSON fix retry');
          }
          throw retryError;
        }
      }
      
    } catch (fetchError) {
      clearTimeout(timeoutId);
      if (fetchError.name === 'AbortError') {
        console.error('❌ Ollama request timed out after 45 seconds');
        throw new Error('Ollama request timed out. The model may be processing a large text. Please try with shorter notes or wait a moment.');
      }
      if (fetchError.code === 'ECONNREFUSED') {
        console.error('❌ Ollama connection refused - is Ollama running?');
        throw new Error('Ollama is not running. Please start Ollama service.');
      }
      throw fetchError;
    }
    
    if (!jsonMatch) {
      console.error('❌ No valid JSON found in Ollama response after retry');
      throw new Error('No valid JSON found in Ollama response');
    }
    
    const jsonString = jsonMatch[0];
    console.log('🤖 Extracted JSON:', jsonString.substring(0, 200) + '...');
    
    // Parse and validate JSON
    let parsedData;
    try {
      parsedData = JSON.parse(jsonString);
    } catch (parseError) {
      console.error('❌ JSON parse error:', parseError.message);
      throw new Error('Invalid JSON format from Ollama');
    }
    
    // Validate and normalize the parsed data
    try {
      parsedData = validateAndNormalizePatientData(parsedData);
      console.log('✅ Successfully parsed and validated medical data:', Object.keys(parsedData).length, 'fields');
    } catch (validationError) {
      console.error('❌ Validation error:', validationError.message);
      console.error('❌ Validation stack:', validationError.stack);
      console.error('❌ Raw model output:', rawOutput?.substring(0, 500));
      throw new Error(`Data validation failed: ${validationError.message}`);
    }
    
    return parsedData;
    
  } catch (error) {
    console.error('❌ Ollama parsing error:', error.message);
    console.error('❌ Ollama error stack:', error.stack);
    console.error('❌ Error details:', {
      rawModelOutput: error.rawModelOutput || 'N/A',
      notesLength: medicalNotes?.length || 0,
      errorType: error.constructor.name
    });
    
    // Check if Ollama is not running (connection refused)
    if (error.message.includes('ECONNREFUSED') || error.message.includes('fetch failed')) {
      console.error('⚠️ Ollama is not running or not accessible at http://localhost:11434');
      console.log('🔄 Falling back to basic text parsing...');
    } else {
      console.log('🔄 Falling back to basic text parsing due to error...');
    }
    
    // Fallback to basic text parsing if Ollama fails
    try {
    return await basicTextParser(medicalNotes);
    } catch (fallbackError) {
      console.error('❌ Fallback parser also failed:', fallbackError.message);
      throw new Error(`Both Ollama and fallback parser failed: ${fallbackError.message}`);
    }
  }
}

/**
 * Validate and normalize patient data to ensure all fields exist and types are correct
 * @param {Object} data - Parsed data from Ollama
 * @returns {Object} - Validated and normalized data
 */
function validateAndNormalizePatientData(data) {
  // Import condition mapper utilities
  const { initializeConditionFields, mapConditionsToColumns, CONDITION_FIELDS } = require('../src/utils/conditionMapper');
  
  // Define the complete schema with all Patient fields
  // IMPORTANT: All condition fields default to 0, not null
  const schema = {
    // Demographics
    age: null,
    sex: null,
    height: null,
    baseline_weight: null,
    baseline_bmi: null,
    baseline_weight_date: null,
    
    // Blood pressure
    systolic_bp: null,
    diastolic_bp: null,
    
    // Diabetes
    baseline_hba1c: null,
    baseline_hba1c_date: null,
    hba1c_percent: null,
    hba1c_mmol: null,
    baseline_fasting_glucose: null,
    random_glucose: null,
    t2dm: 0, // Initialize to 0
    prediabetes: 0, // Initialize to 0
    diabetes_type: null,
    
    // Lipids
    baseline_tc: null,
    baseline_hdl: null,
    baseline_ldl: null,
    baseline_tg: null,
    baseline_lipid_date: null,
    lipid_lowering_treatment: null,
    
    // Cardiovascular - All initialize to 0
    ascvd: 0,
    htn: 0,
    hypertension: 0,
    dyslipidaemia: 0,
    ischaemic_heart_disease: 0,
    heart_failure: 0,
    cerebrovascular_disease: 0,
    pulmonary_hypertension: 0,
    dvt: 0,
    pe: 0,
    antihypertensive_medications: null,
    
    // Sleep and respiratory - All initialize to 0
    osa: 0,
    sleep_studies: 0,
    cpap: 0,
    asthma: 0,
    
    // Renal - All initialize to 0
    ckd: 0,
    kidney_stones: 0,
    
    // Gastrointestinal - Initialize to 0
    gord: 0,
    
    // Metabolic - Initialize to 0
    masld: 0,
    
    // Reproductive - All initialize to 0
    infertility: 0,
    pcos: 0,
    
    // Mental health - All initialize to 0
    anxiety: 0,
    depression: 0,
    bipolar_disorder: 0,
    emotional_eating: 0,
    schizoaffective_disorder: 0,
    
    // Musculoskeletal - All initialize to 0
    oa_knee: 0,
    oa_hip: 0,
    limited_mobility: 0,
    lymphoedema: 0,
    
    // Endocrine - Initialize to 0
    thyroid_disorder: 0,
    
    // Neurological - All initialize to 0
    iih: 0,
    epilepsy: 0,
    functional_neurological_disorder: 0,
    
    // Oncology - Initialize to 0
    cancer: 0,
    
    // Bariatric - All initialize to 0
    bariatric_gastric_band: 0,
    bariatric_sleeve: 0,
    bariatric_bypass: 0,
    bariatric_balloon: 0,
    
    // Clinical data
    total_qualifying_comorbidities: null,
    notes: null,
    conditions: []
  };
  
  // Merge parsed data with schema, ensuring all fields exist
  let validated = { ...schema };
  
  // Define expected types for each field
  const fieldTypes = {
    // Numbers
    age: 'number', height: 'number', baseline_weight: 'number', baseline_bmi: 'number',
    systolic_bp: 'number', diastolic_bp: 'number', baseline_hba1c: 'number', hba1c_percent: 'number',
    hba1c_mmol: 'number', baseline_fasting_glucose: 'number', random_glucose: 'number',
    baseline_tc: 'number', baseline_hdl: 'number', baseline_ldl: 'number', baseline_tg: 'number',
    total_qualifying_comorbidities: 'number',
    // Booleans
    t2dm: 'boolean', prediabetes: 'boolean', htn: 'boolean', hypertension: 'boolean',
    dyslipidaemia: 'boolean', ascvd: 'boolean', ckd: 'boolean', osa: 'boolean',
    sleep_studies: 'boolean', cpap: 'boolean', asthma: 'boolean', ischaemic_heart_disease: 'boolean',
    heart_failure: 'boolean', cerebrovascular_disease: 'boolean', pulmonary_hypertension: 'boolean',
    dvt: 'boolean', pe: 'boolean', gord: 'boolean', kidney_stones: 'boolean', masld: 'boolean',
    infertility: 'boolean', pcos: 'boolean', anxiety: 'boolean', depression: 'boolean',
    bipolar_disorder: 'boolean', emotional_eating: 'boolean', schizoaffective_disorder: 'boolean',
    oa_knee: 'boolean', oa_hip: 'boolean', limited_mobility: 'boolean', lymphoedema: 'boolean',
    thyroid_disorder: 'boolean', iih: 'boolean', epilepsy: 'boolean',
    functional_neurological_disorder: 'boolean', cancer: 'boolean',
    bariatric_gastric_band: 'boolean', bariatric_sleeve: 'boolean',
    bariatric_bypass: 'boolean', bariatric_balloon: 'boolean',
    // Strings
    sex: 'string', diabetes_type: 'string', baseline_weight_date: 'string',
    baseline_hba1c_date: 'string', baseline_lipid_date: 'string',
    lipid_lowering_treatment: 'string', antihypertensive_medications: 'string', notes: 'string',
    // Arrays
    conditions: 'array'
  };
  
  for (const [key, value] of Object.entries(data)) {
    if (key in schema) {
      const expectedType = fieldTypes[key];
      
      if (value === null || value === undefined) {
        validated[key] = null;
      } else if (key === 'conditions' && Array.isArray(value)) {
        validated[key] = value;
      } else if (expectedType === 'number') {
        if (typeof value === 'number') {
          validated[key] = value;
        } else if (typeof value === 'string') {
          const num = parseFloat(value);
          validated[key] = isNaN(num) ? null : num;
        } else {
          validated[key] = null;
        }
      } else if (expectedType === 'boolean') {
        // Handle 0/1/null for boolean fields
        if (value === null || value === undefined) {
          validated[key] = null;
        } else if (typeof value === 'number') {
          // Direct numeric: 0, 1, or null
          if (value === 0) {
            validated[key] = 0;
          } else if (value === 1) {
            validated[key] = 1;
          } else {
            validated[key] = null;
          }
        } else if (typeof value === 'boolean') {
          // Convert boolean to 0/1
          validated[key] = value ? 1 : 0;
        } else if (typeof value === 'string') {
          const lower = value.toLowerCase().trim();
          // "yes", "true", "present", "1" → 1
          if (lower === 'true' || lower === 'yes' || lower === '1' || lower === 'present' || lower === 'confirmed') {
            validated[key] = 1;
          } 
          // "no", "false", "absent", "0", "denies", "none" → 0
          else if (lower === 'false' || lower === 'no' || lower === '0' || lower === 'absent' || lower === 'denies' || lower === 'none') {
            validated[key] = 0;
          } else {
            validated[key] = null;
          }
        } else {
          validated[key] = null;
        }
      } else if (expectedType === 'string') {
        validated[key] = String(value);
      } else {
        validated[key] = value;
      }
    }
  }
  
  // Ensure conditions is always an array
  if (!Array.isArray(validated.conditions)) {
    validated.conditions = [];
  }
  
  // CRITICAL: Initialize all condition fields to 0 before mapping
  validated = initializeConditionFields(validated);
  
  // CRITICAL: Map conditions array to set specific columns to 1
  if (Array.isArray(validated.conditions) && validated.conditions.length > 0) {
    validated = mapConditionsToColumns(validated, validated.conditions);
    console.log('📋 Mapped conditions to columns:', validated.conditions);
  }
  
  // Verify all required fields are present
  const missingFields = [];
  for (const key of Object.keys(schema)) {
    if (!(key in validated)) {
      missingFields.push(key);
      validated[key] = schema[key];
    }
  }
  
  // Ensure all condition fields are 0 (not null) - double check
  for (const field of CONDITION_FIELDS) {
    if (validated[field] === null || validated[field] === undefined) {
      validated[field] = 0;
    }
  }
  
  if (missingFields.length > 0) {
    console.warn('⚠️ Missing fields in parsed data, filled with defaults:', missingFields);
  }
  
  // Final validation: ensure we have at least the schema fields
  if (Object.keys(validated).length < Object.keys(schema).length) {
    throw new Error(`Validation failed: Expected ${Object.keys(schema).length} fields, got ${Object.keys(validated).length}`);
  }
  
  return validated;
}

/**
 * Fallback basic text parser when Ollama is not available
 * @param {string} medicalNotes - Raw medical history text
 * @returns {Object} - Basic structured data
 */
async function basicTextParser(medicalNotes) {
  // Keep original for case-sensitive matching, but use lowercase for most checks
  const text = medicalNotes.toLowerCase();
  const textOriginal = medicalNotes; // Keep original for exact pattern matching
  const parsedData = {};

  // Age extraction - check for "Age: 45" format first
  const ageMatch1 = textOriginal.match(/Age:\s*(\d+)/i) || text.match(/age:\s*(\d+)/i);
  const ageMatch2 = text.match(/(\d+)\s*y[eo]?[oa]?[rs]?/);
  if (ageMatch1) {
    parsedData.age = parseInt(ageMatch1[1]);
  } else if (ageMatch2) {
    parsedData.age = parseInt(ageMatch2[1]);
  }

  // Sex extraction - check for "Sex: Female" format first
  const sexMatch = textOriginal.match(/Sex:\s*(Male|Female|Other)/i);
  if (sexMatch) {
    parsedData.sex = sexMatch[1].charAt(0).toUpperCase() + sexMatch[1].slice(1).toLowerCase();
  } else if (text.includes('female') || text.includes('f/')) {
    parsedData.sex = 'Female';
  } else if (text.includes('male') || text.includes('m/')) {
    parsedData.sex = 'Male';
  }

  // BMI extraction
  const bmiMatch = text.match(/bmi[:\s]*(\d+\.?\d*)/i);
  if (bmiMatch) {
    parsedData.baseline_bmi = parseFloat(bmiMatch[1]);
  }

  // Weight extraction - check for "Weight (kg): 92" format
  const weightMatch1 = textOriginal.match(/Weight\s*\(?kg\)?[:\s]*(\d+\.?\d*)/i) || text.match(/weight\s*\(?kg\)?[:\s]*(\d+\.?\d*)/i);
  const weightMatch2 = text.match(/weight[:\s]*(\d+\.?\d*)\s*kg/i);
  if (weightMatch1) {
    parsedData.baseline_weight = parseFloat(weightMatch1[1]);
  } else if (weightMatch2) {
    parsedData.baseline_weight = parseFloat(weightMatch2[1]);
  }
  
  // Weight Date extraction
  const weightDateMatch = textOriginal.match(/Weight\s*Date[:\s]*(\d{4}-\d{2}-\d{2})/i) || 
                          text.match(/weight\s*date[:\s]*(\d{4}-\d{2}-\d{2})/i) ||
                          textOriginal.match(/Weight\s*Date[:\s]*(\d{1,2}\/\d{1,2}\/\d{4})/i);
  if (weightDateMatch) {
    const dateStr = weightDateMatch[1];
    if (dateStr.includes('/')) {
      // DD/MM/YYYY format
      const parts = dateStr.split('/');
      parsedData.baseline_weight_date = new Date(parseInt(parts[2]), parseInt(parts[1]) - 1, parseInt(parts[0])).toISOString();
    } else {
      // YYYY-MM-DD format
      parsedData.baseline_weight_date = new Date(dateStr).toISOString();
    }
  }

  // Height extraction - check for "Height (cm): 165" format
  const heightMatch1 = textOriginal.match(/Height\s*\(?cm\)?[:\s]*(\d+\.?\d*)/i) || text.match(/height\s*\(?cm\)?[:\s]*(\d+\.?\d*)/i);
  const heightMatch2 = text.match(/height[:\s]*(\d+\.?\d*)\s*cm/i);
  if (heightMatch1) {
    parsedData.height = parseFloat(heightMatch1[1]);
  } else if (heightMatch2) {
    parsedData.height = parseFloat(heightMatch2[1]);
  }

  // Blood pressure extraction
  const bpMatch = text.match(/bp[:\s]*(\d+)\/(\d+)/i);
  if (bpMatch) {
    parsedData.systolic_bp = parseInt(bpMatch[1]);
    parsedData.diastolic_bp = parseInt(bpMatch[2]);
  }

  // HbA1c extraction - check for % format
  const hba1cMatch = text.match(/hba1c[:\s]*\(?%\)?[:\s]*(\d+\.?\d*)/i) || text.match(/hba1c[:\s]*(\d+\.?\d*)(?:\s*%)?/i);
  if (hba1cMatch) {
    parsedData.hba1c_percent = parseFloat(hba1cMatch[1]);
    parsedData.baseline_hba1c = parseFloat(hba1cMatch[1]);
  }
  
  // HbA1c Date extraction
  const hba1cDateMatch = textOriginal.match(/HbA1c\s*Date[:\s]*(\d{4}-\d{2}-\d{2})/i) || 
                         text.match(/hba1c\s*date[:\s]*(\d{4}-\d{2}-\d{2})/i) ||
                         textOriginal.match(/HbA1c\s*Date[:\s]*(\d{1,2}\/\d{1,2}\/\d{4})/i);
  if (hba1cDateMatch) {
    const dateStr = hba1cDateMatch[1];
    if (dateStr.includes('/')) {
      const parts = dateStr.split('/');
      parsedData.baseline_hba1c_date = new Date(parseInt(parts[2]), parseInt(parts[1]) - 1, parseInt(parts[0])).toISOString();
    } else {
      parsedData.baseline_hba1c_date = new Date(dateStr).toISOString();
    }
  }
  
  // Fasting Glucose extraction
  const fastingGlucoseMatch = text.match(/fasting\s*glucose[:\s]*(\d+\.?\d*)/i) || 
                              text.match(/baseline\s*fasting\s*glucose[:\s]*(\d+\.?\d*)/i);
  if (fastingGlucoseMatch) {
    parsedData.baseline_fasting_glucose = parseFloat(fastingGlucoseMatch[1]);
  }
  
  // Random Glucose extraction
  const randomGlucoseMatch = text.match(/random\s*glucose[:\s]*(\d+\.?\d*)/i);
  if (randomGlucoseMatch) {
    parsedData.random_glucose = parseFloat(randomGlucoseMatch[1]);
  }

  // Lipid values
  const tcMatch = text.match(/total\s*cholesterol[:\s]*(\d+\.?\d*)/i) || text.match(/tc[:\s]*(\d+\.?\d*)/i);
  if (tcMatch) {
    parsedData.baseline_tc = parseFloat(tcMatch[1]);
  }

  const ldlMatch = text.match(/ldl[:\s]*(\d+\.?\d*)/i);
  if (ldlMatch) {
    parsedData.baseline_ldl = parseFloat(ldlMatch[1]);
  }

  const hdlMatch = text.match(/hdl[:\s]*(\d+\.?\d*)/i);
  if (hdlMatch) {
    parsedData.baseline_hdl = parseFloat(hdlMatch[1]);
  }

  const tgMatch = text.match(/triglycerides?[:\s]*(\d+\.?\d*)/i);
  if (tgMatch) {
    parsedData.baseline_tg = parseFloat(tgMatch[1]);
  }
  
  // Lipid Date extraction
  const lipidDateMatch = textOriginal.match(/Lipid\s*Date[:\s]*(\d{4}-\d{2}-\d{2})/i) || 
                         text.match(/lipid\s*date[:\s]*(\d{4}-\d{2}-\d{2})/i) ||
                         textOriginal.match(/Lipid\s*Date[:\s]*(\d{1,2}\/\d{1,2}\/\d{4})/i);
  if (lipidDateMatch) {
    const dateStr = lipidDateMatch[1];
    if (dateStr.includes('/')) {
      const parts = dateStr.split('/');
      parsedData.baseline_lipid_date = new Date(parseInt(parts[2]), parseInt(parts[1]) - 1, parseInt(parts[0])).toISOString();
    } else {
      parsedData.baseline_lipid_date = new Date(dateStr).toISOString();
    }
  }
  
  // Lipid Treatment extraction
  const lipidTreatmentMatch = textOriginal.match(/Lipid\s*Treatment[:\s]*([^\n]+)/i) || 
                              text.match(/lipid\s*treatment[:\s]*([^\n]+)/i);
  if (lipidTreatmentMatch && !lipidTreatmentMatch[1].toLowerCase().includes('none')) {
    parsedData.lipid_lowering_treatment = lipidTreatmentMatch[1].trim();
  }
  
  // Antihypertensive Medications extraction
  const antihypertensiveMatch = textOriginal.match(/Antihypertensive\s*Meds?[:\s]*([^\n]+)/i) || 
                                text.match(/antihypertensive\s*meds?[:\s]*([^\n]+)/i);
  if (antihypertensiveMatch && !antihypertensiveMatch[1].toLowerCase().includes('none')) {
    parsedData.antihypertensive_medications = antihypertensiveMatch[1].trim();
  }

  // Medical conditions with negation awareness
  const conditions = [];
  
  // Check for negations first - must be specific to the condition
  const hasNegation = (condition) => {
    // Split condition by pipe (OR) and check each variant
    const conditionVariants = condition.split('|');
    
    for (const variant of conditionVariants) {
      const escapedVariant = variant.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      
      // Pattern 1: "Condition: No" or "Condition = No"
      const pattern1 = new RegExp(`${escapedVariant}\\s*[:=]\\s*(?:no|negative|absent|none)\\b`, 'i');
      if (pattern1.test(text) || (textOriginal && pattern1.test(textOriginal))) {
        return true;
      }
      
      // Pattern 2: "No Condition" (but only if it's a direct match, not part of another word)
      const pattern2 = new RegExp(`(?:^|\\s)(?:no|denies|absent|negative|without)\\s+${escapedVariant}\\b`, 'i');
      if (pattern2.test(text)) {
        return true;
      }
      
      // Pattern 3: "Condition not present" or "Condition absent"
      const pattern3 = new RegExp(`${escapedVariant}\\s+(?:not|no)\\s+(?:present|diagnosed|confirmed)`, 'i');
      if (pattern3.test(text)) {
        return true;
      }
    }
    
    return false;
  };

  // T2DM with negation check
  if (!hasNegation('t2dm|type\\s*2\\s*diabetes|diabetes\\s*mellitus')) {
  if (text.includes('type 2 diabetes') || text.includes('t2dm')) {
      parsedData.t2dm = 1;
    parsedData.diabetes_type = 'Type 2';
      conditions.push('Type 2 Diabetes Mellitus');
    }
  } else {
    parsedData.t2dm = 0;
  }
  
  // Type 1 diabetes
  if (text.includes('type 1 diabetes') || text.includes('t1dm')) {
    parsedData.diabetes_type = 'Type 1';
    conditions.push('Type 1 Diabetes Mellitus');
  }
  
  // Prediabetes - check for "Yes"
  if (text.includes('prediabetes') || text.includes('impaired glucose')) {
    if (text.match(/prediabetes:\s*yes/i) || !text.match(/prediabetes:\s*no/i)) {
      parsedData.prediabetes = 1;
      conditions.push('Prediabetes');
    } else {
      parsedData.prediabetes = 0;
    }
  }
  
  // HTN and Hypertension - check separately
  const htnYes = textOriginal.match(/HTN:\s*Yes/i) || text.match(/htn:\s*yes/i);
  const htnNo = textOriginal.match(/HTN:\s*No/i) || text.match(/htn:\s*no/i);
  const htnBorderline = textOriginal.match(/HTN:\s*Borderline/i) || text.match(/htn:\s*borderline/i);
  
  const hypertensionYes = textOriginal.match(/Hypertension:\s*Yes/i) || text.match(/hypertension:\s*yes/i);
  const hypertensionNo = textOriginal.match(/Hypertension:\s*No/i) || text.match(/hypertension:\s*no/i);
  const hypertensionBorderline = textOriginal.match(/Hypertension:\s*Borderline/i) || text.match(/hypertension:\s*borderline/i);
  
  if (htnYes || hypertensionYes) {
    parsedData.htn = 1;
    parsedData.hypertension = 1;
    conditions.push('Hypertension');
  } else if (htnNo || hypertensionNo) {
    parsedData.htn = 0;
    parsedData.hypertension = 0;
  } else if (htnBorderline || hypertensionBorderline) {
    parsedData.htn = 0;
    parsedData.hypertension = 0;
  } else if (text.includes('hypertension') || text.includes('htn')) {
    // If mentioned but no explicit yes/no, check for "Yes" in cardiac section
    if (text.match(/Cardiac[^\n]*Hypertension:\s*Yes/i) || text.match(/cardiac[^\n]*hypertension:\s*yes/i)) {
      parsedData.htn = 1;
      parsedData.hypertension = 1;
      conditions.push('Hypertension');
    } else {
      parsedData.htn = 0;
      parsedData.hypertension = 0;
    }
  }
  
  // Dyslipidaemia - check for "Yes"
  const dyslipidNeg = hasNegation('dyslipid|cholesterol');
  if (!dyslipidNeg && (text.includes('dyslipid') || text.includes('high cholesterol') || text.includes('hyperlipidemia'))) {
    // Check for explicit "Yes" - use original text for case-sensitive matching
    const hasYes = textOriginal.match(/Dyslipidaemia:\s*Yes/i) || textOriginal.match(/Dyslipidemia:\s*Yes/i) ||
                   text.match(/dyslipidaemia:\s*yes/i) || text.match(/dyslipidemia:\s*yes/i);
    const hasNo = text.match(/dyslipidaemia:\s*no/i) || text.match(/dyslipidemia:\s*no/i);
    const inComorbidities = text.includes('comorbidities') && text.includes('dyslipidaemia');
    
    if (hasYes || (!hasNo && inComorbidities)) {
      parsedData.dyslipidaemia = 1;
      conditions.push('Dyslipidaemia');
    } else if (hasNo) {
      parsedData.dyslipidaemia = 0;
    } else {
      // Default to 0 if unclear
      parsedData.dyslipidaemia = 0;
    }
  } else if (dyslipidNeg) {
    parsedData.dyslipidaemia = 0;
  }
  
  // ASCVD with negation check
  const ascvdYes = textOriginal.match(/ASCVD:\s*Yes/i) || text.match(/ascvd:\s*yes/i);
  const ascvdNo = textOriginal.match(/ASCVD:\s*No/i) || text.match(/ascvd:\s*no/i);
  if (ascvdYes) {
    parsedData.ascvd = 1;
    conditions.push('Atherosclerotic Cardiovascular Disease');
  } else if (ascvdNo) {
    parsedData.ascvd = 0;
  } else if (!hasNegation('ascvd|cardiovascular|atherosclerotic')) {
    if (text.includes('ascvd') || text.includes('atherosclerotic cardiovascular')) {
      parsedData.ascvd = 1;
      conditions.push('Atherosclerotic Cardiovascular Disease');
    }
  } else {
    parsedData.ascvd = 0;
  }
  
  // CKD with negation check - also check for stage
  const ckdNo = textOriginal.match(/CKD:\s*No/i) || text.match(/ckd:\s*no/i);
  const ckdStage = textOriginal.match(/CKD:\s*Stage\s*(\d+[ab]?)/i) || text.match(/ckd:\s*stage\s*(\d+[ab]?)/i);
  if (ckdNo) {
    parsedData.ckd = 0;
  } else if (ckdStage) {
    parsedData.ckd = 1;
    conditions.push('Chronic Kidney Disease');
  } else if (!hasNegation('ckd|chronic\\s*kidney')) {
    if (text.includes('ckd') || text.includes('chronic kidney')) {
      parsedData.ckd = 1;
      conditions.push('Chronic Kidney Disease');
    }
  } else {
    parsedData.ckd = 0;
  }
  
  // OSA - check for "Mild", "Moderate", "Severe" or "Yes"
  const osaNeg = hasNegation('osa|sleep\\s*apnoea|sleep\\s*apnea');
  if (!osaNeg && (text.includes('osa') || text.includes('sleep apnoea') || text.includes('sleep apnea') || text.includes('obstructive sleep'))) {
    // Check for severity levels or "Yes" - use original text for case-sensitive matching
    const hasMild = textOriginal.match(/OSA:\s*Mild/i) || text.match(/osa:\s*mild/i);
    const hasModerate = textOriginal.match(/OSA:\s*Moderate/i) || text.match(/osa:\s*moderate/i);
    const hasSevere = textOriginal.match(/OSA:\s*Severe/i) || text.match(/osa:\s*severe/i);
    const hasYes = text.match(/osa:\s*yes/i);
    const hasNo = text.match(/osa:\s*no/i);
    const inComorbidities = text.includes('comorbidities') && (text.includes('osa') || text.includes('sleep apnea') || text.includes('obstructive sleep'));
    
    if (hasMild || hasModerate || hasSevere || hasYes || (!hasNo && inComorbidities)) {
      parsedData.osa = 1;
      conditions.push('Obstructive Sleep Apnea');
    } else if (hasNo) {
      parsedData.osa = 0;
    } else if (text.includes('osa:') || text.includes('sleep apnoea:') || text.includes('sleep apnea:')) {
      // If explicitly mentioned with colon, assume positive unless "no"
      parsedData.osa = 1;
      conditions.push('Obstructive Sleep Apnea');
    } else {
      parsedData.osa = 0;
    }
  } else if (osaNeg) {
    parsedData.osa = 0;
  }
  
  // Sleep Studies
  const sleepStudiesYes = textOriginal.match(/Sleep\s*Studies:\s*Yes/i) || text.match(/sleep\s*studies:\s*yes/i);
  const sleepStudiesNo = textOriginal.match(/Sleep\s*Studies:\s*No/i) || text.match(/sleep\s*studies:\s*no/i);
  if (sleepStudiesYes) {
    parsedData.sleep_studies = 1;
  } else if (sleepStudiesNo) {
    parsedData.sleep_studies = 0;
  }
  
  // CPAP
  const cpapYes = textOriginal.match(/CPAP:\s*Yes/i) || text.match(/cpap:\s*yes/i);
  const cpapNo = textOriginal.match(/CPAP:\s*No/i) || text.match(/cpap:\s*no/i);
  if (cpapYes) {
    parsedData.cpap = 1;
  } else if (cpapNo) {
    parsedData.cpap = 0;
  }
  
  // Ischaemic Heart Disease (IHD)
  const ihdYes = textOriginal.match(/IHD:\s*Yes/i) || text.match(/ihd:\s*yes/i);
  const ihdNo = textOriginal.match(/IHD:\s*No/i) || text.match(/ihd:\s*no/i);
  if (ihdYes) {
    parsedData.ischaemic_heart_disease = 1;
    conditions.push('Ischaemic Heart Disease');
  } else if (ihdNo) {
    parsedData.ischaemic_heart_disease = 0;
  } else if (text.includes('ischaemic heart disease') || text.includes('ischemic heart disease')) {
    // Check if it's in comorbidities or mentioned positively
    if (text.includes('comorbidities') && (text.includes('ischaemic heart disease') || text.includes('ischemic heart disease'))) {
      parsedData.ischaemic_heart_disease = 1;
      conditions.push('Ischaemic Heart Disease');
    } else if (!text.match(/ischaemic heart disease:\s*no/i) && !text.match(/ischemic heart disease:\s*no/i)) {
      parsedData.ischaemic_heart_disease = 1;
      conditions.push('Ischaemic Heart Disease');
    } else {
      parsedData.ischaemic_heart_disease = 0;
    }
  }
  
  // Heart Failure
  const hfYes = textOriginal.match(/Heart\s*Failure:\s*Yes/i) || text.match(/heart\s*failure:\s*yes/i);
  const hfNo = textOriginal.match(/Heart\s*Failure:\s*No/i) || text.match(/heart\s*failure:\s*no/i);
  if (hfYes) {
    parsedData.heart_failure = 1;
    conditions.push('Heart Failure');
  } else if (hfNo) {
    parsedData.heart_failure = 0;
  }
  
  // Cerebrovascular Disease (CVD)
  const cvdYes = textOriginal.match(/CVD:\s*Yes/i) || text.match(/cvd:\s*yes/i);
  const cvdNo = textOriginal.match(/CVD:\s*No/i) || text.match(/cvd:\s*no/i);
  if (cvdYes) {
    parsedData.cerebrovascular_disease = 1;
    conditions.push('Cerebrovascular Disease');
  } else if (cvdNo) {
    parsedData.cerebrovascular_disease = 0;
  } else if (text.includes('cerebrovascular disease') || text.includes('cvd')) {
    // Check if it's in comorbidities or mentioned positively
    if (text.includes('comorbidities') && text.includes('cerebrovascular disease')) {
      parsedData.cerebrovascular_disease = 1;
      conditions.push('Cerebrovascular Disease');
    } else if (!text.match(/cerebrovascular disease:\s*no/i) && !text.match(/cvd:\s*no/i)) {
      parsedData.cerebrovascular_disease = 1;
      conditions.push('Cerebrovascular Disease');
    } else {
      parsedData.cerebrovascular_disease = 0;
    }
  }
  
  // DVT
  const dvtYes = textOriginal.match(/DVT:\s*Yes/i) || text.match(/dvt:\s*yes/i);
  const dvtNo = textOriginal.match(/DVT:\s*No/i) || text.match(/dvt:\s*no/i);
  if (dvtYes) {
    parsedData.dvt = 1;
  } else if (dvtNo) {
    parsedData.dvt = 0;
  }
  
  // PE
  const peYes = textOriginal.match(/PE:\s*Yes/i) || text.match(/pe:\s*yes/i);
  const peNo = textOriginal.match(/PE:\s*No/i) || text.match(/pe:\s*no/i);
  if (peYes) {
    parsedData.pe = 1;
  } else if (peNo) {
    parsedData.pe = 0;
  }
  
  // Pulmonary Hypertension
  const phYes = textOriginal.match(/Pulmonary\s*HTN:\s*Yes/i) || text.match(/pulmonary\s*htn:\s*yes/i);
  const phNo = textOriginal.match(/Pulmonary\s*HTN:\s*No/i) || text.match(/pulmonary\s*htn:\s*no/i);
  if (phYes) {
    parsedData.pulmonary_hypertension = 1;
  } else if (phNo) {
    parsedData.pulmonary_hypertension = 0;
  }
  
  // Asthma
  const asthmaYes = textOriginal.match(/Asthma:\s*Yes/i) || text.match(/asthma:\s*yes/i);
  const asthmaNo = textOriginal.match(/Asthma:\s*No/i) || text.match(/asthma:\s*no/i);
  if (asthmaYes) {
    parsedData.asthma = 1;
  } else if (asthmaNo) {
    parsedData.asthma = 0;
  }
  
  // Kidney Stones
  const kidneyStonesYes = textOriginal.match(/Kidney\s*Stones:\s*Yes/i) || text.match(/kidney\s*stones:\s*yes/i);
  const kidneyStonesNo = textOriginal.match(/Kidney\s*Stones:\s*No/i) || text.match(/kidney\s*stones:\s*no/i);
  if (kidneyStonesYes) {
    parsedData.kidney_stones = 1;
  } else if (kidneyStonesNo) {
    parsedData.kidney_stones = 0;
  }
  
  // Thyroid Disorder
  const thyroidYes = textOriginal.match(/Thyroid\s*Disorder:\s*(Yes|Hypothyroidism|Hyperthyroidism)/i) || 
                     text.match(/thyroid\s*disorder:\s*(yes|hypothyroidism|hyperthyroidism)/i);
  const thyroidNo = textOriginal.match(/Thyroid\s*Disorder:\s*No/i) || text.match(/thyroid\s*disorder:\s*no/i);
  if (thyroidYes) {
    parsedData.thyroid_disorder = 1;
    conditions.push('Thyroid Disorder');
  } else if (thyroidNo) {
    parsedData.thyroid_disorder = 0;
  } else if (text.includes('hypothyroidism') || text.includes('hyperthyroidism') || text.includes('thyroid disorder')) {
    // Check if mentioned in comorbidities or medications (levothyroxine indicates hypothyroidism)
    if (text.includes('comorbidities') && (text.includes('hypothyroidism') || text.includes('thyroid'))) {
      parsedData.thyroid_disorder = 1;
      conditions.push('Thyroid Disorder');
    } else if (text.includes('levothyroxine')) {
      parsedData.thyroid_disorder = 1;
      conditions.push('Thyroid Disorder');
    } else if (!text.match(/thyroid disorder:\s*no/i)) {
      parsedData.thyroid_disorder = 1;
      conditions.push('Thyroid Disorder');
    } else {
      parsedData.thyroid_disorder = 0;
    }
  }
  
  // Infertility
  const infertilityYes = textOriginal.match(/Infertility:\s*Yes/i) || text.match(/infertility:\s*yes/i);
  const infertilityNo = textOriginal.match(/Infertility:\s*No/i) || text.match(/infertility:\s*no/i);
  if (infertilityYes) {
    parsedData.infertility = 1;
  } else if (infertilityNo) {
    parsedData.infertility = 0;
  }
  
  // PCOS
  const pcosYes = textOriginal.match(/PCOS:\s*Yes/i) || text.match(/pcos:\s*yes/i);
  const pcosNo = textOriginal.match(/PCOS:\s*No/i) || text.match(/pcos:\s*no/i);
  if (pcosYes) {
    parsedData.pcos = 1;
  } else if (pcosNo) {
    parsedData.pcos = 0;
  }
  
  // IIH
  const iihYes = textOriginal.match(/IIH:\s*Yes/i) || text.match(/iih:\s*yes/i);
  const iihNo = textOriginal.match(/IIH:\s*No/i) || text.match(/iih:\s*no/i);
  if (iihYes) {
    parsedData.iih = 1;
  } else if (iihNo) {
    parsedData.iih = 0;
  }
  
  // Epilepsy
  const epilepsyYes = textOriginal.match(/Epilepsy:\s*Yes/i) || text.match(/epilepsy:\s*yes/i);
  const epilepsyNo = textOriginal.match(/Epilepsy:\s*No/i) || text.match(/epilepsy:\s*no/i);
  if (epilepsyYes) {
    parsedData.epilepsy = 1;
  } else if (epilepsyNo) {
    parsedData.epilepsy = 0;
  }
  
  // FND
  const fndYes = textOriginal.match(/FND:\s*Yes/i) || text.match(/fnd:\s*yes/i);
  const fndNo = textOriginal.match(/FND:\s*No/i) || text.match(/fnd:\s*no/i);
  if (fndYes) {
    parsedData.functional_neurological_disorder = 1;
  } else if (fndNo) {
    parsedData.functional_neurological_disorder = 0;
  }
  
  // Cancer
  const cancerYes = textOriginal.match(/Cancer:\s*(Yes|.*reported)/i) && !textOriginal.match(/Cancer:\s*No/i);
  const cancerNo = textOriginal.match(/Cancer:\s*No/i) || text.match(/cancer:\s*none reported/i);
  if (cancerYes && !cancerNo) {
    parsedData.cancer = 1;
  } else if (cancerNo) {
    parsedData.cancer = 0;
  }
  
  // Bariatric procedures
  const gastricBandYes = textOriginal.match(/Gastric\s*Band:\s*Yes/i) || text.match(/gastric\s*band:\s*yes/i);
  const gastricBandNo = textOriginal.match(/Gastric\s*Band:\s*No/i) || text.match(/gastric\s*band:\s*no/i);
  if (gastricBandYes) {
    parsedData.bariatric_gastric_band = 1;
  } else if (gastricBandNo) {
    parsedData.bariatric_gastric_band = 0;
  }
  
  const sleeveYes = textOriginal.match(/Sleeve:\s*Yes/i) || text.match(/sleeve:\s*yes/i);
  const sleeveNo = textOriginal.match(/Sleeve:\s*No/i) || text.match(/sleeve:\s*no/i);
  if (sleeveYes) {
    parsedData.bariatric_sleeve = 1;
  } else if (sleeveNo) {
    parsedData.bariatric_sleeve = 0;
  }
  
  const bypassYes = textOriginal.match(/Bypass:\s*Yes/i) || text.match(/bypass:\s*yes/i);
  const bypassNo = textOriginal.match(/Bypass:\s*No/i) || text.match(/bypass:\s*no/i);
  if (bypassYes) {
    parsedData.bariatric_bypass = 1;
  } else if (bypassNo) {
    parsedData.bariatric_bypass = 0;
  }
  
  const balloonYes = textOriginal.match(/Balloon:\s*Yes/i) || text.match(/balloon:\s*yes/i);
  const balloonNo = textOriginal.match(/Balloon:\s*No/i) || text.match(/balloon:\s*no/i);
  if (balloonYes) {
    parsedData.bariatric_balloon = 1;
  } else if (balloonNo) {
    parsedData.bariatric_balloon = 0;
  }
  
  // Anxiety - check for "Mild", "Moderate" or "Yes"
  const anxietyMild = textOriginal.match(/Anxiety:\s*Mild/i) || text.match(/anxiety:\s*mild/i);
  const anxietyModerate = textOriginal.match(/Anxiety:\s*Moderate/i) || text.match(/anxiety:\s*moderate/i);
  const anxietyYes = textOriginal.match(/Anxiety:\s*Yes/i) || text.match(/anxiety:\s*yes/i);
  const anxietyNo = textOriginal.match(/Anxiety:\s*No/i) || text.match(/anxiety:\s*no/i);
  if (anxietyMild || anxietyModerate || anxietyYes || text.includes('mild intermittent')) {
    parsedData.anxiety = 1;
    conditions.push('Anxiety');
  } else if (anxietyNo) {
    parsedData.anxiety = 0;
  } else if (text.includes('anxiety') && !hasNegation('anxiety')) {
    parsedData.anxiety = 1;
    conditions.push('Anxiety');
  } else {
    parsedData.anxiety = 0;
  }
  
  // Depression
  const depressionYes = textOriginal.match(/Depression:\s*Yes/i) || text.match(/depression:\s*yes/i);
  const depressionNo = textOriginal.match(/Depression:\s*No/i) || text.match(/depression:\s*no/i);
  if (depressionYes) {
    parsedData.depression = 1;
    conditions.push('Depression');
  } else if (depressionNo) {
    parsedData.depression = 0;
  }
  
  // Bipolar
  const bipolarYes = textOriginal.match(/Bipolar:\s*Yes/i) || text.match(/bipolar:\s*yes/i);
  const bipolarNo = textOriginal.match(/Bipolar:\s*No/i) || text.match(/bipolar:\s*no/i);
  if (bipolarYes) {
    parsedData.bipolar_disorder = 1;
  } else if (bipolarNo) {
    parsedData.bipolar_disorder = 0;
  }
  
  // Schizoaffective
  const schizoaffectiveYes = textOriginal.match(/Schizoaffective:\s*Yes/i) || text.match(/schizoaffective:\s*yes/i);
  const schizoaffectiveNo = textOriginal.match(/Schizoaffective:\s*No/i) || text.match(/schizoaffective:\s*no/i);
  if (schizoaffectiveYes) {
    parsedData.schizoaffective_disorder = 1;
  } else if (schizoaffectiveNo) {
    parsedData.schizoaffective_disorder = 0;
  }
  
  // OA Knee
  const oaKneeYes = textOriginal.match(/OA\s*Knee:\s*Yes/i) || text.match(/oa\s*knee:\s*yes/i);
  const oaKneeNo = textOriginal.match(/OA\s*Knee:\s*No/i) || text.match(/oa\s*knee:\s*no/i);
  if (oaKneeYes) {
    parsedData.oa_knee = 1;
    conditions.push('Osteoarthritis Knee');
  } else if (oaKneeNo) {
    parsedData.oa_knee = 0;
  }
  
  // OA Hip
  const oaHipYes = textOriginal.match(/OA\s*Hip:\s*Yes/i) || text.match(/oa\s*hip:\s*yes/i);
  const oaHipNo = textOriginal.match(/OA\s*Hip:\s*No/i) || text.match(/oa\s*hip:\s*no/i);
  if (oaHipYes) {
    parsedData.oa_hip = 1;
  } else if (oaHipNo) {
    parsedData.oa_hip = 0;
  }
  
  // Limited Mobility
  const limitedMobilityYes = textOriginal.match(/Limited\s*Mobility:\s*Yes/i) || text.match(/limited\s*mobility:\s*yes/i);
  const limitedMobilityNo = textOriginal.match(/Limited\s*Mobility:\s*No/i) || text.match(/limited\s*mobility:\s*no/i);
  if (limitedMobilityYes) {
    parsedData.limited_mobility = 1;
    conditions.push('Limited Mobility');
  } else if (limitedMobilityNo) {
    parsedData.limited_mobility = 0;
  }
  
  // Lymphoedema
  const lymphoedemaYes = textOriginal.match(/Lymphoedema:\s*Yes/i) || text.match(/lymphoedema:\s*yes/i);
  const lymphoedemaNo = textOriginal.match(/Lymphoedema:\s*No/i) || text.match(/lymphoedema:\s*no/i);
  if (lymphoedemaYes) {
    parsedData.lymphoedema = 1;
  } else if (lymphoedemaNo) {
    parsedData.lymphoedema = 0;
  }
  
  // Emotional Eating - check for "Mild" or "Yes"
  const emotionalNeg = hasNegation('emotional\\s*eating');
  if (!emotionalNeg && (text.includes('emotional eating') || text.includes('emotional eating:'))) {
    // Use original text for case-sensitive matching
    const hasMild = textOriginal.match(/Emotional Eating:\s*Mild/i) || text.match(/emotional eating:\s*mild/i) ||
                    text.match(/emotional.*eating.*mild/i);
    const hasYes = text.match(/emotional eating:\s*yes/i);
    const hasNo = text.match(/emotional eating:\s*no/i);
    
    if (hasMild || hasYes || (text.includes('emotional eating:') && !hasNo)) {
      parsedData.emotional_eating = 1;
      conditions.push('Emotional Eating');
    } else if (hasNo) {
      parsedData.emotional_eating = 0;
    } else {
      parsedData.emotional_eating = 0;
    }
  } else if (emotionalNeg) {
    parsedData.emotional_eating = 0;
  }
  
  // GORD - check for "Occasional" or "Yes"
  const gordNeg = hasNegation('gord|gerd|gastroesophageal\\s*reflux');
  if (!gordNeg && (text.includes('gord') || text.includes('gerd') || text.includes('gastroesophageal reflux'))) {
    // Use original text for case-sensitive matching
    const hasOccasional = textOriginal.match(/GORD:\s*Occasional/i) || text.match(/gord:\s*occasional/i) ||
                          text.match(/gerd:\s*occasional/i) || text.match(/gastroesophageal.*reflux.*occasional/i);
    const hasYes = text.match(/gord:\s*yes/i);
    const hasNo = text.match(/gord:\s*no/i);
    
    if (hasOccasional || hasYes || ((text.includes('gord:') || text.includes('gerd:')) && !hasNo)) {
      parsedData.gord = 1;
      conditions.push('Gastroesophageal Reflux Disease');
    } else if (hasNo) {
      parsedData.gord = 0;
    } else {
      parsedData.gord = 0;
    }
  } else if (gordNeg) {
    parsedData.gord = 0;
  }
  
  // MASLD - check for "possible" vs confirmed
  if (text.includes('masld') || text.includes('nafld')) {
    if (text.includes('possible masld') || text.includes('possible nafld')) {
      parsedData.masld = 0; // Possible = 0
    } else if (text.includes('masld confirmed') || text.includes('nafld confirmed')) {
      parsedData.masld = 1; // Confirmed = 1
      conditions.push('MASLD');
    } else if (!hasNegation('masld|nafld')) {
      parsedData.masld = 1; // Mentioned without "possible" = 1
      conditions.push('MASLD');
    } else {
      parsedData.masld = 0;
    }
  }

  // Medications extraction
  const medications = [];
  const medMatches = text.match(/(metformin|lisinopril|atorvastatin|insulin|glipizide|glimepiride|semaglutide|liraglutide|dulaglutide|exenatide|sitagliptin|saxagliptin|linagliptin|alogliptin|dapagliflozin|empagliflozin|canagliflozin|pioglitazone|rosiglitazone|gliclazide|tolbutamide|chlorpropamide|acarbose|miglitol|bromocriptine|colesevelam|pramlintide|nateglinide|repaglinide)/gi);
  if (medMatches) {
    medications.push(...medMatches.map(med => med.toLowerCase()));
  }
  parsedData.medications = medications;

  // Count comorbidities (only count fields = 1)
  const comorbidityFields = ['t2dm', 'prediabetes', 'htn', 'hypertension', 'dyslipidaemia', 'ascvd', 'ckd', 'osa', 'masld', 
    'anxiety', 'depression', 'bipolar_disorder', 'emotional_eating', 'schizoaffective_disorder',
    'oa_knee', 'oa_hip', 'limited_mobility', 'lymphoedema', 'thyroid_disorder', 'iih', 'epilepsy',
    'functional_neurological_disorder', 'cancer', 'ischaemic_heart_disease', 'heart_failure',
    'cerebrovascular_disease', 'pulmonary_hypertension', 'dvt', 'pe', 'gord', 'kidney_stones',
    'infertility', 'pcos', 'bariatric_gastric_band', 'bariatric_sleeve', 'bariatric_bypass', 'bariatric_balloon'];
  const comorbidityCount = comorbidityFields.filter(field => parsedData[field] === 1).length;
  parsedData.total_qualifying_comorbidities = comorbidityCount;
  
  // Set conditions array
  parsedData.conditions = conditions;
  
  // Extract notes from clinical notes section
  const notesMatch = textOriginal.match(/Clinical\s*Notes[:\s]*([^\n]+(?:\n(?!Last\s*Visit)[^\n]+)*)/i) || 
                    text.match(/clinical\s*notes[:\s]*([^\n]+(?:\n(?!last\s*visit)[^\n]+)*)/i);
  if (notesMatch) {
    parsedData.notes = notesMatch[1].trim();
  }
  
  // Extract total qualifying comorbidities if mentioned
  const comorbiditiesMatch = textOriginal.match(/Total\s*Qualifying\s*Comorbidities[:\s]*(\d+)/i) || 
                             text.match(/total\s*qualifying\s*comorbidities[:\s]*(\d+)/i);
  if (comorbiditiesMatch) {
    parsedData.total_qualifying_comorbidities = parseInt(comorbiditiesMatch[1]);
  }

  console.log('🔄 Basic parser extracted:', Object.keys(parsedData).length, 'fields');
  
  // Validate and normalize to ensure full schema
  try {
    return validateAndNormalizePatientData(parsedData);
  } catch (validationError) {
    console.error('❌ Basic parser validation failed:', validationError.message);
    // Return parsed data anyway, but log the error
    return validateAndNormalizePatientData({}); // Return empty schema if validation fails
  }
}

module.exports = { runOllamaParser, basicTextParser };

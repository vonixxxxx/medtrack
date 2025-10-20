const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

/**
 * Intelligent Medical Parser with Patient Matching
 * Uses local LLM to extract structured data and automatically match patients
 */

const SYSTEM_PROMPT = `You are a clinical data parser. Given a medical note, extract and map each data element to the structured fields in a patient's EMR schema.

CRITICAL REQUIREMENTS:
1. Identify the patient name and extract it as "Patient_Name"
2. Extract all relevant clinical data with high accuracy
3. Return structured JSON adhering to this exact schema:

{
    "Patient_Name": "string",
    "Sex": "string",
    "Age": "number",
    "Conditions": ["string array"],
    "Medications": [{"name": "string", "dose": "string", "frequency": "string"}],
    "Allergies": ["string array"],
    "Labs": [{"metric": "string", "value": "number", "unit": "string", "date": "string"}],
    "Vitals": [{"type": "string", "value": "number", "unit": "string"}],
    "Impression": "string",
    "Plan": "string",
    "Date_Of_Entry": "string"
}

EXTRACTION RULES:
- Patient_Name: Extract the full name mentioned in the note
- Sex: "Male", "Female", or "Other"
- Age: Extract as number
- Conditions: List all medical conditions mentioned
- Medications: Extract name, dose, and frequency for each medication
- Allergies: List all allergies mentioned
- Labs: Extract metric name, value, unit, and date if available
- Vitals: Extract vital signs with values and units
- Impression: Clinical impression or assessment
- Plan: Treatment plan or next steps
- Date_Of_Entry: Date of the note (use current date if not specified)

Return ONLY valid JSON. No additional text or formatting.`;

/**
 * Run intelligent medical parsing with Ollama
 * @param {string} medicalNotes - Raw medical history text
 * @returns {Object} - Structured medical data with patient matching info
 */
async function runIntelligentMedicalParser(medicalNotes) {
  try {
    console.log('🤖 Starting intelligent medical parsing...');
    
    const prompt = `${SYSTEM_PROMPT}\n\nMedical Note:\n"""${medicalNotes}"""`;
    
    const response = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llama3.2:latest",
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.1,
          top_p: 0.9,
          max_tokens: 2048
        }
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama request failed: ${response.status}`);
    }

    const data = await response.json();
    const rawOutput = data.response || "";
    
    console.log('📝 Raw LLM output:', rawOutput);
    
    // Extract JSON from response
    const jsonMatch = rawOutput.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No valid JSON found in LLM response');
    }
    
    const extractedData = JSON.parse(jsonMatch[0]);
    console.log('✅ Successfully parsed medical data:', extractedData);
    
    return {
      success: true,
      data: extractedData,
      confidence: 0.9, // High confidence for structured extraction
      model_used: "llama3.2:latest"
    };
    
  } catch (error) {
    console.error('❌ Intelligent parsing error:', error);
    
    // Fallback to basic parsing
    console.log('🔄 Falling back to basic text parsing...');
    return runBasicMedicalParser(medicalNotes);
  }
}

/**
 * Basic fallback parser using keyword matching
 * @param {string} medicalNotes - Raw medical history text
 * @returns {Object} - Basic structured data
 */
function runBasicMedicalParser(medicalNotes) {
  const text = medicalNotes.toLowerCase();
  
  // Extract basic demographics
  const ageMatch = text.match(/(\d+)\s*(?:y\/o|years?|yo)/);
  const sexMatch = text.match(/\b(male|female|m|f)\b/);
  
  // Extract conditions using keyword matching
  const conditions = [];
  const conditionKeywords = {
    'Type 2 Diabetes': ['t2dm', 'type 2 diabetes', 'diabetes mellitus', 'dm'],
    'Hypertension': ['htn', 'hypertension', 'high blood pressure'],
    'Obesity': ['obesity', 'obese', 'bmi'],
    'CKD': ['ckd', 'chronic kidney disease', 'kidney disease'],
    'Dyslipidemia': ['dyslipid', 'dyslipidaemia', 'high cholesterol'],
    'CAD': ['cad', 'coronary artery disease', 'heart disease'],
    'Asthma': ['asthma'],
    'COPD': ['copd', 'chronic obstructive pulmonary disease']
  };
  
  Object.entries(conditionKeywords).forEach(([condition, keywords]) => {
    if (keywords.some(keyword => text.includes(keyword))) {
      conditions.push(condition);
    }
  });
  
  // Extract medications
  const medications = [];
  const medKeywords = {
    'Metformin': ['metf', 'metformin'],
    'Lisinopril': ['lisin', 'lisinopril'],
    'Atorvastatin': ['atorva', 'atorvastatin'],
    'Insulin': ['insulin'],
    'Aspirin': ['aspirin', 'asa']
  };
  
  Object.entries(medKeywords).forEach(([med, keywords]) => {
    if (keywords.some(keyword => text.includes(keyword))) {
      medications.push({
        name: med,
        dose: 'Unknown',
        frequency: 'Unknown'
      });
    }
  });
  
  // Extract lab values
  const labs = [];
  const hba1cMatch = text.match(/a1c[:\s]*(\d+\.?\d*)\s*%/i);
  if (hba1cMatch) {
    labs.push({
      metric: 'HbA1c',
      value: parseFloat(hba1cMatch[1]),
      unit: '%',
      date: new Date().toISOString().split('T')[0]
    });
  }
  
  const ldlMatch = text.match(/ldl[:\s]*(\d+)/i);
  if (ldlMatch) {
    labs.push({
      metric: 'LDL',
      value: parseFloat(ldlMatch[1]),
      unit: 'mg/dL',
      date: new Date().toISOString().split('T')[0]
    });
  }
  
  // Extract vital signs
  const vitals = [];
  const bpMatch = text.match(/bp[:\s]*(\d+)\/(\d+)/i);
  if (bpMatch) {
    vitals.push({
      type: 'blood_pressure',
      value: parseFloat(bpMatch[1]),
      unit: 'mmHg',
      value_secondary: parseFloat(bpMatch[2])
    });
  }
  
  const bmiMatch = text.match(/bmi[:\s]*(\d+\.?\d*)/i);
  if (bmiMatch) {
    vitals.push({
      type: 'bmi',
      value: parseFloat(bmiMatch[1]),
      unit: 'kg/m²'
    });
  }
  
  return {
    success: true,
    data: {
      Patient_Name: 'Unknown Patient',
      Sex: sexMatch ? sexMatch[1] : 'Unknown',
      Age: ageMatch ? parseInt(ageMatch[1]) : null,
      Conditions: conditions,
      Medications: medications,
      Allergies: [],
      Labs: labs,
      Vitals: vitals,
      Impression: 'Basic parsing completed',
      Plan: 'Review and update as needed',
      Date_Of_Entry: new Date().toISOString().split('T')[0]
    },
    confidence: 0.6,
    model_used: "basic_keyword_parser"
  };
}

/**
 * Find patient by name with fuzzy matching
 * @param {string} patientName - Name to search for
 * @param {Array} patients - Array of patient objects
 * @returns {Object} - Matching result with confidence score
 */
function findPatientByName(patientName, patients) {
  if (!patientName || !patients || patients.length === 0) {
    return { found: false, matches: [], confidence: 0 };
  }
  
  const searchName = patientName.toLowerCase().trim();
  const matches = [];
  
  patients.forEach(patient => {
    const patientNames = [
      patient.user?.name?.toLowerCase(),
      patient.user?.email?.toLowerCase(),
      patient.nhs_number,
      patient.mrn
    ].filter(Boolean);
    
    patientNames.forEach(name => {
      if (name && name.includes(searchName)) {
        matches.push({
          patient,
          confidence: name === searchName ? 1.0 : 0.8,
          matchType: 'exact' // or 'partial'
        });
      }
    });
  });
  
  // Sort by confidence
  matches.sort((a, b) => b.confidence - a.confidence);
  
  return {
    found: matches.length > 0,
    matches,
    confidence: matches.length > 0 ? matches[0].confidence : 0
  };
}

/**
 * Map extracted data to database fields
 * @param {Object} extractedData - Data from LLM parsing
 * @returns {Object} - Mapped data for database update
 */
function mapExtractedDataToDatabase(extractedData) {
  const mappedData = {};
  
  // Basic demographics
  if (extractedData.Sex) {
    mappedData.sex = extractedData.Sex.toLowerCase();
  }
  if (extractedData.Age) {
    // Calculate DOB from age (approximate)
    const currentYear = new Date().getFullYear();
    const birthYear = currentYear - extractedData.Age;
    mappedData.dob = new Date(birthYear, 0, 1);
  }
  
  // Conditions mapping
  const conditions = extractedData.Conditions || [];
  conditions.forEach(condition => {
    const conditionLower = condition.toLowerCase();
    
    if (conditionLower.includes('diabetes') || conditionLower.includes('t2dm')) {
      mappedData.t2dm = true;
    }
    if (conditionLower.includes('hypertension') || conditionLower.includes('htn')) {
      mappedData.htn = true;
      mappedData.hypertension = true;
    }
    if (conditionLower.includes('obesity') || conditionLower.includes('obese')) {
      // BMI will be set from vitals
    }
    if (conditionLower.includes('ckd') || conditionLower.includes('kidney')) {
      mappedData.ckd = true;
    }
    if (conditionLower.includes('dyslipid') || conditionLower.includes('cholesterol')) {
      mappedData.dyslipidaemia = true;
    }
  });
  
  // Lab results
  const labs = extractedData.Labs || [];
  labs.forEach(lab => {
    const metric = lab.metric.toLowerCase();
    const value = lab.value;
    
    if (metric.includes('hba1c') || metric.includes('a1c')) {
      mappedData.baseline_hba1c = value;
      mappedData.hba1c_percent = value;
      console.log('🔍 Lab date debug:', { metric, date: lab.date, type: typeof lab.date });
      if (lab.date && lab.date !== 'Invalid Date' && lab.date !== 'null' && lab.date !== 'undefined') {
        const date = new Date(lab.date);
        console.log('🔍 Date parsing:', { original: lab.date, parsed: date, isValid: !isNaN(date.getTime()) });
        if (!isNaN(date.getTime()) && date.getTime() > 0) {
          mappedData.baseline_hba1c_date = date;
        }
      }
    }
    if (metric.includes('ldl')) {
      mappedData.baseline_ldl = value;
    }
    if (metric.includes('hdl')) {
      mappedData.baseline_hdl = value;
    }
    if (metric.includes('triglyceride') || metric.includes('tg')) {
      mappedData.baseline_tg = value;
    }
    if (metric.includes('total cholesterol') || metric.includes('tc')) {
      mappedData.baseline_tc = value;
    }
  });
  
  // Vital signs
  const vitals = extractedData.Vitals || [];
  vitals.forEach(vital => {
    const type = vital.type.toLowerCase();
    
    if (type.includes('blood_pressure') || type.includes('bp')) {
      mappedData.systolic_bp = vital.value;
      if (vital.value_secondary) {
        mappedData.diastolic_bp = vital.value_secondary;
      }
    }
    if (type.includes('bmi')) {
      mappedData.baseline_bmi = vital.value;
    }
  });
  
  // Medications
  const medications = extractedData.Medications || [];
  if (medications.length > 0) {
    mappedData.all_medications_from_scr = medications.map(med => 
      `${med.name} ${med.dose} ${med.frequency || ''}`.trim()
    ).join(', ');
  }
  
  // Notes
  if (extractedData.Impression) {
    mappedData.notes = extractedData.Impression;
  }
  
  // Filter out invalid dates and null values
  const filteredData = {};
  Object.entries(mappedData).forEach(([key, value]) => {
    if (value instanceof Date) {
      if (!isNaN(value.getTime()) && value.getTime() > 0) {
        filteredData[key] = value;
      } else {
        console.log(`⚠️ Filtering out invalid date for ${key}:`, value);
      }
    } else if (value !== null && value !== undefined && value !== 'Invalid Date') {
      filteredData[key] = value;
    }
  });
  
  console.log('🔍 Filtered data:', filteredData);
  return filteredData;
}

module.exports = {
  runIntelligentMedicalParser,
  findPatientByName,
  mapExtractedDataToDatabase
};

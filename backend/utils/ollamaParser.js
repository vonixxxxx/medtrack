const fetch = require('node-fetch');

/**
 * Parse medical history using local Ollama model
 * @param {string} medicalNotes - Raw medical history text
 * @returns {Object} - Structured medical data
 */
async function runOllamaParser(medicalNotes) {
  try {
    console.log('🤖 Starting Ollama parsing...');
    
    const prompt = `You are a medical NLP system. Extract all structured data from this unstructured medical text and return ONLY a valid JSON object.

Extract and map the following information:
- Demographics: age, sex, name
- Physical measurements: height, weight, BMI, blood pressure
- Lab values: HbA1c (as percentage), total cholesterol, HDL, LDL, triglycerides, creatinine, eGFR
- Medical conditions: diabetes type, hypertension, dyslipidemia, ASCVD, OSA, CKD, etc.
- Medications: list all medications with dosages
- Dates: lab dates, visit dates
- Other: notes, comorbidities count

IMPORTANT: Return ONLY valid JSON. No explanations or markdown. Use these exact field names:
{
  "age": number,
  "sex": "Male" or "Female",
  "height": number (cm),
  "weight": number (kg),
  "bmi": number,
  "systolic_bp": number,
  "diastolic_bp": number,
  "hba1c_percent": number,
  "baseline_hba1c": number,
  "baseline_tc": number,
  "baseline_hdl": number,
  "baseline_ldl": number,
  "baseline_tg": number,
  "creatinine": number,
  "egfr": number,
  "t2dm": boolean,
  "prediabetes": boolean,
  "diabetes_type": "Type 1" or "Type 2",
  "htn": boolean,
  "hypertension": boolean,
  "dyslipidaemia": boolean,
  "ascvd": boolean,
  "ckd": boolean,
  "osa": boolean,
  "obesity": boolean,
  "medications": ["medication1", "medication2"],
  "notes": "string",
  "comorbidities_count": number
}

Medical text:
"""${medicalNotes}"""`;

    const response = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llama3.2:latest", // Using the available model
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.1, // Low temperature for consistent parsing
          top_p: 0.9
        }
      }),
    });

    if (!response.ok) {
      console.error(`Ollama request failed: ${response.status} ${response.statusText}`);
      throw new Error(`Ollama request failed: ${response.status}`);
    }

    const data = await response.json();
    const rawOutput = data.response || "";
    
    console.log('🤖 Raw Ollama output:', rawOutput.substring(0, 200) + '...');
    
    // Extract JSON from the response
    const jsonMatch = rawOutput.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      console.error('No JSON found in Ollama response');
      return {};
    }
    
    const jsonString = jsonMatch[0];
    console.log('🤖 Extracted JSON:', jsonString);
    
    const parsedData = JSON.parse(jsonString);
    console.log('✅ Successfully parsed medical data:', Object.keys(parsedData));
    
    return parsedData;
    
  } catch (error) {
    console.error('❌ Ollama parsing error:', error);
    
    // Fallback to basic text parsing if Ollama fails
    console.log('🔄 Falling back to basic text parsing...');
    return await basicTextParser(medicalNotes);
  }
}

/**
 * Fallback basic text parser when Ollama is not available
 */
async function basicTextParser(medicalNotes) {
  const text = medicalNotes.toLowerCase();
  const parsedData = {};

  // Age extraction
  const ageMatch = text.match(/(\d+)\s*y[eo]?[oa]?/);
  if (ageMatch) {
    parsedData.age = parseInt(ageMatch[1]);
  }

  // Sex extraction
  if (text.includes('male') || text.includes('m/')) {
    parsedData.sex = 'Male';
  } else if (text.includes('female') || text.includes('f/')) {
    parsedData.sex = 'Female';
  }

  // BMI extraction
  const bmiMatch = text.match(/bmi[:\s]*(\d+\.?\d*)/i);
  if (bmiMatch) {
    parsedData.bmi = parseFloat(bmiMatch[1]);
  }

  // Weight extraction
  const weightMatch = text.match(/weight[:\s]*(\d+\.?\d*)\s*kg/i);
  if (weightMatch) {
    parsedData.weight = parseFloat(weightMatch[1]);
  }

  // Height extraction
  const heightMatch = text.match(/height[:\s]*(\d+\.?\d*)\s*cm/i);
  if (heightMatch) {
    parsedData.height = parseFloat(heightMatch[1]);
  }

  // Blood pressure extraction
  const bpMatch = text.match(/bp[:\s]*(\d+)\/(\d+)/i);
  if (bpMatch) {
    parsedData.systolic_bp = parseInt(bpMatch[1]);
    parsedData.diastolic_bp = parseInt(bpMatch[2]);
  }

  // HbA1c extraction
  const hba1cMatch = text.match(/hba1c[:\s]*(\d+\.?\d*)(?:\s*%)?/i);
  if (hba1cMatch) {
    parsedData.hba1c_percent = parseFloat(hba1cMatch[1]);
    parsedData.baseline_hba1c = parseFloat(hba1cMatch[1]);
  }

  // Lipid values
  const tcMatch = text.match(/ldl[:\s]*(\d+\.?\d*)/i);
  if (tcMatch) {
    parsedData.baseline_ldl = parseFloat(tcMatch[1]);
  }

  const hdlMatch = text.match(/hdl[:\s]*(\d+\.?\d*)/i);
  if (hdlMatch) {
    parsedData.baseline_hdl = parseFloat(hdlMatch[1]);
  }

  const tgMatch = text.match(/triglycerides?[:\s]*(\d+\.?\d*)/i);
  if (tgMatch) {
    parsedData.baseline_tg = parseFloat(tgMatch[1]);
  }

  // Medical conditions
  if (text.includes('type 2 diabetes') || text.includes('t2dm')) {
    parsedData.t2dm = true;
    parsedData.diabetes_type = 'Type 2';
  }
  if (text.includes('type 1 diabetes') || text.includes('t1dm')) {
    parsedData.diabetes_type = 'Type 1';
  }
  if (text.includes('prediabetes')) {
    parsedData.prediabetes = true;
  }
  if (text.includes('hypertension') || text.includes('htn')) {
    parsedData.htn = true;
    parsedData.hypertension = true;
  }
  if (text.includes('dyslipid') || text.includes('cholesterol')) {
    parsedData.dyslipidaemia = true;
  }
  if (text.includes('ascvd') || text.includes('cardiovascular')) {
    parsedData.ascvd = true;
  }
  if (text.includes('ckd') || text.includes('chronic kidney')) {
    parsedData.ckd = true;
  }
  if (text.includes('osa') || text.includes('sleep apnoea')) {
    parsedData.osa = true;
  }
  if (text.includes('obesity') || (parsedData.bmi && parsedData.bmi > 30)) {
    parsedData.obesity = true;
  }

  // Medications extraction
  const medications = [];
  const medMatches = text.match(/(metformin|lisinopril|atorvastatin|insulin|glipizide|glimepiride|semaglutide|liraglutide|dulaglutide|exenatide|sitagliptin|saxagliptin|linagliptin|alogliptin|dapagliflozin|empagliflozin|canagliflozin|pioglitazone|rosiglitazone|gliclazide|tolbutamide|chlorpropamide|acarbose|miglitol|bromocriptine|colesevelam|pramlintide|nateglinide|repaglinide)/gi);
  if (medMatches) {
    medications.push(...medMatches.map(med => med.toLowerCase()));
  }
  parsedData.medications = medications;

  // Count comorbidities
  const conditions = ['t2dm', 'htn', 'dyslipidaemia', 'ascvd', 'ckd', 'osa', 'obesity'];
  const comorbidityCount = conditions.filter(condition => parsedData[condition]).length;
  parsedData.comorbidities_count = comorbidityCount;

  console.log('🔄 Basic parser extracted:', Object.keys(parsedData));
  return parsedData;
}

module.exports = { runOllamaParser, basicTextParser };

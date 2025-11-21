const express = require('express');
const cors = require('cors');
const { PrismaClient } = require('@prisma/client');
require('dotenv').config();

// Import intelligent medical parser
const { 
  runIntelligentMedicalParser, 
  findPatientByName, 
  mapExtractedDataToDatabase 
} = require('./utils/intelligentMedicalParser.js');

const app = express();

// Initialize Prisma with singleton pattern for production
const prisma = global.prisma || new PrismaClient();
if (process.env.NODE_ENV !== 'production') global.prisma = prisma;

// Simple in-memory store for survey completion status
const surveyCompletionStatus = new Map();

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Import and use AI routes
const aiRoutes = require('./src/routes/ai');
app.use('/api/ai', aiRoutes);

// Import and use OpenEMR feature routes
const encounterRoutes = require('./src/routes/encounters');
const soapNoteRoutes = require('./src/routes/soap-notes');
const problemRoutes = require('./src/routes/problems');
const allergyRoutes = require('./src/routes/allergies');
const immunizationRoutes = require('./src/routes/immunizations');
const prescriptionRoutes = require('./src/routes/prescriptions');
const billingRoutes = require('./src/routes/billing');

// Import new feature routes
const drugInteractionRoutes = require('./src/routes/drug-interactions');
const sideEffectRoutes = require('./src/routes/side-effects');
const adherenceRoutes = require('./src/routes/adherence');
const patientProfileRoutes = require('./src/routes/patient-profiles');
const diaryRoutes = require('./src/routes/diary');
const pillRecognitionRoutes = require('./src/routes/pill-recognition');

app.use('/api/encounters', encounterRoutes);
app.use('/api/soap-notes', soapNoteRoutes);
app.use('/api/problems', problemRoutes);
app.use('/api/allergies', allergyRoutes);
app.use('/api/immunizations', immunizationRoutes);
app.use('/api/prescriptions', prescriptionRoutes);
app.use('/api/billing', billingRoutes);

// New feature routes
app.use('/api/drug-interactions', drugInteractionRoutes);
app.use('/api/side-effects', sideEffectRoutes);
app.use('/api/adherence', adherenceRoutes);
app.use('/api/patient-profiles', patientProfileRoutes);
app.use('/api/diary', diaryRoutes);
app.use('/api/pill-recognition', pillRecognitionRoutes);

// Confir-Med API routes
const monopharmacyRoutes = require('./src/routes/monopharmacy');
const polypharmacyRoutes = require('./src/routes/polypharmacy');
app.use('/api/mono_se', monopharmacyRoutes);
app.use('/api/poly_se', polypharmacyRoutes);

// Medication validation endpoint (must be before /api/medications route)
app.post('/api/medications/validateMedication', async (req, res) => {
  try {
    // Accept multiple possible payload shapes from various frontends
    const rawInput = (req.body && (req.body.medication ?? req.body.medication_name ?? req.body.name ?? req.body.query ?? req.body.text ?? req.body.term)) ?? '';
    const medication = typeof rawInput === 'string' ? rawInput : String(rawInput || '').trim();
    console.log('Validating medication:', medication);
    
    // Use the new comprehensive medication matching service
    const { validateMedication } = require('./src/services/medicationMatchingService');
    const { callBioGPTProduction } = require('./src/utils/biogptClient');
    
    // Inject BioGPT caller
    const callBioGPT = async (prompt) => {
      try {
        return await callBioGPTProduction(prompt);
      } catch (error) {
        console.error('BioGPT call failed:', error.message);
        // Return a default response if BioGPT fails (don't block validation)
        return {
          is_medication: true,
          confidence: 0.5,
          drug_class: null,
          is_generic: true,
          is_brand: false
        };
      }
    };
    
    const result = await validateMedication(medication, { callBioGPTFn: callBioGPT });
    
    if (result.found) {
      // Success - return verified medication
      // Map result to expected frontend format
      return res.json({
        success: true,
        found: true,
        data: {
          generic_name: result.name,
          name: result.name,
          brand_names: [],
          drug_class: result.bio?.drug_class || 'Unknown',
          dosage_forms: ['tablet'],
          typical_strengths: [],
          confidence: result.score,
          alternatives: [],
          original_input: medication,
          source: result.source,
          rxcui: result.rxcui || null,
          bio: result.bio
        }
      });
    } else {
      // No match found - return proper no-match response
      return res.json({
        success: false,
        found: false,
        error: result.message || `No medication found for "${medication}"`,
        reason: result.reason,
        suggestions: result.suggestions || [],
        original_input: medication
      });
    }
  } catch (error) {
    console.error('Medication validation error:', error);
    res.status(500).json({ 
      success: false, 
      found: false,
      error: 'Failed to validate medication', 
      details: error.message 
    });
  }
});

// Enhanced medication tracking routes
const medicationTrackingRoutes = require('./src/routes/medication-tracking');
app.use('/api/medications', medicationTrackingRoutes);

// Test endpoint
app.get('/api/test-public', (req, res) => {
  res.json({ message: 'Backend is running!' });
});

// Auth endpoints
app.get('/api/auth/me', async (req, res) => {
  try {
    // For demo purposes, get the most recent user
    // In a real app, you'd verify the JWT token and return actual user data
    const user = await prisma.user.findFirst({
      orderBy: { createdAt: 'desc' }
    });
    
    if (user) {
      res.json({
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        hospitalCode: user.hospitalCode
      });
    } else {
      res.status(404).json({ error: 'User not found' });
    }
  } catch (error) {
    console.error('Error getting user data:', error);
    res.status(500).json({ error: 'Failed to get user data' });
  }
});

// Auth routes
app.post('/api/auth/signup', async (req, res) => {
  try {
    const { email, password, role, hospitalCode, patientData } = req.body;
    
    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email }
    });
    
    if (existingUser) {
      return res.status(400).json({ 
        error: 'User already exists',
        details: 'An account with this email already exists. Please login instead.'
      });
    }
    
    // Create user
    const user = await prisma.user.create({
      data: {
        email,
        password, // In production, hash this
        role: role || 'patient',
        hospitalCode: hospitalCode || '123456789',
        name: patientData?.name || null,
        surveyCompleted: false // Initialize survey as not completed
      }
    });

    // Create patient profile if role is patient
    if (role === 'patient') {
      // Remove name from patientData since it goes in User
      const { name, ...patientFields } = patientData || {};
      
      // Convert date strings to Date objects
      if (patientFields.dob) {
        patientFields.dob = new Date(patientFields.dob);
      }
      if (patientFields.baseline_weight_date) {
        patientFields.baseline_weight_date = new Date(patientFields.baseline_weight_date);
      }
      if (patientFields.baseline_hba1c_date) {
        patientFields.baseline_hba1c_date = new Date(patientFields.baseline_hba1c_date);
      }
      if (patientFields.baseline_lipid_date) {
        patientFields.baseline_lipid_date = new Date(patientFields.baseline_lipid_date);
      }
      
      const patient = await prisma.patient.create({
        data: {
          userId: user.id,
          patient_audit_id: `PAT-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          imd_decile: Math.floor(Math.random() * 10) + 1,
          ...patientFields
        }
      });
      console.log('Patient created with ID:', patient.id, 'for user:', user.email);
    }

    // Create clinician profile if role is clinician
    if (role === 'clinician') {
      await prisma.clinician.create({
        data: {
          userId: user.id,
          hospitalCode: hospitalCode || '123456789'
        }
      });
    }

    // Survey completion status is now stored in database (surveyCompleted field)
    // No need for in-memory map initialization

    res.json({
      success: true,
      user,
      token: 'mock-jwt-token-' + Date.now()
    });
  } catch (error) {
    console.error('Signup error:', error);
    console.error('Error details:', error.message);
    console.error('Stack trace:', error.stack);
    res.status(500).json({ error: 'Signup failed', details: error.message });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    // Find user
    const user = await prisma.user.findUnique({
      where: { email }
    });

    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // In production, verify password hash
    if (user.password !== password) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate JWT token (simplified for demo)
    const token = `demo-token-${user.id}-${Date.now()}`;

    res.json({ 
      success: true, 
      message: 'Login successful',
      token: token,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        hospitalCode: user.hospitalCode
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Failed to login' });
  }
});

// Get patients for clinician
app.get('/api/doctor/patients', async (req, res) => {
  try {
    // For demo purposes, get the most recent clinician's hospital code
    // In a real app, you'd get this from the JWT token
    const clinician = await prisma.clinician.findFirst({
      orderBy: { createdAt: 'desc' }
    });
    
    if (!clinician) {
      return res.json([]);
    }
    
    const clinicianHospitalCode = clinician.hospitalCode;
    console.log('Clinician hospital code:', clinicianHospitalCode);
    
    const patients = await prisma.patient.findMany({
      where: {
        user: {
          hospitalCode: clinicianHospitalCode,
          role: 'patient'
        }
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
            hospitalCode: true
          }
        },
        conditions: true
      }
    });
    
    console.log(`Found ${patients.length} patients for hospital code ${clinicianHospitalCode}`);
    
    // Debug: Log each patient's hospital code
    patients.forEach(patient => {
      console.log(`Patient ${patient.user.email} has hospital code: ${patient.user.hospitalCode}`);
    });

    const transformedPatients = patients.map(patient => {
      const age = patient.dob ? 
        Math.floor((new Date() - new Date(patient.dob)) / (365.25 * 24 * 60 * 60 * 1000)) : null;
      
      return {
        id: patient.id,
        userId: patient.userId,
        name: patient.user.name || `Patient ${patient.id}`,
        email: patient.user.email,
        hospitalCode: patient.user.hospitalCode,
        age,
        sex: patient.sex || null,
        ethnic_group: patient.ethnic_group || null,
        ethnicity: patient.ethnicity || null,
        location: patient.location || null,
        postcode: patient.postcode || null,
        nhs_number: patient.nhs_number || null,
        mrn: patient.mrn || null,
        height: patient.height || null,
        baseline_weight: patient.baseline_weight || null,
        baseline_bmi: patient.baseline_bmi || null,
        baseline_weight_date: patient.baseline_weight_date?.toISOString().split('T')[0] || null,
        ascvd: patient.ascvd || false,
        htn: patient.htn || false,
        dyslipidaemia: patient.dyslipidaemia || false,
        osa: patient.osa || false,
        sleep_studies: patient.sleep_studies || false,
        cpap: patient.cpap || false,
        t2dm: patient.t2dm || false,
        prediabetes: patient.prediabetes || false,
        diabetes_type: patient.diabetes_type || null,
        baseline_hba1c: patient.baseline_hba1c || null,
        baseline_hba1c_date: patient.baseline_hba1c_date?.toISOString().split('T')[0] || null,
        hba1cPercent: patient.hba1c_percent || null,
        hba1cMmolMol: patient.hba1c_mmol || null,
        baseline_tc: patient.baseline_tc || null,
        baseline_hdl: patient.baseline_hdl || null,
        baseline_ldl: patient.baseline_ldl || null,
        baseline_tg: patient.baseline_tg || null,
        baseline_lipid_date: patient.baseline_lipid_date?.toISOString().split('T')[0] || null,
        lipid_lowering_treatment: patient.lipid_lowering_treatment || null,
        antihypertensive_medications: patient.antihypertensive_medications || null,
        asthma: patient.asthma || false,
        hypertension: patient.hypertension || false,
        ischaemic_heart_disease: patient.ischaemic_heart_disease || false,
        heart_failure: patient.heart_failure || false,
        cerebrovascular_disease: patient.cerebrovascular_disease || false,
        pulmonary_hypertension: patient.pulmonary_hypertension || false,
        dvt: patient.dvt || false,
        pe: patient.pe || false,
        gord: patient.gord || false,
        ckd: patient.ckd || false,
        kidney_stones: patient.kidney_stones || false,
        masld: patient.masld || false,
        infertility: patient.infertility || false,
        pcos: patient.pcos || false,
        anxiety: patient.anxiety || false,
        depression: patient.depression || false,
        bipolar_disorder: patient.bipolar_disorder || false,
        emotional_eating: patient.emotional_eating || false,
        schizoaffective_disorder: patient.schizoaffective_disorder || false,
        oa_knee: patient.oa_knee || false,
        oa_hip: patient.oa_hip || false,
        limited_mobility: patient.limited_mobility || false,
        lymphoedema: patient.lymphoedema || false,
        thyroid_disorder: patient.thyroid_disorder || false,
        iih: patient.iih || false,
        epilepsy: patient.epilepsy || false,
        functional_neurological_disorder: patient.functional_neurological_disorder || false,
        cancer: patient.cancer || false,
        bariatric_gastric_band: patient.bariatric_gastric_band || false,
        bariatric_sleeve: patient.bariatric_sleeve || false,
        bariatric_bypass: patient.bariatric_bypass || false,
        bariatric_balloon: patient.bariatric_balloon || false,
        diagnoses_coded_in_scr: patient.diagnoses_coded_in_scr || null,
        total_qualifying_comorbidities: patient.total_qualifying_comorbidities || null,
        mes: patient.mes || null,
        notes: patient.notes || null,
        criteria_for_wegovy: patient.criteria_for_wegovy || null,
        conditions: patient.conditions.map(c => c.normalized),
        lastVisit: null,
        changePercent: null
      };
    });

    res.json(transformedPatients);
  } catch (error) {
    console.error('Error fetching patients:', error);
    res.status(500).json({ error: 'Failed to fetch patients' });
  }
});

// Parse medical history
app.post('/api/doctor/parse-history', async (req, res) => {
  try {
    const { patientId, medicalNotes } = req.body;
    
    if (!patientId || !medicalNotes) {
      return res.status(400).json({ 
        error: 'Missing required fields: patientId or medicalNotes',
        success: false 
      });
    }

    console.log(`🔍 Parsing medical history for patient ${patientId}`);
    console.log(`📝 Medical notes: ${medicalNotes.substring(0, 100)}...`);

    // Use Ollama parser for AI-powered extraction
    let parsedData;
    let rawModelOutput = null;
    try {
      const { runOllamaParser } = require('./utils/ollamaParser');
      parsedData = await runOllamaParser(medicalNotes);
      console.log('📊 Parsed data from AI:', Object.keys(parsedData));
    } catch (parseError) {
      console.error('❌ Parser error:', parseError.message);
      console.error('❌ Parser stack:', parseError.stack);
      console.error('❌ Error details:', {
        rawModelOutput: rawModelOutput || 'N/A',
        patientId: patientId,
        notesLength: medicalNotes?.length || 0
      });
      throw new Error(`Failed to parse medical notes: ${parseError.message}`);
    }

    // Get current patient data for comparison
    const currentPatient = await prisma.patient.findUnique({
      where: { id: patientId }
    });

    if (!currentPatient) {
      return res.status(404).json({ 
        error: 'Patient not found',
        success: false 
      });
    }

    const auditLogs = [];
    const updates = {};

    // Helper: parse flexible dates like DD/MM/YYYY or ISO
    function parseFlexibleDate(input) {
      if (!input) return null;
      if (input instanceof Date) return isNaN(input.getTime()) ? null : input;
      const str = String(input).trim();
      // Try ISO first
      const iso = new Date(str);
      if (!isNaN(iso.getTime())) return iso;
      // Try DD/MM/YYYY
      const m = str.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
      if (m) {
        const d = parseInt(m[1], 10);
        const mo = parseInt(m[2], 10) - 1;
        const y = parseInt(m[3], 10);
        const dt = new Date(y, mo, d);
        return isNaN(dt.getTime()) ? null : dt;
      }
      return null;
    }

    const dateFields = new Set([
      'baseline_hba1c_date',
      'baseline_lipid_date',
      'baseline_weight_date',
      'start_date',
      'hba1c_date',
      'lipid_date'
    ]);

    // Map AI parsed data to database fields
    // Note: Patient model doesn't have 'age' field, only 'dob' (date of birth)
    const fieldMapping = {
      sex: 'sex',
      height: 'height',
      weight: 'baseline_weight',
      baseline_weight: 'baseline_weight',
      bmi: 'baseline_bmi',
      baseline_bmi: 'baseline_bmi',
      baseline_weight_date: 'baseline_weight_date',
      systolic_bp: 'systolic_bp',
      diastolic_bp: 'diastolic_bp',
      hba1c_percent: 'hba1c_percent',
      baseline_hba1c: 'baseline_hba1c',
      baseline_hba1c_date: 'baseline_hba1c_date',
      baseline_fasting_glucose: 'baseline_fasting_glucose',
      random_glucose: 'random_glucose',
      baseline_tc: 'baseline_tc',
      baseline_hdl: 'baseline_hdl',
      baseline_ldl: 'baseline_ldl',
      baseline_tg: 'baseline_tg',
      baseline_lipid_date: 'baseline_lipid_date',
      lipid_lowering_treatment: 'lipid_lowering_treatment',
      antihypertensive_medications: 'antihypertensive_medications',
      creatinine: 'creatinine',
      egfr: 'egfr',
      t2dm: 't2dm',
      prediabetes: 'prediabetes',
      diabetes_type: 'diabetes_type',
      htn: 'htn',
      hypertension: 'hypertension',
      dyslipidaemia: 'dyslipidaemia',
      ascvd: 'ascvd',
      ckd: 'ckd',
      osa: 'osa',
      sleep_studies: 'sleep_studies',
      cpap: 'cpap',
      asthma: 'asthma',
      ischaemic_heart_disease: 'ischaemic_heart_disease',
      heart_failure: 'heart_failure',
      cerebrovascular_disease: 'cerebrovascular_disease',
      pulmonary_hypertension: 'pulmonary_hypertension',
      dvt: 'dvt',
      pe: 'pe',
      gord: 'gord',
      kidney_stones: 'kidney_stones',
      masld: 'masld',
      thyroid_disorder: 'thyroid_disorder',
      infertility: 'infertility',
      pcos: 'pcos',
      iih: 'iih',
      epilepsy: 'epilepsy',
      functional_neurological_disorder: 'functional_neurological_disorder',
      cancer: 'cancer',
      anxiety: 'anxiety',
      depression: 'depression',
      bipolar_disorder: 'bipolar_disorder',
      emotional_eating: 'emotional_eating',
      schizoaffective_disorder: 'schizoaffective_disorder',
      oa_knee: 'oa_knee',
      oa_hip: 'oa_hip',
      limited_mobility: 'limited_mobility',
      lymphoedema: 'lymphoedema',
      bariatric_gastric_band: 'bariatric_gastric_band',
      bariatric_sleeve: 'bariatric_sleeve',
      bariatric_bypass: 'bariatric_bypass',
      bariatric_balloon: 'bariatric_balloon',
      obesity: 'obesity',
      notes: 'notes',
      comorbidities_count: 'total_qualifying_comorbidities'
    };

    // Allowed patient fields (must exist in Prisma Patient model)
    const allowedPatientFields = new Set([
      'sex','dob','height','baseline_weight','baseline_bmi','baseline_weight_date',
      'ascvd','htn','hypertension','dyslipidaemia','ischaemic_heart_disease','heart_failure',
      'cerebrovascular_disease','pulmonary_hypertension','dvt','pe','osa','sleep_studies','cpap','asthma',
      't2dm','prediabetes','diabetes_type','hba1c_percent','baseline_hba1c','baseline_hba1c_date',
      'hba1c_mmol','baseline_fasting_glucose','random_glucose',
      'baseline_tc','baseline_hdl','baseline_ldl','baseline_tg','baseline_lipid_date','lipid_lowering_treatment',
      'antihypertensive_medications','gord','ckd','kidney_stones','masld','infertility','pcos','anxiety','depression',
      'bipolar_disorder','emotional_eating','schizoaffective_disorder','oa_knee','oa_hip','limited_mobility','lymphoedema',
      'thyroid_disorder','iih','epilepsy','functional_neurological_disorder','cancer',
      'bariatric_gastric_band','bariatric_sleeve','bariatric_bypass','bariatric_balloon',
      'diagnoses_coded_in_scr','total_qualifying_comorbidities','mes','notes','criteria_for_wegovy'
    ]);

    // Helper: Convert 0/1/null to Boolean? for Prisma
    function convertBooleanValue(value) {
      if (value === null || value === undefined) return null;
      if (typeof value === 'number') {
        if (value === 0) return false;
        if (value === 1) return true;
        return null;
      }
      if (typeof value === 'boolean') return value;
      return null;
    }

    // Conditions are already mapped to columns in validateAndNormalizePatientData
    // parsedData.conditions array was used to set specific columns to 1
    // All condition fields are now 0 or 1 (never null)
    console.log('📋 Condition fields initialized and mapped from conditions array');

    // Compare parsed data with current data and create updates
    // First, handle all boolean fields from parsedData (0/1/null)
    const booleanFields = [
      't2dm', 'prediabetes', 'htn', 'hypertension', 'dyslipidaemia', 'ascvd', 'ckd', 'osa',
      'sleep_studies', 'cpap', 'asthma', 'ischaemic_heart_disease', 'heart_failure',
      'cerebrovascular_disease', 'pulmonary_hypertension', 'dvt', 'pe', 'gord', 'kidney_stones',
      'masld', 'infertility', 'pcos', 'anxiety', 'depression', 'bipolar_disorder',
      'emotional_eating', 'schizoaffective_disorder', 'oa_knee', 'oa_hip', 'limited_mobility',
      'lymphoedema', 'thyroid_disorder', 'iih', 'epilepsy', 'functional_neurological_disorder',
      'cancer', 'bariatric_gastric_band', 'bariatric_sleeve', 'bariatric_bypass', 'bariatric_balloon'
    ];

    for (const field of booleanFields) {
      const currentValue = currentPatient[field];
      // parsedData[field] is already 0 or 1 (never null) from validateAndNormalizePatientData
      const parsedValue = parsedData[field];
      
      // Convert 0/1 to false/true for Prisma Boolean fields
      let newValue = null;
      if (parsedValue === 0) {
        newValue = false;
      } else if (parsedValue === 1) {
        newValue = true;
      } else {
        // Should never happen, but default to false if somehow null/undefined
        newValue = false;
        console.warn(`⚠️ Field ${field} was null/undefined, defaulting to false`);
      }
      
      // Only create audit log if value changed
      if (currentValue !== newValue) {
        try {
          auditLogs.push({
            patientId: String(patientId), // Ensure string
            field_name: String(field), // Ensure string
            old_value: currentValue !== null && currentValue !== undefined ? String(currentValue) : '',
            new_value: String(newValue), // Ensure string
            ai_confidence: 0.9,
            ai_suggestion: conditionFieldMap[field] === 1 
              ? `AI detected from conditions array: ${newValue}`
              : `AI detected: ${newValue}`,
            clinician_approved: false
          });
          updates[field] = newValue;
        } catch (logError) {
          console.error(`⚠️ Error creating audit log for field ${field}:`, logError.message);
          // Still add to updates even if audit log fails
          updates[field] = newValue;
        }
      }
    }

    // Handle non-boolean fields from fieldMapping
    for (const [aiField, dbField] of Object.entries(fieldMapping)) {
      // Skip boolean fields (already handled above)
      if (booleanFields.includes(dbField)) continue;
      
      // Skip age field - Patient model doesn't have age, only dob
      if (aiField === 'age') continue;
      
      if (parsedData[aiField] !== null && parsedData[aiField] !== undefined) {
        // Skip fields that are not part of Patient model
        if (!allowedPatientFields.has(dbField)) continue;
        
        // Ensure numeric fields stay numeric (not converted to boolean)
        const numericFields = ['height', 'baseline_weight', 'baseline_bmi', 'baseline_hba1c', 
                               'hba1c_percent', 'baseline_fasting_glucose', 'random_glucose',
                               'baseline_tc', 'baseline_hdl', 'baseline_ldl', 'baseline_tg'];
        if (numericFields.includes(dbField) && typeof parsedData[aiField] === 'boolean') {
          // Skip if it's a boolean - this shouldn't happen but protect against it
          continue;
        }

        const currentValue = currentPatient[dbField];
        let newValue = parsedData[aiField];
        
        // Coerce dates
        if (dateFields.has(dbField)) {
          const dt = parseFlexibleDate(newValue);
          newValue = dt || undefined;
        }
        
        if (newValue !== undefined && currentValue?.toString() !== newValue?.toString()) {
          try {
            auditLogs.push({
              patientId: String(patientId), // Ensure string
              field_name: String(dbField), // Ensure string
              old_value: currentValue !== null && currentValue !== undefined ? String(currentValue) : '',
              new_value: String(newValue), // Ensure string
              ai_confidence: 0.9,
              ai_suggestion: `AI detected: ${newValue}`,
              clinician_approved: false
            });
            updates[dbField] = newValue;
          } catch (logError) {
            console.error(`⚠️ Error creating audit log for field ${dbField}:`, logError.message);
            // Still add to updates even if audit log fails
            updates[dbField] = newValue;
          }
        }
      }
    }

    // Handle medications separately
    if (parsedData.medications && Array.isArray(parsedData.medications)) {
      const medicationsString = parsedData.medications.join(', ');
      if (currentPatient.all_medications_from_scr !== medicationsString) {
        try {
          auditLogs.push({
            patientId: String(patientId), // Ensure string
            field_name: 'all_medications_from_scr',
            old_value: currentPatient.all_medications_from_scr ? String(currentPatient.all_medications_from_scr) : '',
            new_value: String(medicationsString), // Ensure string
            ai_confidence: 0.9,
            ai_suggestion: `AI detected medications: ${medicationsString}`,
            clinician_approved: false
          });
          updates.all_medications_from_scr = medicationsString;
        } catch (logError) {
          console.error('⚠️ Error creating audit log for medications:', logError.message);
          // Still add to updates even if audit log fails
          updates.all_medications_from_scr = medicationsString;
        }
      }
    }

    // Save audit logs
    if (auditLogs.length > 0) {
      try {
        // Validate audit log data before saving
        const validAuditLogs = auditLogs.filter(log => {
          return log.patientId && log.field_name && log.new_value !== undefined;
        });
        
        if (validAuditLogs.length > 0) {
          await prisma.aiAuditLog.createMany({
            data: validAuditLogs
          });
          console.log(`✅ Saved ${validAuditLogs.length} audit logs`);
        } else {
          console.warn('⚠️ No valid audit logs to save');
        }
      } catch (auditError) {
        console.error('❌ Error saving audit logs:', auditError.message);
        console.error('❌ Audit log error details:', auditError);
        // Don't fail the entire request if audit logs fail
      }
    }

    // Extract conditions from parsedData.conditions array
    // Use normalizeConditionName helper from conditionMapper
    const { normalizeConditionName } = require('./src/utils/conditionMapper');
    let extractedConditions = [];
    
    // Primary: Use parsedData.conditions array
    if (Array.isArray(parsedData.conditions)) {
      extractedConditions = parsedData.conditions.map(condition => 
        normalizeConditionName(condition)
      );
    }
    
    // Fallback: Check boolean flags (only if conditions array is empty or missing)
    const conditionFlags = {
      t2dm: 'Type 2 Diabetes Mellitus',
      prediabetes: 'Prediabetes',
      htn: 'Hypertension',
      hypertension: 'Hypertension',
      dyslipidaemia: 'Dyslipidaemia',
      ascvd: 'Atherosclerotic Cardiovascular Disease',
      ckd: 'Chronic Kidney Disease',
      osa: 'Obstructive Sleep Apnea',
      masld: 'MASLD',
      anxiety: 'Anxiety',
      depression: 'Depression',
      bipolar_disorder: 'Bipolar Disorder',
      emotional_eating: 'Emotional Eating',
      schizoaffective_disorder: 'Schizoaffective Disorder',
      oa_knee: 'Osteoarthritis Knee',
      oa_hip: 'Osteoarthritis Hip',
      limited_mobility: 'Limited Mobility',
      lymphoedema: 'Lymphoedema',
      thyroid_disorder: 'Thyroid Disorder',
      iih: 'Idiopathic Intracranial Hypertension',
      epilepsy: 'Epilepsy',
      functional_neurological_disorder: 'Functional Neurological Disorder',
      cancer: 'Cancer',
      ischaemic_heart_disease: 'Ischaemic Heart Disease',
      heart_failure: 'Heart Failure',
      cerebrovascular_disease: 'Cerebrovascular Disease',
      pulmonary_hypertension: 'Pulmonary Hypertension',
      dvt: 'Deep Vein Thrombosis',
      pe: 'Pulmonary Embolism',
      gord: 'Gastroesophageal Reflux Disease',
      kidney_stones: 'Kidney Stones',
      infertility: 'Infertility',
      pcos: 'Polycystic Ovary Syndrome',
      bariatric_gastric_band: 'Gastric Band',
      bariatric_sleeve: 'Gastric Sleeve',
      bariatric_bypass: 'Gastric Bypass',
      bariatric_balloon: 'Gastric Balloon'
    };
    
    // Only use boolean flags as fallback if conditions array is empty
    if (extractedConditions.length === 0) {
      for (const [key, label] of Object.entries(conditionFlags)) {
        if (parsedData[key] === 1) {
          const normalized = normalizeConditionName(label);
          if (!extractedConditions.includes(normalized)) {
            extractedConditions.push(normalized);
          }
        }
      }
    }
    
    // Remove duplicates
    extractedConditions = [...new Set(extractedConditions)];

    console.log(`✅ AI parsed ${Object.keys(parsedData).length} fields, found ${auditLogs.length} updates`);

    // CRITICAL: Actually apply the updates to the patient record
    if (Object.keys(updates).length > 0) {
      try {
        // Clean updates object - remove any undefined/null values and ensure proper types
        const cleanedUpdates = {};
        for (const [key, value] of Object.entries(updates)) {
          if (value !== undefined && value !== null) {
            // Handle date fields
            if (dateFields.has(key)) {
              const parsedDate = parseFlexibleDate(value);
              if (parsedDate) {
                cleanedUpdates[key] = parsedDate;
              }
            } else {
              cleanedUpdates[key] = value;
            }
          }
        }

        // Update the patient record
        await prisma.patient.update({
          where: { id: patientId },
          data: cleanedUpdates
        });

        console.log(`✅ Successfully updated patient ${patientId} with ${Object.keys(cleanedUpdates).length} fields`);
      } catch (updateError) {
        console.error('❌ Error updating patient record:', updateError.message);
        console.error('❌ Update error details:', updateError);
        // Don't fail the entire request - return success but log the error
        // The audit logs are still created, so the clinician can review them
      }
    } else {
      console.log('ℹ️ No updates to apply (all values match current patient data)');
    }

    res.json({
      success: true,
      parsedData,
      updates,
      auditLogs: auditLogs.length,
      conditions: extractedConditions.map(c => ({
        name: c,
        normalized: c.toLowerCase()
      })),
      message: `AI successfully extracted ${Object.keys(parsedData).length} data points and ${extractedConditions.length} conditions. ${Object.keys(updates).length} fields updated.`
    });
  } catch (error) {
    console.error('❌ Error parsing medical history:', error.message);
    console.error('❌ Error stack:', error.stack);
    console.error('❌ Full error object:', {
      message: error.message,
      stack: error.stack,
      name: error.name,
      patientId: req.body?.patientId || 'unknown',
      notesLength: req.body?.medicalNotes?.length || 0
    });
    
    res.status(500).json({ 
      success: false,
      error: 'Failed to parse medical history',
      details: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});

// Intelligent Medical Parsing with Patient Matching
app.post('/api/doctor/intelligent-parse', async (req, res) => {
  try {
    const { medicalNotes, hospitalCode } = req.body;
    
    if (!medicalNotes) {
      return res.status(400).json({ 
        error: 'Missing required field: medicalNotes',
        success: false 
      });
    }

    console.log('🔍 Starting intelligent medical parsing...');
    console.log('📝 Medical notes:', medicalNotes.substring(0, 200) + '...');

    // Step 1: Parse medical notes with LLM
    const parseResult = await runIntelligentMedicalParser(medicalNotes);
    
    if (!parseResult.success) {
      return res.status(500).json({
        error: 'Failed to parse medical notes',
        success: false
      });
    }

    const extractedData = parseResult.data;
    console.log('✅ Extracted data:', extractedData);

    // Step 2: Find patients by name
    const patients = await prisma.patient.findMany({
      where: {
        user: {
          hospitalCode: hospitalCode || '123456789' // Default hospital code for demo
        }
      },
      include: {
        user: true
      }
    });

    const patientMatch = findPatientByName(extractedData.Patient_Name, patients);
    
    let targetPatient = null;
    let patientAction = 'none';

    if (patientMatch.found) {
      if (patientMatch.matches.length === 1) {
        // Single match - use it
        targetPatient = patientMatch.matches[0].patient;
        patientAction = 'matched';
        console.log(`✅ Single patient match found: ${targetPatient.user.name}`);
      } else {
        // Multiple matches - return for clinician selection
        return res.json({
          success: true,
          action: 'select_patient',
          extractedData,
          patientMatches: patientMatch.matches.map(match => ({
            id: match.patient.id,
            name: match.patient.user.name,
            email: match.patient.user.email,
            nhsNumber: match.patient.nhs_number,
            mrn: match.patient.mrn,
            confidence: match.confidence
          })),
          message: `Found ${patientMatch.matches.length} potential patient matches. Please select the correct patient.`
        });
      }
    } else {
      // No match - create new patient
      patientAction = 'create_new';
      console.log('🆕 No patient match found, will create new patient');
    }

    // Step 3: Process the patient data
    if (patientAction === 'create_new') {
      // Create new patient
      const mappedData = mapExtractedDataToDatabase(extractedData);
      
      // Create user first
      const newUser = await prisma.user.create({
        data: {
          email: `patient-${Date.now()}@example.com`,
          password: 'temp-password',
          name: extractedData.Patient_Name || 'Unknown Patient',
          role: 'patient',
          hospitalCode: hospitalCode || '123456789',
          surveyCompleted: true
        }
      });

      // Create patient record
      const newPatient = await prisma.patient.create({
        data: {
          userId: newUser.id,
          ...mappedData
        }
      });

      targetPatient = newPatient;
      console.log(`✅ Created new patient: ${newUser.name} (ID: ${newPatient.id})`);
    }

    // Step 4: Update existing patient with extracted data
    if (targetPatient && patientAction !== 'create_new') {
      const mappedData = mapExtractedDataToDatabase(extractedData);
      
      // Create audit logs for changes
      const auditLogs = [];
      const updates = {};

      // Compare and track changes
      Object.entries(mappedData).forEach(([field, value]) => {
        const currentValue = targetPatient[field];
        if (currentValue !== value && value !== null && value !== undefined) {
          auditLogs.push({
            patientId: targetPatient.id,
            field_name: field,
            old_value: currentValue?.toString() || null,
            new_value: value.toString(),
            ai_confidence: parseResult.confidence,
            ai_suggestion: `AI extracted: ${value}`,
            clinician_approved: false
          });
          updates[field] = value;
        }
      });

      // Save audit logs
      if (auditLogs.length > 0) {
        await prisma.aiAuditLog.createMany({
          data: auditLogs
        });
      }

      // Update patient data
      if (Object.keys(updates).length > 0) {
        // Remove invalid dates before saving
        const cleanedUpdates = Object.fromEntries(
          Object.entries(updates).filter(([key, value]) => {
            // Only allow fields present in Patient model
            if (!allowedPatientFields.has(key)) return false;
            if (value instanceof Date) return !isNaN(value.getTime());
            if (key.endsWith('_date') && typeof value === 'string') return false;
            return value !== undefined && value !== null && value !== 'Invalid Date';
          })
        );

        await prisma.patient.update({
          where: { id: targetPatient.id },
          data: cleanedUpdates
        });
      }

      console.log(`✅ Updated patient ${targetPatient.id} with ${Object.keys(updates).length} changes`);
    }

    // Step 5: Create medical note record
    const medicalNote = await prisma.medicalNote.create({
      data: {
        patientId: targetPatient.id,
        raw_text: medicalNotes,
        note_type: 'consultation',
        source: 'ai_parsed',
        ai_processed: true,
        ai_confidence: parseResult.confidence,
        ai_model_used: parseResult.model_used,
        extracted_data: JSON.stringify(extractedData),
        patient_name: extractedData.Patient_Name,
        age: extractedData.Age,
        sex: extractedData.Sex,
        conditions: JSON.stringify(extractedData.Conditions || []),
        medications: JSON.stringify(extractedData.Medications || []),
        allergies: JSON.stringify(extractedData.Allergies || []),
        lab_results: JSON.stringify(extractedData.Labs || []),
        vital_signs: JSON.stringify(extractedData.Vitals || []),
        impression: extractedData.Impression,
        plan: extractedData.Plan
      }
    });

    // Step 6: Create lab results and vital signs records
    const labResults = extractedData.Labs || [];
    const vitalSigns = extractedData.Vitals || [];

    // Save lab results
    for (const lab of labResults) {
      await prisma.labResult.create({
        data: {
          patientId: targetPatient.id,
          metric_name: lab.metric,
          value: lab.value,
          unit: lab.unit,
          date: lab.date ? new Date(lab.date) : new Date(),
          source_note_id: medicalNote.id,
          manually_entered: false
        }
      });
    }

    // Save vital signs
    for (const vital of vitalSigns) {
      await prisma.vitalSign.create({
        data: {
          patientId: targetPatient.id,
          vital_type: vital.type,
          value: vital.value,
          unit: vital.unit,
          date: new Date(),
          value_secondary: vital.value_secondary,
          source_note_id: medicalNote.id,
          manually_entered: false
        }
      });
    }

    // Step 7: Create medication records
    const medications = extractedData.Medications || [];
    for (const med of medications) {
      await prisma.patientMedication.create({
        data: {
          patientId: targetPatient.id,
          name: med.name,
          dosage: med.dose,
          frequency: med.frequency,
          route: 'oral', // Default route
          start_date: new Date(),
          status: 'active',
          source_note_id: medicalNote.id,
          manually_entered: false
        }
      });
    }

    console.log(`✅ Intelligent parsing completed for patient ${targetPatient.id}`);

    res.json({
      success: true,
      action: patientAction,
      patient: {
        id: targetPatient.id,
        name: targetPatient.user?.name || extractedData.Patient_Name,
        email: targetPatient.user?.email,
        nhsNumber: targetPatient.nhs_number,
        mrn: targetPatient.mrn
      },
      extractedData,
      medicalNoteId: medicalNote.id,
      labResultsCount: labResults.length,
      vitalSignsCount: vitalSigns.length,
      medicationsCount: medications.length,
      message: `Successfully ${patientAction === 'create_new' ? 'created new patient' : 'updated existing patient'} with AI-extracted data`
    });

  } catch (error) {
    console.error('❌ Intelligent parsing error:', error);
    res.status(500).json({ 
      error: 'Failed to process medical notes intelligently',
      success: false,
      details: error.message
    });
  }
});

// Get patient metrics
app.get('/api/metrics/patient/:patientId', async (req, res) => {
  try {
    const { patientId } = req.params;
    const metrics = await prisma.metricTrend.findMany({
      where: { patientId },
      orderBy: { date: 'desc' }
    });
    res.json(metrics);
  } catch (error) {
    console.error('Error fetching patient metrics:', error);
    res.status(500).json({ error: 'Failed to fetch metrics' });
  }
});

// Get patient lab results
app.get('/api/lab-results/patient/:patientId', async (req, res) => {
  try {
    const { patientId } = req.params;
    const labResults = await prisma.labResult.findMany({
      where: { patientId },
      orderBy: { date: 'desc' }
    });
    res.json(labResults);
  } catch (error) {
    console.error('Error fetching lab results:', error);
    res.status(500).json({ error: 'Failed to fetch lab results' });
  }
});

// Get patient vital signs
app.get('/api/vital-signs/patient/:patientId', async (req, res) => {
  try {
    const { patientId } = req.params;
    const vitalSigns = await prisma.vitalSign.findMany({
      where: { patientId },
      orderBy: { date: 'desc' }
    });
    res.json(vitalSigns);
  } catch (error) {
    console.error('Error fetching vital signs:', error);
    res.status(500).json({ error: 'Failed to fetch vital signs' });
  }
});

// Approve audit log
app.post('/api/doctor/audit-logs/:logId/approve', async (req, res) => {
  try {
    const { logId } = req.params;
    await prisma.aiAuditLog.update({
      where: { id: logId },
      data: { clinician_approved: true }
    });
    res.json({ success: true });
  } catch (error) {
    console.error('Error approving audit log:', error);
    res.status(500).json({ error: 'Failed to approve audit log' });
  }
});

// Reject audit log
app.post('/api/doctor/audit-logs/:logId/reject', async (req, res) => {
  try {
    const { logId } = req.params;
    await prisma.aiAuditLog.delete({
      where: { id: logId }
    });
    res.json({ success: true });
  } catch (error) {
    console.error('Error rejecting audit log:', error);
    res.status(500).json({ error: 'Failed to reject audit log' });
  }
});

// Update patient data
app.put('/api/doctor/patients/:patientId', async (req, res) => {
  try {
    const { patientId } = req.params;
    const updateData = req.body;

    const updatedPatient = await prisma.patient.update({
      where: { id: patientId },
      data: updateData
    });

    res.json({
      success: true,
      patient: updatedPatient
    });
  } catch (error) {
    console.error('Error updating patient data:', error);
    res.status(500).json({ error: 'Failed to update patient data' });
  }
});

// Get audit logs
app.get('/api/doctor/patients/:patientId/audit-logs', async (req, res) => {
  try {
    const { patientId } = req.params;

    const auditLogs = await prisma.aiAuditLog.findMany({
      where: { patientId },
      orderBy: { createdAt: 'desc' }
    });

    res.json(auditLogs);
  } catch (error) {
    console.error('Error fetching audit logs:', error);
    res.status(500).json({ error: 'Failed to fetch audit logs' });
  }
});

// Missing endpoints that the frontend expects
app.get('/api/ai/status', (req, res) => {
  res.json({ 
    status: 'available',
    model: 'mock-ai-model',
    version: '1.0.0'
  });
});

app.get('/api/auth/survey-status', async (req, res) => {
  try {
    // Get user ID from query parameter or use the most recent user
    let userId = req.query.userId;
    
    if (!userId) {
      // For demo purposes, get the most recent user
      const latestUser = await prisma.user.findFirst({
        orderBy: { createdAt: 'desc' }
      });
      if (!latestUser) {
        return res.json({ 
          surveyCompleted: false,
          lastCompleted: null
        });
      }
      userId = latestUser.id;
    }
    
    // Check survey status from database
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { surveyCompleted: true, email: true }
    });
    
    if (!user) {
      return res.json({ 
        surveyCompleted: false,
        lastCompleted: null
      });
    }
    
    console.log('Survey status requested for user:', user.email, '- completed:', user.surveyCompleted);
    
    res.json({ 
      surveyCompleted: user.surveyCompleted || false,
      lastCompleted: user.surveyCompleted ? new Date().toISOString() : null
    });
  } catch (error) {
    console.error('Error checking survey status:', error);
    res.json({ 
      surveyCompleted: false,
      lastCompleted: null
    });
  }
});

app.post('/api/auth/survey-data', async (req, res) => {
  try {
    console.log('Survey data received:', req.body);
    
    // For demo purposes, update the most recent patient record
    // In a real app, you'd get user ID from JWT token and save to specific patient
    const surveyData = req.body;
    
    // Find the most recent patient (for demo purposes)
    const latestPatient = await prisma.patient.findFirst({
      orderBy: { createdAt: 'desc' }
    });
    
    console.log('Latest patient found:', latestPatient ? latestPatient.id : 'No patient found');
    
    if (latestPatient) {
      // Update user name first
      console.log('Updating user name for patient:', latestPatient.userId);
      try {
        await prisma.user.update({
          where: { id: latestPatient.userId },
          data: { name: surveyData.name }
        });
        console.log('User name updated successfully');
      } catch (userUpdateError) {
        console.error('Error updating user name:', userUpdateError);
        throw userUpdateError;
      }
      
      // Safely convert date
      let dob = null;
      if (surveyData.dateOfBirth) {
        try {
          dob = new Date(surveyData.dateOfBirth);
          // Check if date is valid
          if (isNaN(dob.getTime())) {
            console.log('Invalid date, using null');
            dob = null;
          } else {
            console.log('Date converted successfully:', dob);
          }
        } catch (error) {
          console.log('Date conversion error:', error);
          dob = null;
        }
      }
      
      // Update patient record with survey data
      const updateData = {
        // Basic demographics
        dob: dob,
        sex: surveyData.biologicalSex || null,
        ethnicity: surveyData.ethnicity || null,
        ethnic_group: surveyData.ethnicity || null,
        location: surveyData.location || null,
        postcode: surveyData.postcode || null,
        nhs_number: surveyData.nhsNumber || null,
        
        // Physical measurements
        height: surveyData.height,
        baseline_weight: surveyData.weight,
        baseline_bmi: surveyData.baselineBMI,
        baseline_weight_date: surveyData.baselineWeightDate ? new Date(surveyData.baselineWeightDate) : null,
        
        // Medical conditions
        ascvd: surveyData.ascvd,
        htn: surveyData.htn,
        hypertension: surveyData.hypertension,
        dyslipidaemia: surveyData.dyslipidaemia,
        ischaemic_heart_disease: surveyData.ischaemicHeartDisease,
        heart_failure: surveyData.heartFailure,
        cerebrovascular_disease: surveyData.cerebrovascularDisease,
        pulmonary_hypertension: surveyData.pulmonaryHypertension,
        dvt: surveyData.dvt,
        pe: surveyData.pe,
        osa: surveyData.osa,
        sleep_studies: surveyData.sleepStudies,
        cpap: surveyData.cpap,
        asthma: surveyData.asthma,
        t2dm: surveyData.t2dm,
        prediabetes: surveyData.prediabetes,
        diabetes_type: surveyData.diabetesType,
        gord: surveyData.gord,
        ckd: surveyData.ckd,
        kidney_stones: surveyData.kidneyStones,
        masld: surveyData.masld,
        infertility: surveyData.infertility,
        pcos: surveyData.pcos,
        anxiety: surveyData.anxiety,
        depression: surveyData.depression,
        bipolar_disorder: surveyData.bipolarDisorder,
        emotional_eating: surveyData.emotionalEating,
        schizoaffective_disorder: surveyData.schizoaffectiveDisorder,
        oa_knee: surveyData.oaKnee,
        oa_hip: surveyData.oaHip,
        limited_mobility: surveyData.limitedMobility,
        lymphoedema: surveyData.lymphoedema,
        thyroid_disorder: surveyData.thyroidDisorder,
        iih: surveyData.iih,
        epilepsy: surveyData.epilepsy,
        functional_neurological_disorder: surveyData.functionalNeurologicalDisorder,
        cancer: surveyData.cancer,
        bariatric_gastric_band: surveyData.bariatricGastricBand,
        bariatric_sleeve: surveyData.bariatricSleeve,
        bariatric_bypass: surveyData.bariatricBypass,
        bariatric_balloon: surveyData.bariatricBalloon,
        
        // Medical data
        baseline_hba1c: surveyData.baselineHbA1c,
        baseline_hba1c_date: surveyData.baselineHbA1cDate ? new Date(surveyData.baselineHbA1cDate) : null,
        baseline_fasting_glucose: surveyData.baselineFastingGlucose,
        random_glucose: surveyData.randomGlucose,
        baseline_tc: surveyData.baselineTC,
        baseline_hdl: surveyData.baselineHDL,
        baseline_ldl: surveyData.baselineLDL,
        baseline_tg: surveyData.baselineTG,
        baseline_lipid_date: surveyData.baselineLipidDate ? new Date(surveyData.baselineLipidDate) : null,
        lipid_lowering_treatment: surveyData.lipidLoweringTreatment,
        antihypertensive_medications: surveyData.antihypertensiveMedications,
        all_medications_from_scr: surveyData.allMedicationsFromSCR,
        diagnoses_coded_in_scr: surveyData.diagnosesCodedInSCR,
        total_qualifying_comorbidities: surveyData.totalQualifyingComorbidities,
        mes: surveyData.mes,
        notes: surveyData.notes,
        criteria_for_wegovy: surveyData.criteriaForWegovy
      };
      
      // Filter out undefined values to prevent database errors
      const filteredUpdateData = Object.fromEntries(
        Object.entries(updateData).filter(([key, value]) => value !== undefined)
      );
      
      console.log('Updating patient record with filtered data:', filteredUpdateData);
      try {
        await prisma.patient.update({
          where: { id: latestPatient.id },
          data: filteredUpdateData
        });
        console.log('Survey data saved to patient record:', latestPatient.id);
      } catch (patientUpdateError) {
        console.error('Error updating patient record:', patientUpdateError);
        console.error('Update data that caused error:', filteredUpdateData);
        throw patientUpdateError;
      }
    }
    
    res.json({ 
      success: true,
      message: 'Survey data saved successfully'
    });
  } catch (error) {
    console.error('Error saving survey data:', error);
    console.error('Error details:', error.message);
    console.error('Error stack:', error.stack);
    res.status(500).json({ 
      success: false,
      error: 'Failed to save survey data',
      details: error.message
    });
  }
});

app.put('/api/auth/complete-survey', async (req, res) => {
  try {
    console.log('Survey completion requested');
    
    // Get user ID from query parameter or use the most recent user
    let userId = req.body.userId || req.query.userId;
    
    if (!userId) {
      // For demo purposes, get the most recent user
      const latestUser = await prisma.user.findFirst({
        orderBy: { createdAt: 'desc' }
      });
      if (!latestUser) {
        return res.status(400).json({ 
          success: false,
          error: 'User not found'
        });
      }
      userId = latestUser.id;
    }
    
    // Update survey completion status in database
    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: { surveyCompleted: true },
      select: { id: true, email: true, surveyCompleted: true }
    });

    console.log('Survey marked as completed for user:', updatedUser.email, updatedUser.id);

    res.json({ 
      success: true,
      message: 'Survey marked as completed',
      user: updatedUser
    });
  } catch (error) {
    console.error('Error completing survey:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to mark survey as completed',
      details: error.message
    });
  }
});

// Get user medications endpoint
app.get('/api/meds/user', async (req, res) => {
  try {
    // Get user ID from query params or find the most recent user (for demo)
    // In production, get from JWT token
    let userId = req.query.userId;
    console.log('GET /api/meds/user - userId from query:', userId);
    
    if (!userId) {
      // For demo: get the most recent user
      const latestUser = await prisma.user.findFirst({
        orderBy: { createdAt: 'desc' }
      });
      if (!latestUser) {
        return res.json({ medications: [] });
      }
      userId = latestUser.id;
    }
    
    // Find patient associated with user
    const patient = await prisma.patient.findFirst({
      where: { userId }
    });
    
    console.log('Patient found:', patient ? patient.id : 'NOT FOUND');
    
    if (!patient) {
      console.log('No patient found for user:', userId);
      return res.json({ medications: [] });
    }
    
    // Get all medications for this patient (including inactive for debugging)
    const allMeds = await prisma.patientMedication.findMany({
      where: {
        patientId: patient.id
      },
      orderBy: {
        start_date: 'desc'
      }
    });
    
    console.log('Total medications in DB for patient:', allMeds.length);
    allMeds.forEach(m => console.log('  -', m.name, m.status, m.id));
    
    // Filter to active medications
    const medications = allMeds.filter(m => m.status === 'active');
    console.log('Active medications:', medications.length);
    
    // Transform to match frontend expected format
    const transformedMedications = medications.map(med => {
      // Parse dosage to extract strength and unit if possible
      const dosageMatch = med.dosage?.match(/^(\d+(?:\.\d+)?)(\w+)$/);
      const strength = dosageMatch ? dosageMatch[1] : null;
      const unit = dosageMatch ? dosageMatch[2] : null;
      
      return {
        id: med.id,
        medication_name: med.name,
        name: med.name,
        generic_name: med.name,
        strength: strength,
        unit: unit,
        dosage: med.dosage,
        frequency: med.frequency,
        frequency_display: med.frequency === 'daily' ? 'Once daily' :
                          med.frequency === 'twice_daily' ? 'Twice daily' :
                          med.frequency === 'three_times_daily' ? 'Three times daily' :
                          med.frequency === 'four_times_daily' ? 'Four times daily' :
                          med.frequency === 'weekly' ? 'Once weekly' :
                          med.frequency === 'monthly' ? 'Once monthly' :
                          med.frequency === 'as_needed' ? 'As needed' :
                          med.frequency,
        start_date: med.start_date.toISOString().split('T')[0],
        end_date: med.end_date ? med.end_date.toISOString().split('T')[0] : null,
        status: med.status,
        route: med.route,
        createdAt: med.created_at.toISOString()
      };
    });
    
    res.json({ medications: transformedMedications });
  } catch (error) {
    console.error('Error fetching medications:', error);
    res.status(500).json({ error: 'Failed to fetch medications', details: error.message });
  }
});

app.get('/api/meds/schedule', (req, res) => {
  res.json([]);
});

app.get('/api/meds/cycles', (req, res) => {
  res.json([]);
});

// This endpoint is now defined above before the /api/medications route

// Save medication endpoint
app.post('/api/meds/user', async (req, res) => {
  try {
    const medicationData = req.body;
    console.log('Saving medication:', medicationData);
    
    // Get user ID from request body or find the most recent user (for demo)
    // In production, get from JWT token
    let userId = medicationData.userId;
    if (!userId) {
      // For demo: get the most recent user
      const latestUser = await prisma.user.findFirst({
        orderBy: { createdAt: 'desc' }
      });
      if (!latestUser) {
        return res.status(400).json({ success: false, error: 'User not found' });
      }
      userId = latestUser.id;
    }
    
    // Find patient associated with user
    const patient = await prisma.patient.findFirst({
      where: { userId }
    });
    
    if (!patient) {
      return res.status(400).json({ success: false, error: 'Patient profile not found for user' });
    }
    
    // Parse strength and unit
    const strength = medicationData.strength || '';
    const unit = medicationData.unit || '';
    const dosage = strength && unit ? `${strength}${unit}` : (medicationData.dosage || 'As directed');
    
    // Parse frequency
    const frequency = medicationData.frequency || 'daily';
    
    // Create medication in database
    console.log('Creating medication with data:', {
      patientId: patient.id,
      name: medicationData.medication_name || medicationData.generic_name || 'Unknown Medication',
      dosage: dosage,
      frequency: frequency,
      start_date: medicationData.start_date ? new Date(medicationData.start_date) : new Date()
    });
    
    const savedMedication = await prisma.patientMedication.create({
      data: {
        patientId: patient.id,
        name: medicationData.medication_name || medicationData.generic_name || 'Unknown Medication',
        dosage: dosage,
        frequency: frequency,
        route: 'oral', // Default route
        start_date: medicationData.start_date ? new Date(medicationData.start_date) : new Date(),
        status: 'active',
        manually_entered: true
      }
    });
    
    console.log('✅ Medication saved successfully:', savedMedication.id, savedMedication.name);
    
    // Verify it was saved
    const verifyMed = await prisma.patientMedication.findUnique({
      where: { id: savedMedication.id }
    });
    console.log('✅ Verified medication in database:', verifyMed ? 'Found' : 'NOT FOUND');
    
    res.status(201).json({
      success: true,
      message: 'Medication saved successfully',
      medication: {
        id: savedMedication.id,
        medication_name: savedMedication.name,
        generic_name: savedMedication.name,
        strength: strength,
        unit: unit,
        dosage: savedMedication.dosage,
        frequency: savedMedication.frequency,
        frequency_display: medicationData.frequency_display || frequency,
        drug_class: medicationData.drug_class || null,
        start_date: savedMedication.start_date.toISOString().split('T')[0],
        status: savedMedication.status,
        createdAt: savedMedication.created_at.toISOString()
      }
    });
  } catch (error) {
    console.error('Error saving medication:', error);
    console.error('Error stack:', error.stack);
    res.status(500).json({ success: false, error: 'Failed to save medication', details: error.message });
  }
});

app.get('/api/metrics/user', (req, res) => {
  res.json([]);
});

app.get('/api/health-metrics', (req, res) => {
  res.json([]);
});

app.get('/api/medication-schedules', (req, res) => {
  res.json([]);
});

app.get('/api/ai/health-report', (req, res) => {
  res.json({
    status: 'success',
    report: {
      summary: 'AI health analysis not available in demo mode',
      recommendations: [],
      riskFactors: [],
      lastUpdated: new Date().toISOString()
    }
  });
});

app.get('/api/ai/assistant/status', (req, res) => {
  res.json({
    status: 'available',
    capabilities: ['health_analysis', 'medication_review', 'symptom_assessment']
  });
});

app.get('/api/ai/models', (req, res) => {
  res.json([
    {
      id: 'mock-model-1',
      name: 'Health Analysis Model',
      status: 'available',
      description: 'Mock AI model for health analysis'
    }
  ]);
});

app.get('/api/ai/assistant', (req, res) => {
  res.json({
    status: 'available',
    message: 'AI Assistant is ready to help with your health questions'
  });
});

// Add any other missing endpoints that might be called
app.get('/api/*', (req, res) => {
  res.json({ 
    message: 'Endpoint not implemented in demo mode',
    path: req.path,
    method: req.method
  });
});

// Export app for Supabase Edge Functions deployment
// If running standalone, start the server
if (require.main === module) {
  const PORT = process.env.PORT || 8080;
  app.listen(PORT, () => {
    console.log(`🚀 Simple backend server running on port ${PORT}`);
    console.log(`📊 Test endpoint: http://localhost:${PORT}/api/test-public`);
    console.log(`❤️  Health check: http://localhost:${PORT}/api/health`);
  });
}

// Export app for use in Supabase Edge Functions
module.exports = app;

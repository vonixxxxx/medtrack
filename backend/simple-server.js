const express = require('express');
const cors = require('cors');
const { PrismaClient } = require('@prisma/client');
require('dotenv').config();

const app = express();
const prisma = new PrismaClient();

// Simple in-memory store for survey completion status
const surveyCompletionStatus = new Map();

// Middleware
app.use(cors());
app.use(express.json());

// Import and use AI routes
const aiRoutes = require('./src/routes/ai');
app.use('/api/ai', aiRoutes);

// Test endpoint
app.get('/api/test-public', (req, res) => {
  res.json({ message: 'Backend is running!' });
});

// Auth endpoints
app.get('/api/auth/me', (req, res) => {
  // For demo purposes, return a mock user
  // In a real app, you'd verify the JWT token and return actual user data
  res.json({
    id: 'demo-user-id',
    email: 'demo@example.com',
    name: 'Demo User',
    role: 'patient',
    hospitalCode: '123456789'
  });
});

// Auth routes
app.post('/api/auth/signup', async (req, res) => {
  try {
    const { email, password, role, hospitalCode, patientData } = req.body;
    
    // Create user
    const user = await prisma.user.create({
      data: {
        email,
        password, // In production, hash this
        role: role || 'patient',
        hospitalCode: hospitalCode || '123456789',
        name: patientData?.name || null
      }
    });

    // Create patient profile if role is patient
    if (role === 'patient' && patientData) {
      // Remove name from patientData since it goes in User
      const { name, ...patientFields } = patientData;
      
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

    // Initialize survey completion status for new user
    surveyCompletionStatus.set(user.id, false);

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
    const patients = await prisma.patient.findMany({
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

    const transformedPatients = patients.map(patient => {
      const age = patient.dob ? 
        Math.floor((new Date() - new Date(patient.dob)) / (365.25 * 24 * 60 * 60 * 1000)) : null;
      
      return {
        id: patient.id,
        userId: patient.userId,
        name: patient.user.name || `Patient ${patient.id}`,
        email: patient.user.email,
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
      return res.status(400).json({ error: 'Patient ID and medical notes are required' });
    }

    console.log(`🔍 Parsing medical history for patient ${patientId}`);
    console.log(`📝 Medical notes: ${medicalNotes.substring(0, 100)}...`);

    // Enhanced AI parsing with comprehensive condition detection
    const text = medicalNotes.toLowerCase();
    const parsedData = {};

    // Diabetes detection
    if (text.includes('type 2 diabetes') || text.includes('t2dm') || text.includes('diabetes mellitus type 2')) {
      parsedData.t2dm = true;
      parsedData.diabetes_type = 'Type 2';
    }
    if (text.includes('type 1 diabetes') || text.includes('t1dm') || text.includes('diabetes mellitus type 1')) {
      parsedData.diabetes_type = 'Type 1';
    }
    if (text.includes('prediabetes') || text.includes('impaired glucose tolerance') || text.includes('impaired fasting glucose')) {
      parsedData.prediabetes = true;
    }

    // Cardiovascular conditions
    if (text.includes('ascvd') || text.includes('atherosclerotic') || text.includes('cardiovascular disease')) {
      parsedData.ascvd = true;
    }
    if (text.includes('hypertension') || text.includes('htn') || text.includes('high blood pressure')) {
      parsedData.htn = true;
      parsedData.hypertension = true;
    }
    if (text.includes('dyslipidaemia') || text.includes('cholesterol') || text.includes('hyperlipidemia') || text.includes('hypercholesterolemia')) {
      parsedData.dyslipidaemia = true;
    }
    if (text.includes('ischaemic heart disease') || text.includes('coronary artery disease') || text.includes('cad')) {
      parsedData.ischaemic_heart_disease = true;
    }
    if (text.includes('heart failure') || text.includes('congestive heart failure') || text.includes('chf')) {
      parsedData.heart_failure = true;
    }
    if (text.includes('cerebrovascular disease') || text.includes('stroke') || text.includes('cva')) {
      parsedData.cerebrovascular_disease = true;
    }
    if (text.includes('pulmonary hypertension') || text.includes('pah')) {
      parsedData.pulmonary_hypertension = true;
    }
    if (text.includes('dvt') || text.includes('deep vein thrombosis')) {
      parsedData.dvt = true;
    }
    if (text.includes('pe') || text.includes('pulmonary embolism')) {
      parsedData.pe = true;
    }

    // Sleep and respiratory conditions
    if (text.includes('osa') || text.includes('sleep apnoea') || text.includes('obstructive sleep apnea')) {
      parsedData.osa = true;
    }
    if (text.includes('sleep studies') || text.includes('polysomnography')) {
      parsedData.sleep_studies = true;
    }
    if (text.includes('cpap') || text.includes('continuous positive airway pressure')) {
      parsedData.cpap = true;
    }
    if (text.includes('asthma') || text.includes('bronchial asthma')) {
      parsedData.asthma = true;
    }

    // Gastrointestinal conditions
    if (text.includes('gord') || text.includes('gerd') || text.includes('gastroesophageal reflux')) {
      parsedData.gord = true;
    }

    // Renal conditions
    if (text.includes('ckd') || text.includes('chronic kidney disease') || text.includes('renal failure')) {
      parsedData.ckd = true;
    }
    if (text.includes('kidney stones') || text.includes('nephrolithiasis') || text.includes('renal calculi')) {
      parsedData.kidney_stones = true;
    }

    // Metabolic conditions
    if (text.includes('masld') || text.includes('nafld') || text.includes('fatty liver') || text.includes('steatohepatitis')) {
      parsedData.masld = true;
    }

    // Reproductive conditions
    if (text.includes('infertility') || text.includes('infertile')) {
      parsedData.infertility = true;
    }
    if (text.includes('pcos') || text.includes('polycystic ovary syndrome')) {
      parsedData.pcos = true;
    }

    // Mental health conditions
    if (text.includes('anxiety') || text.includes('anxious') || text.includes('panic')) {
      parsedData.anxiety = true;
    }
    if (text.includes('depression') || text.includes('depressed') || text.includes('mood disorder')) {
      parsedData.depression = true;
    }
    if (text.includes('bipolar') || text.includes('manic depression')) {
      parsedData.bipolar_disorder = true;
    }
    if (text.includes('emotional eating') || text.includes('binge eating')) {
      parsedData.emotional_eating = true;
    }
    if (text.includes('schizoaffective')) {
      parsedData.schizoaffective_disorder = true;
    }

    // Musculoskeletal conditions
    if (text.includes('osteoarthritis knee') || text.includes('oa knee') || text.includes('knee arthritis')) {
      parsedData.oa_knee = true;
    }
    if (text.includes('osteoarthritis hip') || text.includes('oa hip') || text.includes('hip arthritis')) {
      parsedData.oa_hip = true;
    }
    if (text.includes('limited mobility') || text.includes('mobility issues')) {
      parsedData.limited_mobility = true;
    }
    if (text.includes('lymphoedema') || text.includes('lymphedema')) {
      parsedData.lymphoedema = true;
    }

    // Endocrine conditions
    if (text.includes('thyroid') || text.includes('hypothyroidism') || text.includes('hyperthyroidism')) {
      parsedData.thyroid_disorder = true;
    }

    // Neurological conditions
    if (text.includes('iih') || text.includes('idiopathic intracranial hypertension') || text.includes('pseudotumor cerebri')) {
      parsedData.iih = true;
    }
    if (text.includes('epilepsy') || text.includes('seizure')) {
      parsedData.epilepsy = true;
    }
    if (text.includes('functional neurological disorder') || text.includes('fnd')) {
      parsedData.functional_neurological_disorder = true;
    }

    // Oncology
    if (text.includes('cancer') || text.includes('malignancy') || text.includes('tumor') || text.includes('neoplasm')) {
      parsedData.cancer = true;
    }

    // Bariatric surgery
    if (text.includes('gastric band') || text.includes('lap band')) {
      parsedData.bariatric_gastric_band = true;
    }
    if (text.includes('sleeve') || text.includes('sleeve gastrectomy')) {
      parsedData.bariatric_sleeve = true;
    }
    if (text.includes('bypass') || text.includes('gastric bypass') || text.includes('roux-en-y')) {
      parsedData.bariatric_bypass = true;
    }
    if (text.includes('balloon') || text.includes('gastric balloon')) {
      parsedData.bariatric_balloon = true;
    }

    // Extract numerical values with improved patterns
    const hba1cMatch = text.match(/hba1c[:\s]*([0-9.]+)(?:\s*%)?/i);
    if (hba1cMatch) {
      parsedData.baseline_hba1c = parseFloat(hba1cMatch[1]);
    }

    const bmiMatch = text.match(/bmi[:\s]*([0-9.]+)/i);
    if (bmiMatch) {
      parsedData.baseline_bmi = parseFloat(bmiMatch[1]);
    }

    const weightMatch = text.match(/weight[:\s]*([0-9.]+)\s*kg/i);
    if (weightMatch) {
      parsedData.baseline_weight = parseFloat(weightMatch[1]);
    }

    const heightMatch = text.match(/height[:\s]*([0-9.]+)\s*cm/i);
    if (heightMatch) {
      parsedData.height = parseFloat(heightMatch[1]);
    }

    // Blood pressure
    const bpMatch = text.match(/blood pressure[:\s]*([0-9]+)\/([0-9]+)/i);
    if (bpMatch) {
      parsedData.systolic_bp = parseInt(bpMatch[1]);
      parsedData.diastolic_bp = parseInt(bpMatch[2]);
    }

    // Lipid values
    const tcMatch = text.match(/total cholesterol[:\s]*([0-9.]+)/i);
    if (tcMatch) {
      parsedData.baseline_tc = parseFloat(tcMatch[1]);
    }

    const hdlMatch = text.match(/hdl[:\s]*([0-9.]+)/i);
    if (hdlMatch) {
      parsedData.baseline_hdl = parseFloat(hdlMatch[1]);
    }

    const ldlMatch = text.match(/ldl[:\s]*([0-9.]+)/i);
    if (ldlMatch) {
      parsedData.baseline_ldl = parseFloat(ldlMatch[1]);
    }

    const tgMatch = text.match(/triglycerides?[:\s]*([0-9.]+)/i);
    if (tgMatch) {
      parsedData.baseline_tg = parseFloat(tgMatch[1]);
    }

    // Get current patient data for comparison
    const currentPatient = await prisma.patient.findUnique({
      where: { id: patientId }
    });

    if (!currentPatient) {
      return res.status(404).json({ error: 'Patient not found' });
    }

    const auditLogs = [];
    const updates = {};

    // Compare parsed data with current data
    for (const [field, value] of Object.entries(parsedData)) {
      if (value !== null && value !== undefined) {
        const currentValue = currentPatient[field];
        if (currentValue !== value) {
          auditLogs.push({
            patientId,
            field_name: field,
            old_value: currentValue?.toString() || null,
            new_value: value.toString(),
            ai_confidence: 0.85,
            ai_suggestion: `AI detected: ${value}`,
            clinician_approved: false
          });
          updates[field] = value;
        }
      }
    }

    // Save audit logs
    if (auditLogs.length > 0) {
      await prisma.aiAuditLog.createMany({
        data: auditLogs
      });
    }

    // Extract conditions for the conditions table
    const conditions = [];
    const conditionKeywords = [
      'diabetes', 'hypertension', 'asthma', 'anxiety', 'depression', 'arthritis', 'thyroid', 'cancer'
    ];
    
    for (const keyword of conditionKeywords) {
      if (text.includes(keyword)) {
        conditions.push(keyword);
      }
    }

    // Save conditions
    const savedConditions = await Promise.all(
      conditions.map(condition => 
        prisma.condition.create({
          data: {
            patientId: patientId,
            name: condition,
            normalized: condition.toLowerCase()
          }
        }).catch(() => null) // Ignore duplicates
      )
    );

    console.log(`✅ Parsed ${Object.keys(parsedData).length} fields, found ${auditLogs.length} updates`);

    res.json({
      success: true,
      parsedData,
      updates,
      auditLogs: auditLogs.length,
      conditions: savedConditions.filter(c => c).map(c => ({
        id: c.id,
        name: c.name,
        normalized: c.normalized
      })),
      message: `Found ${auditLogs.length} potential updates requiring review`
    });
  } catch (error) {
    console.error('Error parsing medical history:', error);
    res.status(500).json({ error: 'Failed to parse medical history' });
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

app.get('/api/auth/survey-status', (req, res) => {
  try {
    // For demo purposes, check if any user has completed the survey
    // In a real app, you'd get specific user ID from JWT token
    const hasCompletedSurvey = Array.from(surveyCompletionStatus.values()).some(status => status === true);
    
    console.log('Survey status requested - completed:', hasCompletedSurvey);
    console.log('Survey completion status map:', Object.fromEntries(surveyCompletionStatus));
    
    res.json({ 
      surveyCompleted: hasCompletedSurvey,
      lastCompleted: hasCompletedSurvey ? new Date().toISOString() : null
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
    
    if (latestPatient) {
      // Update patient record with survey data
      const updateData = {
        // Basic demographics
        name: surveyData.name,
        dob: surveyData.dateOfBirth ? new Date(surveyData.dateOfBirth) : null,
        sex: surveyData.biologicalSex,
        ethnicity: surveyData.ethnicity,
        ethnic_group: surveyData.ethnicity,
        location: surveyData.location,
        postcode: surveyData.postcode,
        nhs_number: surveyData.nhsNumber,
        
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
      
      await prisma.patient.update({
        where: { id: latestPatient.id },
        data: updateData
      });
      
      console.log('Survey data saved to patient record:', latestPatient.id);
    }
    
    res.json({ 
      success: true,
      message: 'Survey data saved successfully'
    });
  } catch (error) {
    console.error('Error saving survey data:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to save survey data'
    });
  }
});

app.put('/api/auth/complete-survey', (req, res) => {
  try {
    console.log('Survey completion requested');
    
    // Mark survey as completed for all users (demo purposes)
    // In a real app, you'd get specific user ID from JWT token
    for (const [userId, status] of surveyCompletionStatus.entries()) {
      surveyCompletionStatus.set(userId, true);
    }
    
    // Also add a global completion flag for demo purposes
    surveyCompletionStatus.set('global', true);

    console.log('Survey marked as completed for all users');
    console.log('Updated survey completion status map:', Object.fromEntries(surveyCompletionStatus));

    res.json({ 
      success: true,
      message: 'Survey marked as completed'
    });
  } catch (error) {
    console.error('Error completing survey:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to mark survey as completed'
    });
  }
});

app.get('/api/meds/user', (req, res) => {
  res.json([]);
});

app.get('/api/meds/schedule', (req, res) => {
  res.json([]);
});

app.get('/api/meds/cycles', (req, res) => {
  res.json([]);
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

const PORT = process.env.PORT || 4000;

app.listen(PORT, () => {
  console.log(`🚀 Simple backend server running on port ${PORT}`);
  console.log(`📊 Test endpoint: http://localhost:${PORT}/api/test-public`);
});

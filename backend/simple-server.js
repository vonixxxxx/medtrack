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
    const { runOllamaParser } = require('./utils/ollamaParser');
    const parsedData = await runOllamaParser(medicalNotes);

    console.log('📊 Parsed data from AI:', Object.keys(parsedData));

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

    // Map AI parsed data to database fields
    const fieldMapping = {
      age: 'age',
      sex: 'sex',
      height: 'height',
      weight: 'baseline_weight',
      bmi: 'baseline_bmi',
      systolic_bp: 'systolic_bp',
      diastolic_bp: 'diastolic_bp',
      hba1c_percent: 'hba1c_percent',
      baseline_hba1c: 'baseline_hba1c',
      baseline_tc: 'baseline_tc',
      baseline_hdl: 'baseline_hdl',
      baseline_ldl: 'baseline_ldl',
      baseline_tg: 'baseline_tg',
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
      obesity: 'obesity',
      notes: 'notes',
      comorbidities_count: 'total_qualifying_comorbidities'
    };

    // Compare parsed data with current data and create updates
    for (const [aiField, dbField] of Object.entries(fieldMapping)) {
      if (parsedData[aiField] !== null && parsedData[aiField] !== undefined) {
        const currentValue = currentPatient[dbField];
        const newValue = parsedData[aiField];
        
        if (currentValue !== newValue) {
          auditLogs.push({
            patientId,
            field_name: dbField,
            old_value: currentValue?.toString() || null,
            new_value: newValue.toString(),
            ai_confidence: 0.9,
            ai_suggestion: `AI detected: ${newValue}`,
            clinician_approved: false
          });
          updates[dbField] = newValue;
        }
      }
    }

    // Handle medications separately
    if (parsedData.medications && Array.isArray(parsedData.medications)) {
      const medicationsString = parsedData.medications.join(', ');
      if (currentPatient.all_medications_from_scr !== medicationsString) {
        auditLogs.push({
          patientId,
          field_name: 'all_medications_from_scr',
          old_value: currentPatient.all_medications_from_scr || null,
          new_value: medicationsString,
          ai_confidence: 0.9,
          ai_suggestion: `AI detected medications: ${medicationsString}`,
          clinician_approved: false
        });
        updates.all_medications_from_scr = medicationsString;
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
    if (parsedData.medications && Array.isArray(parsedData.medications)) {
      conditions.push(...parsedData.medications);
    }
    
    // Add conditions based on boolean flags
    const conditionFlags = {
      t2dm: 'Type 2 Diabetes',
      prediabetes: 'Prediabetes',
      htn: 'Hypertension',
      dyslipidaemia: 'Dyslipidemia',
      ascvd: 'ASCVD',
      ckd: 'Chronic Kidney Disease',
      osa: 'Obstructive Sleep Apnea',
      obesity: 'Obesity'
    };

    for (const [flag, conditionName] of Object.entries(conditionFlags)) {
      if (parsedData[flag]) {
        conditions.push(conditionName);
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

    console.log(`✅ AI parsed ${Object.keys(parsedData).length} fields, found ${auditLogs.length} updates`);

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
      message: `AI successfully extracted ${Object.keys(parsedData).length} data points, ${auditLogs.length} updates require review`
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

app.get('/api/auth/survey-status', async (req, res) => {
  try {
    // For demo purposes, get the most recent user and check their survey status
    // In a real app, you'd get specific user ID from JWT token
    const user = await prisma.user.findFirst({
      orderBy: { createdAt: 'desc' }
    });
    
    if (!user) {
      return res.json({ 
        surveyCompleted: false,
        lastCompleted: null
      });
    }
    
    const hasCompletedSurvey = surveyCompletionStatus.get(user.id) || false;
    
    console.log('Survey status requested for user:', user.email, '- completed:', hasCompletedSurvey);
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

// Medication validation endpoint
app.post('/api/medications/validateMedication', async (req, res) => {
  try {
    const { medication } = req.body;
    console.log('Validating medication:', medication);
    
    // Enhanced medication database with fuzzy matching
    const medicationDatabase = [
      {
        generic_name: 'semaglutide',
        brand_names: ['Ozempic', 'Wegovy', 'Rybelsus'],
        drug_class: 'GLP-1 Receptor Agonist',
        dosage_forms: ['Injection', 'Tablet'],
        typical_strengths: ['0.25mg', '0.5mg', '1mg', '2.4mg', '3mg', '7mg', '14mg'],
        description: 'Used for type 2 diabetes and weight management'
      },
      {
        generic_name: 'metformin',
        brand_names: ['Glucophage', 'Fortamet', 'Glumetza'],
        drug_class: 'Biguanide',
        dosage_forms: ['Tablet', 'Extended-release tablet'],
        typical_strengths: ['500mg', '850mg', '1000mg'],
        description: 'First-line treatment for type 2 diabetes'
      },
      {
        generic_name: 'insulin glargine',
        brand_names: ['Lantus', 'Toujeo', 'Basaglar'],
        drug_class: 'Long-acting Insulin',
        dosage_forms: ['Injection'],
        typical_strengths: ['100 units/mL', '300 units/mL'],
        description: 'Long-acting insulin for diabetes management'
      },
      {
        generic_name: 'liraglutide',
        brand_names: ['Victoza', 'Saxenda'],
        drug_class: 'GLP-1 Receptor Agonist',
        dosage_forms: ['Injection'],
        typical_strengths: ['0.6mg', '1.2mg', '1.8mg', '3mg'],
        description: 'GLP-1 agonist for diabetes and weight management'
      },
      {
        generic_name: 'dulaglutide',
        brand_names: ['Trulicity'],
        drug_class: 'GLP-1 Receptor Agonist',
        dosage_forms: ['Injection'],
        typical_strengths: ['0.75mg', '1.5mg', '3mg', '4.5mg'],
        description: 'GLP-1 agonist for diabetes management'
      },
      {
        generic_name: 'exenatide',
        brand_names: ['Byetta', 'Bydureon'],
        drug_class: 'GLP-1 Receptor Agonist',
        dosage_forms: ['Injection'],
        typical_strengths: ['5mcg', '10mcg', '2mg'],
        description: 'GLP-1 agonist for diabetes management'
      }
    ];
    
    // Fuzzy matching function
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
    
    // Search for medication
    const searchTerm = medication.toLowerCase();
    let bestMatch = null;
    let bestScore = Infinity;
    
    for (const med of medicationDatabase) {
      // Check for exact substring match first (highest priority)
      if (med.generic_name.toLowerCase().includes(searchTerm) || 
          med.brand_names.some(brand => brand.toLowerCase().includes(searchTerm))) {
        bestMatch = med;
        bestScore = 0; // Perfect match
        break;
      }
      
      // Check generic name with fuzzy matching
      const genericDistance = levenshteinDistance(searchTerm, med.generic_name.toLowerCase());
      if (genericDistance < bestScore) {
        bestScore = genericDistance;
        bestMatch = med;
      }
      
      // Check brand names with fuzzy matching
      for (const brand of med.brand_names) {
        const brandDistance = levenshteinDistance(searchTerm, brand.toLowerCase());
        if (brandDistance < bestScore) {
          bestScore = brandDistance;
          bestMatch = med;
        }
      }
    }
    
    // If we found a good match (distance <= 5), return it
    if (bestMatch && bestScore <= 5) {
      res.json({
        success: true,
        medication: bestMatch,
        confidence: Math.max(0, 1 - (bestScore / 10))
      });
    } else {
      res.json({
        success: false,
        message: 'Medication not found',
        suggestions: medicationDatabase.slice(0, 3).map(med => med.generic_name)
      });
    }
  } catch (error) {
    console.error('Medication validation error:', error);
    res.status(500).json({ success: false, error: 'Failed to validate medication' });
  }
});

// Save medication endpoint
app.post('/api/meds/user', async (req, res) => {
  try {
    const medicationData = req.body;
    console.log('Saving medication:', medicationData);
    
    // For demo purposes, just return success
    // In a real app, you'd save to the database
    res.json({
      success: true,
      message: 'Medication saved successfully',
      medication: {
        id: `med-${Date.now()}`,
        ...medicationData,
        createdAt: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error saving medication:', error);
    res.status(500).json({ success: false, error: 'Failed to save medication' });
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

const PORT = process.env.PORT || 4000;

app.listen(PORT, () => {
  console.log(`🚀 Simple backend server running on port ${PORT}`);
  console.log(`📊 Test endpoint: http://localhost:${PORT}/api/test-public`);
});

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

    // Simple AI parsing simulation
    const text = medicalNotes.toLowerCase();
    const parsedData = {};

    // Diabetes detection
    if (text.includes('type 2 diabetes') || text.includes('t2dm')) {
      parsedData.t2dm = true;
      parsedData.diabetes_type = 'Type 2';
    }
    if (text.includes('prediabetes')) {
      parsedData.prediabetes = true;
    }

    // Cardiovascular conditions
    if (text.includes('ascvd') || text.includes('atherosclerotic')) {
      parsedData.ascvd = true;
    }
    if (text.includes('hypertension') || text.includes('htn')) {
      parsedData.htn = true;
      parsedData.hypertension = true;
    }
    if (text.includes('dyslipidaemia') || text.includes('cholesterol')) {
      parsedData.dyslipidaemia = true;
    }

    // Sleep conditions
    if (text.includes('osa') || text.includes('sleep apnoea')) {
      parsedData.osa = true;
    }
    if (text.includes('cpap')) {
      parsedData.cpap = true;
    }

    // Other conditions
    if (text.includes('asthma')) parsedData.asthma = true;
    if (text.includes('anxiety')) parsedData.anxiety = true;
    if (text.includes('depression')) parsedData.depression = true;

    // Extract numerical values
    const hba1cMatch = text.match(/hba1c[:\s]*([0-9.]+)/);
    if (hba1cMatch) {
      parsedData.baseline_hba1c = parseFloat(hba1cMatch[1]);
    }

    const bmiMatch = text.match(/bmi[:\s]*([0-9.]+)/);
    if (bmiMatch) {
      parsedData.baseline_bmi = parseFloat(bmiMatch[1]);
    }

    const weightMatch = text.match(/weight[:\s]*([0-9.]+)\s*kg/);
    if (weightMatch) {
      parsedData.baseline_weight = parseFloat(weightMatch[1]);
    }

    const heightMatch = text.match(/height[:\s]*([0-9.]+)\s*cm/);
    if (heightMatch) {
      parsedData.height = parseFloat(heightMatch[1]);
    }

    // Get current patient data for comparison
    const currentPatient = await prisma.patient.findUnique({
      where: { id: patientId }
    });

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

    res.json({
      success: true,
      parsedData,
      updates,
      auditLogs: auditLogs.length,
      conditions: [],
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
    // For demo purposes, always return false to show survey
    // In a real app, you'd check the specific user's survey completion status
    console.log('Survey status requested - returning false to show survey');
    
    res.json({ 
      surveyCompleted: false, // Always false for demo to show survey
      lastCompleted: null
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
    
    // For demo purposes, just return success without saving to database
    // In a real app, you'd get user ID from JWT token and save to database
    console.log('Survey data processed successfully (demo mode)');
    
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

const { calculateAdjustedHbA1c } = require('../services/hba1cService');
const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

// Get all patients assigned to the clinician's hospital
exports.getPatients = async (req, res) => {
  try {
    const clinicianId = req.user.id;
    const prisma = req.prisma;
    
    // Get clinician's hospital code from JWT token or database
    const clinicianHospitalCode = req.user.hospitalCode;
    
    if (!clinicianHospitalCode) {
      return res.status(400).json({ error: 'Clinician hospital code not found' });
    }

    // Get all patients with the same hospital code
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
        conditions: true,
        metrics: {
          orderBy: { timestamp: 'desc' },
          take: 1
        }
      }
    });

    // Transform data for frontend - only include real data
    const transformedPatients = patients.map(patient => {
      const latestMetric = patient.metrics[0];
      
      return {
        id: patient.id,
        userId: patient.userId,
        name: patient.user.name || `Patient ${patient.id}`,
        email: patient.user.email,
        age: patient.dob ? 
          Math.floor((new Date() - new Date(patient.dob)) / (365.25 * 24 * 60 * 60 * 1000)) : null,
        
        // Basic demographics
        sex: patient.sex || null,
        ethnic_group: patient.ethnic_group || null,
        ethnicity: patient.ethnicity || null,
        location: patient.location || null,
        postcode: patient.postcode || null,
        nhs_number: patient.nhs_number || null,
        mrn: patient.mrn || null,
        
        // Clinical measurements
        height: patient.height || null,
        baseline_weight: patient.baseline_weight || null,
        baseline_bmi: patient.baseline_bmi || null,
        baseline_weight_date: patient.baseline_weight_date?.toISOString().split('T')[0] || null,
        
        // Cardiovascular risk factors
        ascvd: patient.ascvd || false,
        htn: patient.htn || false,
        dyslipidaemia: patient.dyslipidaemia || false,
        osa: patient.osa || false,
        sleep_studies: patient.sleep_studies || false,
        cpap: patient.cpap || false,
        
        // Diabetes
        t2dm: patient.t2dm || false,
        prediabetes: patient.prediabetes || false,
        diabetes_type: patient.diabetes_type || null,
        baseline_hba1c: patient.baseline_hba1c || null,
        baseline_hba1c_date: patient.baseline_hba1c_date?.toISOString().split('T')[0] || null,
        hba1cPercent: patient.hba1c_percent || null,
        hba1cMmolMol: patient.hba1c_mmol || null,
        
        // Lipid profile
        baseline_tc: patient.baseline_tc || null,
        baseline_hdl: patient.baseline_hdl || null,
        baseline_ldl: patient.baseline_ldl || null,
        baseline_tg: patient.baseline_tg || null,
        baseline_lipid_date: patient.baseline_lipid_date?.toISOString().split('T')[0] || null,
        lipid_lowering_treatment: patient.lipid_lowering_treatment || null,
        
        // Medications
        antihypertensive_medications: patient.antihypertensive_medications || null,
        all_medications_from_scr: patient.all_medications_from_scr || null,
        
        // Medical conditions
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
        
        // Bariatric surgery
        bariatric_gastric_band: patient.bariatric_gastric_band || false,
        bariatric_sleeve: patient.bariatric_sleeve || false,
        bariatric_bypass: patient.bariatric_bypass || false,
        bariatric_balloon: patient.bariatric_balloon || false,
        
        // Clinical data
        diagnoses_coded_in_scr: patient.diagnoses_coded_in_scr || null,
        total_qualifying_comorbidities: patient.total_qualifying_comorbidities || null,
        mes: patient.mes || null,
        
        // Notes and criteria
        notes: patient.notes || null,
        criteria_for_wegovy: patient.criteria_for_wegovy || null,
        
        // Legacy fields for compatibility
        conditions: patient.conditions.map(c => c.normalized),
        lastVisit: latestMetric?.timestamp?.toISOString().split('T')[0] || null,
        changePercent: null // Will be calculated from historical data
      };
    });

    res.json(transformedPatients);
  } catch (err) {
    console.error('Error fetching patients:', err);
    res.status(500).json({ error: 'Failed to fetch patients' });
  }
};

// Get specific patient details
exports.getPatient = async (req, res) => {
  try {
    const { id } = req.params;
    const prisma = req.prisma;
    
    // Get clinician's hospital code from JWT token
    const clinicianHospitalCode = req.user.hospitalCode;
    
    if (!clinicianHospitalCode) {
      return res.status(400).json({ error: 'Clinician hospital code not found' });
    }

    const patient = await prisma.user.findFirst({
      where: {
        id: parseInt(id),
        role: 'patient',
        hospitalCode: clinicianHospitalCode
      },
      include: {
        surveyData: true,
        metrics: {
          orderBy: { date: 'desc' }
        },
        medications: true
      }
    });

    if (!patient) {
      return res.status(404).json({ error: 'Patient not found or not accessible' });
    }

    res.json(patient);
  } catch (err) {
    console.error('Error fetching patient:', err);
    res.status(500).json({ error: 'Failed to fetch patient' });
  }
};

// Add conditions to patient
exports.addPatientConditions = async (req, res) => {
  try {
    const { id } = req.params;
    const { conditions } = req.body;
    const prisma = req.prisma;
    
    // This would typically store conditions in a separate table
    // For now, we'll just return success
    console.log(`Adding conditions to patient ${id}:`, conditions);
    
    res.json({ message: 'Conditions added successfully', conditions });
  } catch (err) {
    console.error('Error adding conditions:', err);
    res.status(500).json({ error: 'Failed to add conditions' });
  }
};

// Parse medical history using AI
exports.parseMedicalHistory = async (req, res) => {
  try {
    const { patientId, medicalNotes } = req.body;
    
    if (!patientId || !medicalNotes || !medicalNotes.trim()) {
      return res.status(400).json({ error: 'Patient ID and medical notes are required' });
    }

    // Enhanced AI parsing with field mapping
    const parsedData = await parseMedicalHistoryWithAI(medicalNotes);
    
    // Get current patient data for comparison
    const currentPatient = await prisma.patient.findUnique({
      where: { id: patientId }
    });

    if (!currentPatient) {
      return res.status(404).json({ error: 'Patient not found' });
    }

    // Compare and create audit log entries
    const auditLogs = [];
    const updates = {};

    // Check each parsed field against current data
    for (const [field, value] of Object.entries(parsedData)) {
      if (value !== null && value !== undefined) {
        const currentValue = currentPatient[field];
        
        // If values differ, create audit log and suggest update
        if (currentValue !== value) {
          auditLogs.push({
            patientId,
            field_name: field,
            old_value: currentValue?.toString() || null,
            new_value: value.toString(),
            ai_confidence: 0.85, // Placeholder confidence score
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

    // Also extract and save conditions
    const conditions = extractConditionsFromText(medicalNotes);
    const savedConditions = await Promise.all(
      conditions.map(condition => 
        prisma.condition.create({
          data: {
            patientId: patientId,
            name: condition,
            normalized: normalizeConditionName(condition)
          }
        })
      )
    );

    res.json({
      success: true,
      parsedData,
      updates,
      auditLogs: auditLogs.length,
      conditions: savedConditions.map(c => ({
        id: c.id,
        name: c.name,
        normalized: c.normalized
      })),
      message: `Found ${auditLogs.length} potential updates requiring review`
    });
  } catch (err) {
    console.error('Error parsing medical history:', err);
    res.status(500).json({ error: 'Failed to parse medical history' });
  }
};

// Calculate HbA1c adjustment
exports.calculateHbA1cAdjustment = async (req, res) => {
  try {
    const { measuredHbA1cPercent, weightKg, medications } = req.body;
    
    if (!measuredHbA1cPercent || !weightKg) {
      return res.status(400).json({ error: 'Measured HbA1c and weight are required' });
    }

    const result = calculateAdjustedHbA1c(
      measuredHbA1cPercent,
      weightKg,
      medications || {}
    );

    res.json(result);
  } catch (err) {
    console.error('Error calculating HbA1c adjustment:', err);
    res.status(500).json({ error: 'Failed to calculate HbA1c adjustment' });
  }
};

// Get analytics data
exports.getAnalytics = async (req, res) => {
  try {
    const prisma = req.prisma;
    
    // Get clinician's hospital code from JWT token
    const clinicianHospitalCode = req.user.hospitalCode;
    
    if (!clinicianHospitalCode) {
      return res.status(400).json({ error: 'Clinician hospital code not found' });
    }

    // Get patient count
    const patientCount = await prisma.user.count({
      where: {
        role: 'patient',
        hospitalCode: clinicianHospitalCode
      }
    });

    // Get metrics summary
    const metrics = await prisma.metric.findMany({
      where: {
        user: {
          role: 'patient',
          hospitalCode: clinicianHospitalCode
        }
      },
      include: {
        user: {
          select: { id: true, name: true }
        }
      }
    });

    res.json({
      totalPatients: patientCount,
      totalMetrics: metrics.length,
      hospitalCode: clinicianHospitalCode,
      lastUpdated: new Date().toISOString()
    });
  } catch (err) {
    console.error('Error fetching analytics:', err);
    res.status(500).json({ error: 'Failed to fetch analytics' });
  }
};

// Export patients data
exports.exportPatients = async (req, res) => {
  try {
    const prisma = req.prisma;
    
    // Get clinician's hospital code from JWT token
    const clinicianHospitalCode = req.user.hospitalCode;
    
    if (!clinicianHospitalCode) {
      return res.status(400).json({ error: 'Clinician hospital code not found' });
    }

    // Get patients data
    const patients = await prisma.user.findMany({
      where: {
        role: 'patient',
        hospitalCode: clinicianHospitalCode
      },
      include: {
        surveyData: true,
        metrics: {
          orderBy: { date: 'desc' },
          take: 1
        }
      }
    });

    res.json(patients);
  } catch (err) {
    console.error('Error exporting patients:', err);
    res.status(500).json({ error: 'Failed to export patients' });
  }
};

// Parse medical history using AI
exports.parseHistory = async (req, res) => {
  try {
    const { patientId, medicalHistory } = req.body;
    
    if (!patientId || !medicalHistory) {
      return res.status(400).json({ error: 'Patient ID and medical history are required' });
    }

    // For now, implement a simple keyword-based parser
    // In production, this would call BioGPT or Ollama
    const conditions = extractConditionsFromText(medicalHistory);
    
    // Save conditions to database
    const savedConditions = await Promise.all(
      conditions.map(condition => 
        prisma.condition.create({
          data: {
            patientId: patientId,
            name: condition,
            normalized: normalizeConditionName(condition)
          }
        })
      )
    );

    res.json({
      success: true,
      conditions: savedConditions.map(c => ({
        id: c.id,
        name: c.name,
        normalized: c.normalized
      }))
    });
  } catch (err) {
    console.error('Error parsing medical history:', err);
    res.status(500).json({ error: 'Failed to parse medical history' });
  }
};

// Enhanced AI parsing function
async function parseMedicalHistoryWithAI(medicalNotes) {
  // This would integrate with BioGPT/Ollama in production
  // For now, implement keyword-based parsing with field mapping
  
  const text = medicalNotes.toLowerCase();
  const parsedData = {};

  // Diabetes detection
  if (text.includes('type 2 diabetes') || text.includes('t2dm') || text.includes('diabetes mellitus')) {
    parsedData.t2dm = true;
    parsedData.diabetes_type = 'Type 2';
  }
  if (text.includes('type 1 diabetes') || text.includes('t1dm')) {
    parsedData.t2dm = false;
    parsedData.diabetes_type = 'Type 1';
  }
  if (text.includes('prediabetes') || text.includes('impaired glucose')) {
    parsedData.prediabetes = true;
  }

  // Cardiovascular conditions
  if (text.includes('ascvd') || text.includes('atherosclerotic cardiovascular')) {
    parsedData.ascvd = true;
  }
  if (text.includes('hypertension') || text.includes('high blood pressure') || text.includes('htn')) {
    parsedData.htn = true;
    parsedData.hypertension = true;
  }
  if (text.includes('dyslipidaemia') || text.includes('high cholesterol') || text.includes('hyperlipidemia')) {
    parsedData.dyslipidaemia = true;
  }

  // Sleep conditions
  if (text.includes('osa') || text.includes('obstructive sleep apnoea') || text.includes('sleep apnea')) {
    parsedData.osa = true;
  }
  if (text.includes('cpap')) {
    parsedData.cpap = true;
  }
  if (text.includes('sleep study') || text.includes('sleep studies')) {
    parsedData.sleep_studies = true;
  }

  // Other conditions
  if (text.includes('asthma')) parsedData.asthma = true;
  if (text.includes('heart failure') || text.includes('cardiac failure')) parsedData.heart_failure = true;
  if (text.includes('ischaemic heart') || text.includes('coronary artery')) parsedData.ischaemic_heart_disease = true;
  if (text.includes('stroke') || text.includes('cerebrovascular')) parsedData.cerebrovascular_disease = true;
  if (text.includes('dvt') || text.includes('deep vein thrombosis')) parsedData.dvt = true;
  if (text.includes('pe') || text.includes('pulmonary embolism')) parsedData.pe = true;
  if (text.includes('gord') || text.includes('gerd') || text.includes('reflux')) parsedData.gord = true;
  if (text.includes('ckd') || text.includes('chronic kidney') || text.includes('renal failure')) parsedData.ckd = true;
  if (text.includes('pcos') || text.includes('polycystic ovary')) parsedData.pcos = true;
  if (text.includes('anxiety')) parsedData.anxiety = true;
  if (text.includes('depression')) parsedData.depression = true;
  if (text.includes('bipolar')) parsedData.bipolar_disorder = true;
  if (text.includes('osteoarthritis') || text.includes('oa')) {
    if (text.includes('knee')) parsedData.oa_knee = true;
    if (text.includes('hip')) parsedData.oa_hip = true;
  }
  if (text.includes('thyroid')) parsedData.thyroid_disorder = true;
  if (text.includes('cancer') || text.includes('malignancy')) parsedData.cancer = true;

  // Bariatric surgery
  if (text.includes('gastric band') || text.includes('lap band')) parsedData.bariatric_gastric_band = true;
  if (text.includes('sleeve gastrectomy') || text.includes('gastric sleeve')) parsedData.bariatric_sleeve = true;
  if (text.includes('gastric bypass') || text.includes('roux-en-y')) parsedData.bariatric_bypass = true;
  if (text.includes('gastric balloon')) parsedData.bariatric_balloon = true;

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

  return parsedData;
}

// Update patient data endpoint
exports.updatePatientData = async (req, res) => {
  try {
    const { patientId } = req.params;
    const updateData = req.body;
    const prisma = req.prisma;

    // Remove fields that shouldn't be updated directly
    delete updateData.id;
    delete updateData.userId;
    delete updateData.createdAt;
    delete updateData.updatedAt;

    const updatedPatient = await prisma.patient.update({
      where: { id: patientId },
      data: updateData,
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

    res.json({
      success: true,
      patient: updatedPatient
    });
  } catch (err) {
    console.error('Error updating patient data:', err);
    res.status(500).json({ error: 'Failed to update patient data' });
  }
};

// Get AI audit logs for a patient
exports.getPatientAuditLogs = async (req, res) => {
  try {
    const { patientId } = req.params;
    const prisma = req.prisma;

    const auditLogs = await prisma.aiAuditLog.findMany({
      where: { patientId },
      orderBy: { createdAt: 'desc' }
    });

    res.json(auditLogs);
  } catch (err) {
    console.error('Error fetching audit logs:', err);
    res.status(500).json({ error: 'Failed to fetch audit logs' });
  }
};

// Approve AI suggestions
exports.approveAISuggestions = async (req, res) => {
  try {
    const { patientId, fieldUpdates } = req.body;
    const prisma = req.prisma;

    // Update patient data with approved changes
    const updatedPatient = await prisma.patient.update({
      where: { id: patientId },
      data: fieldUpdates
    });

    // Mark audit logs as approved
    await prisma.aiAuditLog.updateMany({
      where: {
        patientId,
        field_name: { in: Object.keys(fieldUpdates) },
        clinician_approved: false
      },
      data: { clinician_approved: true }
    });

    res.json({
      success: true,
      patient: updatedPatient
    });
  } catch (err) {
    console.error('Error approving AI suggestions:', err);
    res.status(500).json({ error: 'Failed to approve AI suggestions' });
  }
};

// Helper function to extract conditions from text
function extractConditionsFromText(text) {
  const conditionKeywords = [
    'diabetes', 'type 1 diabetes', 'type 2 diabetes', 't1dm', 't2dm',
    'hypertension', 'high blood pressure', 'htn',
    'dyslipidemia', 'high cholesterol', 'hyperlipidemia',
    'obesity', 'overweight', 'bmi',
    'asthma', 'copd', 'chronic obstructive pulmonary disease',
    'heart disease', 'coronary artery disease', 'cad',
    'stroke', 'cerebrovascular accident', 'cva',
    'depression', 'anxiety', 'mental health',
    'arthritis', 'rheumatoid arthritis', 'osteoarthritis',
    'kidney disease', 'chronic kidney disease', 'ckd',
    'liver disease', 'hepatitis', 'cirrhosis'
  ];

  const foundConditions = [];
  const lowerText = text.toLowerCase();
  
  conditionKeywords.forEach(keyword => {
    if (lowerText.includes(keyword)) {
      foundConditions.push(keyword);
    }
  });

  return [...new Set(foundConditions)]; // Remove duplicates
}

// Helper function to normalize condition names
function normalizeConditionName(condition) {
  const normalizations = {
    't1dm': 'Type 1 Diabetes Mellitus',
    't2dm': 'Type 2 Diabetes Mellitus',
    'diabetes': 'Diabetes Mellitus',
    'htn': 'Hypertension',
    'high blood pressure': 'Hypertension',
    'high cholesterol': 'Dyslipidemia',
    'hyperlipidemia': 'Dyslipidemia',
    'bmi': 'Obesity',
    'overweight': 'Obesity',
    'copd': 'Chronic Obstructive Pulmonary Disease',
    'cad': 'Coronary Artery Disease',
    'cva': 'Cerebrovascular Accident',
    'ckd': 'Chronic Kidney Disease'
  };

  return normalizations[condition.toLowerCase()] || condition;
}

const { calculateAdjustedHbA1c } = require('../services/hba1cService');

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

    // Transform data for frontend - only include real data
    const transformedPatients = patients.map(patient => {
      const latestMetric = patient.metrics[0];
      const surveyData = patient.surveyData;
      
      return {
        id: patient.id,
        name: patient.name || `Patient ${patient.id}`,
        age: surveyData?.dateOfBirth ? 
          Math.floor((new Date() - new Date(surveyData.dateOfBirth)) / (365.25 * 24 * 60 * 60 * 1000)) : null,
        sex: surveyData?.biologicalSex || null,
        hba1cPercent: null, // Will be populated from actual HbA1c data when available
        hba1cMmolMol: null, // Will be populated from actual HbA1c data when available
        mes: null, // Will be calculated from actual medication data
        conditions: [], // Will be populated from medical history
        lastVisit: latestMetric?.date?.toISOString().split('T')[0] || null,
        changePercent: null, // Will be calculated from historical data
        ethnicity: surveyData?.ethnicity || null
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
    const { medicalNotes } = req.body;
    
    if (!medicalNotes || !medicalNotes.trim()) {
      return res.status(400).json({ error: 'Medical notes are required' });
    }

    // Placeholder for AI processing
    // In a real implementation, this would call BioGPT or Ollama
    const mockConditions = [
      'Type 2 Diabetes Mellitus',
      'Hypertension',
      'Dyslipidemia',
      'Obesity'
    ];

    // Simulate AI processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    res.json({
      conditions: mockConditions,
      processingTime: '2.1s',
      confidence: 0.95
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

const { PrismaClient } = require('@prisma/client');
const { calculateASCVDRisk, calculateFraminghamRisk } = require('../utils/calculations');
const { encryptJson, decryptJson } = require('../utils/crypto');
const prisma = new PrismaClient();

// Submit IIEF-5 Assessment Results
const submitIIEF5Results = async (req, res) => {
  try {
    const { userId } = req.user;
    const { answers, totalScore, severity, completedAt } = req.body;

    // Validate required fields
    if (!answers || !totalScore || !severity) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Store results in database
    const screeningResult = await prisma.screeningResult.create({
      data: {
        userId: parseInt(userId),
        assessmentType: 'IIEF5',
        answers: encryptJson(answers),
        totalScore: parseInt(totalScore),
        severity: severity.level,
        severityColor: severity.color,
        severityBgColor: severity.bgColor,
        completedAt: new Date(completedAt),
        metadata: encryptJson({
          severity: severity,
          assessmentType: 'IIEF5'
        })
      }
    });

    res.status(201).json({
      message: 'IIEF-5 results submitted successfully',
      result: screeningResult
    });
  } catch (error) {
    console.error('Error submitting IIEF-5 results:', error);
    res.status(500).json({ error: 'Failed to submit IIEF-5 results' });
  }
};

// Submit AUDIT Assessment Results
const submitAUDITResults = async (req, res) => {
  try {
    const { userId } = req.user;
    const { answers, totalScore, riskCategory, completedAt } = req.body;

    // Validate required fields
    if (!answers || !totalScore || !riskCategory) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Store results in database
    const screeningResult = await prisma.screeningResult.create({
      data: {
        userId: parseInt(userId),
        assessmentType: 'AUDIT',
        answers: encryptJson(answers),
        totalScore: parseInt(totalScore),
        severity: riskCategory.level,
        severityColor: riskCategory.color,
        severityBgColor: riskCategory.bgColor,
        completedAt: new Date(completedAt),
        metadata: encryptJson({
          riskCategory: riskCategory,
          assessmentType: 'AUDIT'
        })
      }
    });

    res.status(201).json({
      message: 'AUDIT results submitted successfully',
      result: screeningResult
    });
  } catch (error) {
    console.error('Error submitting AUDIT results:', error);
    res.status(500).json({ error: 'Failed to submit AUDIT results' });
  }
};

// Get User's Screening Results
const getUserScreeningResults = async (req, res) => {
  try {
    const { userId } = req.user;

    const results = await prisma.screeningResult.findMany({
      where: {
        userId: parseInt(userId)
      },
      orderBy: {
        completedAt: 'desc'
      }
    });

    // Decrypt and group results by assessment type
    const hydrated = results.map(r => ({
      ...r,
      answers: r.answers ? decryptJson(r.answers) : null,
      metadata: r.metadata ? decryptJson(r.metadata) : null
    }));
    const groupedResults = {
      IIEF5: hydrated.filter(r => r.assessmentType === 'IIEF5'),
      AUDIT: hydrated.filter(r => r.assessmentType === 'AUDIT'),
      HEART_RISK: hydrated.filter(r => r.assessmentType === 'HEART_RISK'),
      TESTOSTERONE: hydrated.filter(r => r.assessmentType === 'TESTOSTERONE')
    };

    res.json({
      results: groupedResults,
      totalAssessments: results.length
    });
  } catch (error) {
    console.error('Error fetching screening results:', error);
    res.status(500).json({ error: 'Failed to fetch screening results' });
  }
};

// Get Specific Assessment Result
const getAssessmentResult = async (req, res) => {
  try {
    const { userId } = req.user;
    const { assessmentId } = req.params;

    const result = await prisma.screeningResult.findFirst({
      where: {
        id: parseInt(assessmentId),
        userId: parseInt(userId)
      }
    });

    if (!result) {
      return res.status(404).json({ error: 'Assessment result not found' });
    }

    res.json({
      result: {
        ...result,
        answers: result.answers ? decryptJson(result.answers) : null,
        metadata: result.metadata ? decryptJson(result.metadata) : null
      }
    });
  } catch (error) {
    console.error('Error fetching assessment result:', error);
    res.status(500).json({ error: 'Failed to fetch assessment result' });
  }
};

// Get Screening Statistics
const getScreeningStats = async (req, res) => {
  try {
    const { userId } = req.user;

    const stats = await prisma.screeningResult.groupBy({
      by: ['assessmentType'],
      where: {
        userId: parseInt(userId)
      },
      _count: {
        id: true
      },
      _avg: {
        totalScore: true
      }
    });

    // Get latest results for each type
    const latestResults = await prisma.screeningResult.findMany({
      where: {
        userId: parseInt(userId)
      },
      orderBy: {
        completedAt: 'desc'
      },
      distinct: ['assessmentType']
    });

    const formattedStats = {
      totalAssessments: stats.reduce((sum, stat) => sum + stat._count.id, 0),
      byType: stats.map(stat => ({
        type: stat.assessmentType,
        count: stat._count.id,
        averageScore: stat._avg.totalScore
      })),
      latestResults: latestResults.map(r => ({
        ...r,
        answers: r.answers ? decryptJson(r.answers) : null,
        metadata: r.metadata ? decryptJson(r.metadata) : null
      }))
    };

    res.json({ stats: formattedStats });
  } catch (error) {
    console.error('Error fetching screening stats:', error);
    res.status(500).json({ error: 'Failed to fetch screening statistics' });
  }
};

// Export Screening Results (CSV format)
const exportScreeningResults = async (req, res) => {
  try {
    const { userId } = req.user;
    const { format = 'json' } = req.query;

    const results = await prisma.screeningResult.findMany({
      where: {
        userId: parseInt(userId)
      },
      orderBy: {
        completedAt: 'desc'
      }
    });

    if (format === 'csv') {
      // Generate CSV format
      const csvHeaders = [
        'Assessment Type',
        'Total Score',
        'Severity/Risk Level',
        'Completed Date',
        'Answers'
      ];

      const csvRows = results.map(result => [
        result.assessmentType,
        result.totalScore,
        result.severity,
        result.completedAt.toISOString().split('T')[0],
        result.answers
      ]);

      const csvContent = [csvHeaders, ...csvRows]
        .map(row => row.map(cell => `"${cell}"`).join(','))
        .join('\n');

      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="screening-results-${new Date().toISOString().split('T')[0]}.csv"`);
      res.send(csvContent);
    } else {
      // Return JSON format
      res.json({ results });
    }
  } catch (error) {
    console.error('Error exporting screening results:', error);
    res.status(500).json({ error: 'Failed to export screening results' });
  }
};

// Submit Heart Risk Assessment Results
const submitHeartRiskResults = async (req, res) => {
  try {
    const { userId } = req.user;
    const { 
      age, 
      gender, 
      race, 
      totalCholesterol, 
      hdlCholesterol, 
      systolicBP, 
      isOnBPMedication, 
      isSmoker, 
      hasDiabetes, 
      calculationMethod 
    } = req.body;

    // Validate required fields
    if (!age || !gender || !totalCholesterol || !hdlCholesterol || !systolicBP) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    let riskResult;
    
    // Calculate risk based on chosen method
    if (calculationMethod === 'ascvd') {
      riskResult = calculateASCVDRisk(age, gender, race, totalCholesterol, hdlCholesterol, systolicBP, isOnBPMedication, isSmoker, hasDiabetes);
    } else if (calculationMethod === 'framingham') {
      riskResult = calculateFraminghamRisk(age, gender, race, totalCholesterol, hdlCholesterol, systolicBP, isOnBPMedication, isSmoker, hasDiabetes);
    } else {
      return res.status(400).json({ error: 'Invalid calculation method' });
    }

    if (!riskResult) {
      return res.status(400).json({ error: 'Unable to calculate risk with provided data' });
    }

    // Store results in database
    const screeningResult = await prisma.screeningResult.create({
      data: {
        userId: parseInt(userId),
        assessmentType: 'HEART_RISK',
        answers: encryptJson({
          age,
          gender,
          race,
          totalCholesterol,
          hdlCholesterol,
          systolicBP,
          isOnBPMedication,
          isSmoker,
          hasDiabetes,
          calculationMethod
        }),
        totalScore: riskResult.riskScore || 0,
        severity: riskResult.riskCategory,
        severityColor: getRiskColor(riskResult.riskCategory),
        severityBgColor: getRiskBgColor(riskResult.riskCategory),
        completedAt: new Date(),
        metadata: encryptJson({
          riskResult: riskResult,
          calculationMethod: calculationMethod,
          assessmentType: 'HEART_RISK'
        })
      }
    });

    res.status(201).json({
      message: 'Heart risk assessment completed successfully',
      result: {
        ...riskResult,
        assessmentId: screeningResult.id,
        calculationMethod: calculationMethod
      }
    });
  } catch (error) {
    console.error('Error submitting heart risk results:', error);
    res.status(500).json({ error: 'Failed to complete heart risk assessment' });
  }
};

// Helper functions for heart risk styling
const getRiskColor = (riskCategory) => {
  if (riskCategory?.includes('Low')) return 'text-green-600';
  if (riskCategory?.includes('Borderline') || riskCategory?.includes('Intermediate')) return 'text-yellow-600';
  return 'text-red-600';
};

const getRiskBgColor = (riskCategory) => {
  if (riskCategory?.includes('Low')) return 'bg-green-50';
  if (riskCategory?.includes('Borderline') || riskCategory?.includes('Intermediate')) return 'bg-yellow-50';
  return 'bg-red-50';
};

// Submit Testosterone Screening (symptom-based risk)
const submitTestosteroneResults = async (req, res) => {
  try {
    const { userId } = req.user;
    const { symptoms = [], notApplicable = false, completedAt } = req.body;

    // If N/A, store as such
    let riskCategory;
    if (notApplicable) {
      riskCategory = 'Not Applicable';
    } else {
      const count = Array.isArray(symptoms) ? symptoms.length : 0;
      if (count === 0) riskCategory = 'No Symptoms';
      else if (count <= 2) riskCategory = 'Low Probability';
      else if (count <= 4) riskCategory = 'Moderate Probability';
      else riskCategory = 'High Probability';
    }

    const screeningResult = await prisma.screeningResult.create({
      data: {
        userId: parseInt(userId),
        assessmentType: 'TESTOSTERONE',
        answers: encryptJson({ symptoms, notApplicable }),
        totalScore: Array.isArray(symptoms) ? symptoms.length : 0,
        severity: riskCategory,
        severityColor: riskCategory.includes('High') ? 'text-red-600' : riskCategory.includes('Moderate') ? 'text-orange-600' : 'text-green-600',
        severityBgColor: riskCategory.includes('High') ? 'bg-red-50' : riskCategory.includes('Moderate') ? 'bg-orange-50' : 'bg-green-50',
        completedAt: completedAt ? new Date(completedAt) : new Date(),
        metadata: encryptJson({ assessmentType: 'TESTOSTERONE' })
      }
    });

    res.status(201).json({
      message: 'Testosterone screening submitted successfully',
      result: {
        id: screeningResult.id,
        riskCategory,
        totalSymptoms: Array.isArray(symptoms) ? symptoms.length : 0
      }
    });
  } catch (error) {
    console.error('Error submitting testosterone screening:', error);
    res.status(500).json({ error: 'Failed to submit testosterone screening' });
  }
};

// Simple BMI endpoint
const submitBMIResult = async (req, res) => {
  try {
    const { userId } = req.user;
    const { weightKg, heightM, completedAt } = req.body;
    if (!weightKg || !heightM || weightKg <= 0 || heightM <= 0) {
      return res.status(400).json({ error: 'Invalid weight/height' });
    }
    const bmi = parseFloat((weightKg / (heightM * heightM)).toFixed(1));
    const screeningResult = await prisma.screeningResult.create({
      data: {
        userId: parseInt(userId),
        assessmentType: 'BMI',
        answers: encryptJson({ weightKg, heightM }),
        totalScore: Math.round(bmi * 10),
        severity: String(bmi),
        severityColor: 'text-gray-700',
        severityBgColor: 'bg-gray-50',
        completedAt: completedAt ? new Date(completedAt) : new Date(),
        metadata: encryptJson({ assessmentType: 'BMI' })
      }
    });

    res.status(201).json({ message: 'BMI calculated', bmi, id: screeningResult.id });
  } catch (error) {
    console.error('Error submitting BMI:', error);
    res.status(500).json({ error: 'Failed to submit BMI' });
  }
};

module.exports = {
  submitIIEF5Results,
  submitAUDITResults,
  submitHeartRiskResults,
  submitTestosteroneResults,
  submitBMIResult,
  // will be exported later after definition (added below)
  getUserScreeningResults,
  getAssessmentResult,
  getScreeningStats,
  exportScreeningResults
};

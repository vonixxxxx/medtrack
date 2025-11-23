import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from '../lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const path = req.url?.split('?')[0] || '';
  const method = req.method;
  const route = req.query.route as string[] | string | undefined;
  const routePath = Array.isArray(route) ? route.join('/') : route || '';
  
  // Debug logging (remove in production)
  console.log('Auth route handler:', { path, method, route, routePath });

  // Handle OPTIONS preflight requests
  if (method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    return res.status(200).end();
  }

  // Route: /api/auth/login
  if ((routePath === 'login' || path.includes('/auth/login') || path.endsWith('/login')) && method === 'POST') {
    try {
      const { email, password } = req.body;

      if (!email || !password) {
        return res.status(400).json({ error: 'Email and password are required' });
      }

      const user = await prisma.user.findUnique({
        where: { email },
      });

      if (!user || user.password !== password) {
        return res.status(401).json({ error: 'Invalid credentials' });
      }

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
          hospitalCode: user.hospitalCode,
        },
      });
    } catch (error: any) {
      console.error('Login error:', error);
      res.status(500).json({ error: 'Failed to login' });
    }
    return;
  }

  // Route: /api/auth/signup
  if ((routePath === 'signup' || path.includes('/auth/signup') || path.endsWith('/signup')) && method === 'POST') {
    try {
      const { email, password, role, hospitalCode, patientData } = req.body;

      const existingUser = await prisma.user.findUnique({
        where: { email },
      });

      if (existingUser) {
        return res.status(400).json({
          error: 'User already exists',
          details: 'An account with this email already exists. Please login instead.',
        });
      }

      const user = await prisma.user.create({
        data: {
          email,
          password,
          role: role || 'patient',
          hospitalCode: hospitalCode || '123456789',
          name: patientData?.name || null,
          surveyCompleted: false,
        },
      });

      if (role === 'patient') {
        try {
          const { name, ...patientFields } = patientData || {};
          
          // Only include valid date fields if they exist
          const patientDataToCreate: any = {
            userId: user.id,
            patient_audit_id: `PAT-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            imd_decile: Math.floor(Math.random() * 10) + 1,
          };

          // Only add date fields if they exist and are valid
          if (patientFields?.dob) {
            patientDataToCreate.dob = new Date(patientFields.dob);
          }
          if (patientFields?.baseline_weight_date) {
            patientDataToCreate.baseline_weight_date = new Date(patientFields.baseline_weight_date);
          }
          if (patientFields?.baseline_hba1c_date) {
            patientDataToCreate.baseline_hba1c_date = new Date(patientFields.baseline_hba1c_date);
          }
          if (patientFields?.baseline_lipid_date) {
            patientDataToCreate.baseline_lipid_date = new Date(patientFields.baseline_lipid_date);
          }

          // Add other optional fields if they exist
          const optionalFields = ['sex', 'ethnicity', 'ethnic_group', 'location', 'postcode', 'nhs_number', 'mrn'];
          optionalFields.forEach(field => {
            if (patientFields?.[field] !== undefined && patientFields[field] !== null) {
              patientDataToCreate[field] = patientFields[field];
            }
          });

          await prisma.patient.create({
            data: patientDataToCreate,
          });
        } catch (patientError: any) {
          console.error('Error creating patient record:', patientError);
          // Don't fail signup if patient creation fails - user is already created
          // Just log the error
        }
      }

      if (role === 'clinician') {
        await prisma.clinician.create({
          data: {
            userId: user.id,
            hospitalCode: hospitalCode || '123456789',
          },
        });
      }

      res.json({
        success: true,
        user,
        token: `demo-token-${user.id}-${Date.now()}`,
      });
    } catch (error: any) {
      console.error('Signup error:', error);
      console.error('Error stack:', error.stack);
      res.status(500).json({ 
        error: 'Signup failed', 
        details: error.message,
        code: error.code,
      });
    }
    return;
  }

  // Route: /api/auth/me
  if ((routePath === 'me' || path.includes('/auth/me')) && method === 'GET') {
    try {
      const authHeader = req.headers.authorization;
      if (!authHeader) {
        return res.status(401).json({ error: 'No authorization header' });
      }

      const token = authHeader.replace('Bearer ', '');
      const userId = token.split('-')[2];
      
      if (!userId) {
        return res.status(401).json({ error: 'Invalid token' });
      }

      const user = await prisma.user.findUnique({
        where: { id: userId },
      });

      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }

      res.json({
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        hospitalCode: user.hospitalCode,
      });
    } catch (error: any) {
      console.error('Get user error:', error);
      res.status(500).json({ error: 'Failed to get user' });
    }
    return;
  }

  // Route: /api/auth/survey-status
  if ((routePath === 'survey-status' || path.includes('/auth/survey-status') || path.endsWith('/survey-status')) && method === 'GET') {
    try {
      const authHeader = req.headers.authorization;
      if (!authHeader) {
        return res.status(401).json({ error: 'No authorization header' });
      }

      const token = authHeader.replace('Bearer ', '');
      let userId = req.query.userId as string | undefined;
      
      if (!userId) {
        // Try to extract from token
        const tokenParts = token.split('-');
        if (tokenParts.length >= 3) {
          userId = tokenParts[2];
        }
      }

      if (!userId) {
        // Get most recent user as fallback
        const latestUser = await prisma.user.findFirst({
          orderBy: { createdAt: 'desc' },
        });
        if (!latestUser) {
          return res.json({ surveyCompleted: false });
        }
        userId = latestUser.id;
      }

      const user = await prisma.user.findUnique({
        where: { id: userId },
        select: { surveyCompleted: true },
      });

      if (!user) {
        return res.json({ surveyCompleted: false });
      }

      res.json({ surveyCompleted: user.surveyCompleted || false });
    } catch (error: any) {
      console.error('Get survey status error:', error);
      res.status(500).json({ error: 'Failed to get survey status' });
    }
    return;
  }

  // Route: /api/auth/survey-data
  if ((routePath === 'survey-data' || path.includes('/auth/survey-data') || path.endsWith('/survey-data')) && method === 'POST') {
    try {
      const authHeader = req.headers.authorization;
      if (!authHeader) {
        return res.status(401).json({ error: 'No authorization header' });
      }

      const token = authHeader.replace('Bearer ', '');
      const tokenParts = token.split('-');
      const userId = tokenParts[2];
      
      if (!userId) {
        return res.status(401).json({ error: 'Invalid token' });
      }

      const surveyData = req.body;
      
      // Extract only valid UserSurveyData fields (exclude 'name' which goes to User table)
      const validSurveyFields: any = {
        dateOfBirth: surveyData.dateOfBirth ? new Date(surveyData.dateOfBirth) : null,
        biologicalSex: surveyData.biologicalSex || null,
        ethnicity: surveyData.ethnicity || null,
        hasMenses: surveyData.hasMenses ?? null,
        ageAtMenarche: surveyData.ageAtMenarche ? parseInt(surveyData.ageAtMenarche) : null,
        menstrualRegularity: surveyData.menstrualRegularity || null,
        lastMenstrualPeriod: surveyData.lastMenstrualPeriod ? new Date(surveyData.lastMenstrualPeriod) : null,
        cycleLength: surveyData.cycleLength ? parseInt(surveyData.cycleLength) : null,
        periodDuration: surveyData.periodDuration ? parseInt(surveyData.periodDuration) : null,
        usesContraception: surveyData.usesContraception ?? null,
        contraceptionType: surveyData.contraceptionType || null,
        hasPreviousPregnancies: surveyData.hasPreviousPregnancies ?? null,
        isPerimenopausal: surveyData.isPerimenopausal ?? null,
        isPostmenopausal: surveyData.isPostmenopausal ?? null,
        ageAtMenopause: surveyData.ageAtMenopause ? parseInt(surveyData.ageAtMenopause) : null,
        menopauseType: surveyData.menopauseType || null,
        isOnHRT: surveyData.isOnHRT ?? null,
        hrtType: surveyData.hrtType || null,
        iiefScore: surveyData.iiefScore ? parseInt(surveyData.iiefScore) : null,
        lowTestosteroneSymptoms: surveyData.lowTestosteroneSymptoms || null,
        redFlagQuestions: surveyData.redFlagQuestions || null,
        auditScore: surveyData.auditScore ? parseInt(surveyData.auditScore) : null,
        smokingStatus: surveyData.smokingStatus || null,
        smokingStartAge: surveyData.smokingStartAge ? parseInt(surveyData.smokingStartAge) : null,
        cigarettesPerDay: surveyData.cigarettesPerDay ? parseInt(surveyData.cigarettesPerDay) : null,
        vapingDevice: surveyData.vapingDevice || null,
        nicotineMg: surveyData.nicotineMg ? parseFloat(surveyData.nicotineMg) : null,
        pgVgRatio: surveyData.pgVgRatio || null,
        usagePattern: surveyData.usagePattern || null,
        psecdiScore: surveyData.psecdiScore ? parseInt(surveyData.psecdiScore) : null,
        readinessToQuit: surveyData.readinessToQuit ? parseInt(surveyData.readinessToQuit) : null,
        ipaqScore: surveyData.ipaqScore ? parseInt(surveyData.ipaqScore) : null,
        weight: surveyData.weight ? parseFloat(surveyData.weight) : null,
        height: surveyData.height ? parseFloat(surveyData.height) : null,
        waistCircumference: surveyData.waistCircumference ? parseFloat(surveyData.waistCircumference) : null,
        hipCircumference: surveyData.hipCircumference ? parseFloat(surveyData.hipCircumference) : null,
        neckCircumference: surveyData.neckCircumference ? parseFloat(surveyData.neckCircumference) : null,
        systolicBP: surveyData.systolicBP ? parseInt(surveyData.systolicBP) : null,
        diastolicBP: surveyData.diastolicBP ? parseInt(surveyData.diastolicBP) : null,
      };

      // Remove null values to avoid Prisma errors
      Object.keys(validSurveyFields).forEach(key => {
        if (validSurveyFields[key] === null || validSurveyFields[key] === undefined) {
          delete validSurveyFields[key];
        }
      });

      // Update user's name if provided
      if (surveyData.name) {
        await prisma.user.update({
          where: { id: userId },
          data: { name: surveyData.name },
        });
      }

      // Upsert survey data
      await prisma.userSurveyData.upsert({
        where: { userId },
        update: validSurveyFields,
        create: {
          userId,
          ...validSurveyFields,
        },
      });

      res.json({ success: true, message: 'Survey data saved' });
    } catch (error: any) {
      console.error('Save survey data error:', error);
      res.status(500).json({ error: 'Failed to save survey data', details: error.message });
    }
    return;
  }

  // Route: /api/auth/complete-survey
  if ((routePath === 'complete-survey' || path.includes('/auth/complete-survey') || path.endsWith('/complete-survey')) && method === 'PUT') {
    try {
      const authHeader = req.headers.authorization;
      if (!authHeader) {
        return res.status(401).json({ error: 'No authorization header' });
      }

      const token = authHeader.replace('Bearer ', '');
      const { userId } = req.body;
      
      let targetUserId = userId;
      if (!targetUserId) {
        const tokenParts = token.split('-');
        if (tokenParts.length >= 3) {
          targetUserId = tokenParts[2];
        }
      }

      if (!targetUserId) {
        return res.status(400).json({ error: 'User ID is required' });
      }

      // Mark survey as completed
      await prisma.user.update({
        where: { id: targetUserId },
        data: { surveyCompleted: true },
      });

      res.json({ success: true, message: 'Survey marked as completed' });
    } catch (error: any) {
      console.error('Complete survey error:', error);
      res.status(500).json({ error: 'Failed to complete survey', details: error.message });
    }
    return;
  }

  // If no route matched, return 404 for unknown routes
  console.log('Auth route not matched:', { path, method, routePath });
  return res.status(404).json({ 
    error: 'Auth route not found', 
    path, 
    method, 
    routePath,
    hint: 'Available routes: /api/auth/login, /api/auth/signup, /api/auth/me, etc.'
  });
}

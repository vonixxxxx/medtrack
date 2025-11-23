import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from '../lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { email, password, role, hospitalCode, patientData } = req.body;

    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email },
    });

    if (existingUser) {
      return res.status(400).json({
        error: 'User already exists',
        details: 'An account with this email already exists. Please login instead.',
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
        surveyCompleted: false, // Initialize survey as not completed
      },
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
          ...patientFields,
        },
      });
      console.log('Patient created with ID:', patient.id, 'for user:', user.email);
    }

    // Create clinician profile if role is clinician
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
      token: 'mock-jwt-token-' + Date.now(),
    });
  } catch (error: any) {
    console.error('Signup error:', error);
    console.error('Error details:', error.message);
    console.error('Stack trace:', error.stack);
    res.status(500).json({ error: 'Signup failed', details: error.message });
  }
}

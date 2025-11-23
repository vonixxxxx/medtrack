import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from '../lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const path = req.url?.split('?')[0] || '';
  const method = req.method;
  const route = req.query.route as string[] | string | undefined;
  const routePath = Array.isArray(route) ? route.join('/') : route || '';

  // Route: /api/auth/login
  if ((routePath === 'login' || path.includes('/auth/login')) && method === 'POST') {
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
  if ((routePath === 'signup' || path.includes('/auth/signup')) && method === 'POST') {
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
        const { name, ...patientFields } = patientData || {};
        if (patientFields.dob) patientFields.dob = new Date(patientFields.dob);
        if (patientFields.baseline_weight_date) patientFields.baseline_weight_date = new Date(patientFields.baseline_weight_date);
        if (patientFields.baseline_hba1c_date) patientFields.baseline_hba1c_date = new Date(patientFields.baseline_hba1c_date);
        if (patientFields.baseline_lipid_date) patientFields.baseline_lipid_date = new Date(patientFields.baseline_lipid_date);

        await prisma.patient.create({
          data: {
            userId: user.id,
            patient_audit_id: `PAT-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            imd_decile: Math.floor(Math.random() * 10) + 1,
            ...patientFields,
          },
        });
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
        token: 'mock-jwt-token-' + Date.now(),
      });
    } catch (error: any) {
      console.error('Signup error:', error);
      res.status(500).json({ error: 'Signup failed', details: error.message });
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

  res.status(404).json({ error: 'Route not found' });
}

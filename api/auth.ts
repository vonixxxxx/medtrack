import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from './lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const path = req.url?.split('?')[0] || '';
  const method = req.method;

  // Route: /api/auth/login
  if ((path.includes('/auth/login') || path.endsWith('/auth/login')) && method === 'POST') {
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
  if ((path.includes('/auth/signup') || path.endsWith('/auth/signup')) && method === 'POST') {
    try {
      const { email, password, name, role, hospitalCode } = req.body;

      if (!email || !password || !name) {
        return res.status(400).json({ error: 'Email, password, and name are required' });
      }

      const existingUser = await prisma.user.findUnique({
        where: { email },
      });

      if (existingUser) {
        return res.status(400).json({ error: 'User already exists' });
      }

      const user = await prisma.user.create({
        data: {
          email,
          password, // In production, hash this
          name,
          role: role || 'patient',
          hospitalCode: hospitalCode || null,
        },
      });

      // Create patient record if role is patient
      if (user.role === 'patient') {
        await prisma.patient.create({
          data: {
            userId: user.id,
            name: user.name,
            email: user.email,
          },
        });
      }

      const token = `demo-token-${user.id}-${Date.now()}`;

      res.status(201).json({
        success: true,
        message: 'Signup successful',
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
      console.error('Signup error:', error);
      res.status(500).json({ error: 'Failed to signup' });
    }
    return;
  }

  // Route: /api/auth/me
  if ((path.includes('/auth/me') || path.endsWith('/auth/me')) && method === 'GET') {
    try {
      const authHeader = req.headers.authorization;
      if (!authHeader) {
        return res.status(401).json({ error: 'No authorization header' });
      }

      const token = authHeader.replace('Bearer ', '');
      // Simplified token parsing - in production, verify JWT
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

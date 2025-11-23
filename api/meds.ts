import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from './lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const path = req.url?.split('?')[0] || '';
  const method = req.method;

  // Route: /api/meds/user
  if (path.endsWith('/meds/user')) {
    if (method === 'GET') {
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

        const medications = await prisma.medication.findMany({
          where: { userId },
          include: {
            schedules: true,
          },
        });

        res.json(medications);
      } catch (error: any) {
        console.error('Get medications error:', error);
        res.status(500).json({ error: 'Failed to get medications' });
      }
      return;
    }

    if (method === 'POST') {
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

        const { name, dosage, frequency, startDate, endDate } = req.body;

        const medication = await prisma.medication.create({
          data: {
            userId,
            name,
            dosage,
            frequency,
            startDate: startDate ? new Date(startDate) : new Date(),
            endDate: endDate ? new Date(endDate) : null,
          },
        });

        res.status(201).json(medication);
      } catch (error: any) {
        console.error('Create medication error:', error);
        res.status(500).json({ error: 'Failed to create medication' });
      }
      return;
    }
  }

  // Route: /api/meds/schedule
  if (path.endsWith('/meds/schedule') && method === 'GET') {
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

      const schedules = await prisma.medicationSchedule.findMany({
        where: {
          medication: {
            userId,
          },
        },
        include: {
          medication: true,
        },
      });

      res.json(schedules);
    } catch (error: any) {
      console.error('Get schedule error:', error);
      res.status(500).json({ error: 'Failed to get schedule' });
    }
    return;
  }

  // Route: /api/meds/cycles
  if (path.endsWith('/meds/cycles') && method === 'GET') {
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

      const medications = await prisma.medication.findMany({
        where: { userId },
        include: {
          schedules: true,
        },
      });

      // Transform to cycles format
      const cycles = medications.map(med => ({
        medicationId: med.id,
        medicationName: med.name,
        cycles: med.schedules || [],
      }));

      res.json(cycles);
    } catch (error: any) {
      console.error('Get cycles error:', error);
      res.status(500).json({ error: 'Failed to get cycles' });
    }
    return;
  }

  res.status(404).json({ error: 'Route not found' });
}

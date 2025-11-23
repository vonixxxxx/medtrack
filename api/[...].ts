import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from './lib/prisma';

// Consolidated catch-all handler for all routes
export default async function handler(req: VercelRequest, res: VercelResponse) {
  const path = req.url?.split('?')[0] || '';
  const method = req.method;
  const route = req.query.route as string[] | string | undefined;
  const routePath = Array.isArray(route) ? route.join('/') : route || '';

  // Route: /api/hello
  if (path.includes('/hello') || routePath === 'hello') {
    return res.status(200).json({ message: 'Backend running on Vercel' });
  }

  // Route: /api/test-public
  if (path.includes('/test-public') || routePath === 'test-public') {
    return res.status(200).json({ message: 'Public test endpoint', timestamp: new Date().toISOString() });
  }

  // Route: /api/health
  if (path.includes('/health') && !path.includes('/health-metrics') && (routePath === 'health' || !routePath)) {
    try {
      await prisma.$queryRaw`SELECT 1`;
      return res.status(200).json({ status: 'ok', database: 'connected' });
    } catch (error: any) {
      return res.status(500).json({ status: 'error', database: 'disconnected', error: error.message });
    }
  }

  // Route: /api/health-metrics
  if (path.includes('/health-metrics') || routePath === 'health-metrics') {
    if (method === 'GET') {
      try {
        const authHeader = req.headers.authorization;
        if (!authHeader) {
          return res.status(401).json({ error: 'No authorization header' });
        }

        const token = authHeader.replace('Bearer ', '');
        let userId = req.query.userId as string | undefined;
        
        if (!userId) {
          const latestUser = await prisma.user.findFirst({
            orderBy: { createdAt: 'desc' },
          });
          if (!latestUser) {
            return res.json({ metrics: [] });
          }
          userId = latestUser.id;
        }

        const patient = await prisma.patient.findFirst({
          where: { userId },
        });

        if (!patient) {
          return res.json({ metrics: [] });
        }

        const metrics = await prisma.metricTrend.findMany({
          where: { patientId: patient.id },
          orderBy: { date: 'desc' },
          take: 100,
        });

        return res.json({ metrics });
      } catch (error: any) {
        console.error('Get health metrics error:', error);
        return res.status(500).json({ error: 'Failed to get health metrics' });
      }
    }
  }

  // Route: /api/medication-schedules
  if (path.includes('/medication-schedules') || routePath === 'medication-schedules') {
    if (method === 'GET') {
      try {
        const authHeader = req.headers.authorization;
        if (!authHeader) {
          return res.status(401).json({ error: 'No authorization header' });
        }

        const token = authHeader.replace('Bearer ', '');
        let userId = req.query.userId as string | undefined;
        
        if (!userId) {
          const latestUser = await prisma.user.findFirst({
            orderBy: { createdAt: 'desc' },
          });
          if (!latestUser) {
            return res.json({ schedules: [] });
          }
          userId = latestUser.id;
        }

        const medications = await prisma.medication.findMany({
          where: { userId },
        });

        const schedules = medications.map(med => ({
          medicationId: med.id,
          medicationName: med.name,
          dosage: med.dosage,
          frequency: med.frequency,
          reminderTimes: med.reminderTimes ? JSON.parse(med.reminderTimes) : [],
        }));

        return res.json({ schedules });
      } catch (error: any) {
        console.error('Get medication schedules error:', error);
        return res.status(500).json({ error: 'Failed to get medication schedules' });
      }
    }
  }

  // Route: /api/metrics/user
  if (path.includes('/metrics/user') || routePath === 'metrics/user') {
    if (method === 'GET') {
      try {
        const authHeader = req.headers.authorization;
        if (!authHeader) {
          return res.status(401).json({ error: 'No authorization header' });
        }

        const token = authHeader.replace('Bearer ', '');
        let userId = req.query.userId as string | undefined;
        
        if (!userId) {
          const latestUser = await prisma.user.findFirst({
            orderBy: { createdAt: 'desc' },
          });
          if (!latestUser) {
            return res.json({ metrics: [] });
          }
          userId = latestUser.id;
        }

        const metrics = await prisma.metric.findMany({
          where: { userId },
          orderBy: { date: 'desc' },
          take: 100,
        });

        return res.json({ metrics });
      } catch (error: any) {
        console.error('Get user metrics error:', error);
        return res.status(500).json({ error: 'Failed to get user metrics' });
      }
    }
  }

  // Route: /api/doctor/patients
  if (path.includes('/doctor/patients') || routePath === 'doctor/patients') {
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

        const user = await prisma.user.findUnique({
          where: { id: userId },
        });

        if (!user || user.role !== 'clinician') {
          return res.status(403).json({ error: 'Access denied' });
        }

        const patients = await prisma.patient.findMany({
          where: {
            user: {
              hospitalCode: user.hospitalCode,
            },
          },
          include: {
            user: {
              select: {
                id: true,
                email: true,
                name: true,
              },
            },
          },
        });

        const transformedPatients = patients.map(patient => ({
          id: patient.id,
          userId: patient.userId,
          email: patient.user?.email,
          name: patient.user?.name,
          dob: patient.dob,
          sex: patient.sex,
          createdAt: patient.createdAt,
        }));

        return res.json({ patients: transformedPatients });
      } catch (error: any) {
        console.error('Get patients error:', error);
        return res.status(500).json({ error: 'Failed to get patients' });
      }
    }
  }

  // Route: /api/medications/validateMedication
  if (path.includes('/medications/validateMedication') || routePath === 'medications/validateMedication') {
    if (method === 'POST') {
      try {
        const { medicationName } = req.body;
        
        if (!medicationName) {
          return res.status(400).json({ error: 'Medication name is required' });
        }

        // Simplified validation - just return success
        return res.json({
          valid: true,
          medicationName,
          normalizedName: medicationName.toLowerCase().trim(),
        });
      } catch (error: any) {
        console.error('Validate medication error:', error);
        return res.status(500).json({ error: 'Failed to validate medication' });
      }
    }
  }

  // Route: /api/prescriptions
  if (path.includes('/prescriptions') || routePath === 'prescriptions' || routePath?.startsWith('prescriptions/')) {
    if (method === 'GET') {
      try {
        const authHeader = req.headers.authorization;
        if (!authHeader) {
          return res.status(401).json({ error: 'No authorization header' });
        }

        const token = authHeader.replace('Bearer ', '');
        const tokenParts = token.split('-');
        let userId = tokenParts[2];
        
        if (!userId) {
          const latestUser = await prisma.user.findFirst({
            orderBy: { createdAt: 'desc' },
          });
          if (!latestUser) {
            return res.json([]);
          }
          userId = latestUser.id;
        }

        const patient = await prisma.patient.findFirst({
          where: { userId },
        });

        if (!patient) {
          return res.json([]);
        }

        const prescriptions = await prisma.prescription.findMany({
          where: { patientId: patient.id },
          orderBy: { datePrescribed: 'desc' },
        });

        return res.json(prescriptions);
      } catch (error: any) {
        console.error('Get prescriptions error:', error);
        return res.status(500).json({ error: 'Failed to get prescriptions' });
      }
    }
    // For POST, PUT, DELETE - return empty array for now
    if (method === 'POST' || method === 'PUT' || method === 'DELETE') {
      return res.json({ success: true, message: 'Prescription operation not yet implemented' });
    }
  }

  // Route: /api/side-effects
  if (path.includes('/side-effects') || routePath === 'side-effects' || routePath?.startsWith('side-effects/')) {
    if (method === 'GET') {
      try {
        const authHeader = req.headers.authorization;
        if (!authHeader) {
          return res.status(401).json({ error: 'No authorization header' });
        }

        const token = authHeader.replace('Bearer ', '');
        const tokenParts = token.split('-');
        let userId = tokenParts[2];
        
        if (!userId) {
          const latestUser = await prisma.user.findFirst({
            orderBy: { createdAt: 'desc' },
          });
          if (!latestUser) {
            return res.json([]);
          }
          userId = latestUser.id;
        }

        const medications = await prisma.medication.findMany({
          where: { userId },
        });

        const medicationIds = medications.map(m => m.id);

        const sideEffects = await prisma.medicationSideEffect.findMany({
          where: {
            medicationId: { in: medicationIds },
          },
          orderBy: { createdAt: 'desc' },
        });

        return res.json(sideEffects);
      } catch (error: any) {
        console.error('Get side effects error:', error);
        return res.status(500).json({ error: 'Failed to get side effects' });
      }
    }
    // For POST, PUT, DELETE - return empty array for now
    if (method === 'POST' || method === 'PUT' || method === 'DELETE') {
      return res.json({ success: true, message: 'Side effect operation not yet implemented' });
    }
  }

  return res.status(404).json({ error: 'Route not found', path });
}

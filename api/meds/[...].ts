import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from '../lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const path = req.url?.split('?')[0] || '';
  const method = req.method;
  const route = req.query.route as string[] | string | undefined;
  const routePath = Array.isArray(route) ? route.join('/') : route || '';

  // Route: /api/meds/user
  if (routePath === 'user' || path.includes('/meds/user')) {
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
            return res.json({ medications: [] });
          }
          userId = latestUser.id;
        }

        const patient = await prisma.patient.findFirst({
          where: { userId },
        });

        if (!patient) {
          return res.json({ medications: [] });
        }

        const allMeds = await prisma.patientMedication.findMany({
          where: { patientId: patient.id },
          orderBy: { start_date: 'desc' },
        });

        const medications = allMeds.filter((m) => m.status === 'active');

        const transformedMedications = medications.map((med) => {
          const dosageMatch = med.dosage?.match(/^(\d+(?:\.\d+)?)(\w+)$/);
          const strength = dosageMatch ? dosageMatch[1] : null;
          const unit = dosageMatch ? dosageMatch[2] : null;

          return {
            id: med.id,
            medication_name: med.name,
            name: med.name,
            generic_name: med.name,
            strength: strength,
            unit: unit,
            dosage: med.dosage,
            frequency: med.frequency,
            frequency_display:
              med.frequency === 'daily' ? 'Once daily' :
              med.frequency === 'twice_daily' ? 'Twice daily' :
              med.frequency === 'three_times_daily' ? 'Three times daily' :
              med.frequency === 'four_times_daily' ? 'Four times daily' :
              med.frequency === 'weekly' ? 'Once weekly' :
              med.frequency === 'monthly' ? 'Once monthly' :
              med.frequency === 'as_needed' ? 'As needed' : med.frequency,
            start_date: med.start_date.toISOString().split('T')[0],
            end_date: med.end_date ? med.end_date.toISOString().split('T')[0] : null,
            status: med.status,
            route: med.route,
            createdAt: med.created_at.toISOString(),
          };
        });

        res.json({ medications: transformedMedications });
      } catch (error: any) {
        console.error('Error fetching medications:', error);
        res.status(500).json({ error: 'Failed to fetch medications', details: error.message });
      }
      return;
    }

    if (method === 'POST') {
      try {
        const medicationData = req.body;
        let userId = medicationData.userId;
        
        if (!userId) {
          const latestUser = await prisma.user.findFirst({
            orderBy: { createdAt: 'desc' },
          });
          if (!latestUser) {
            return res.status(400).json({ success: false, error: 'User not found' });
          }
          userId = latestUser.id;
        }

        const patient = await prisma.patient.findFirst({
          where: { userId },
        });

        if (!patient) {
          return res.status(400).json({ success: false, error: 'Patient profile not found for user' });
        }

        const strength = medicationData.strength || '';
        const unit = medicationData.unit || '';
        const dosage = strength && unit ? `${strength}${unit}` : medicationData.dosage || 'As directed';
        const frequency = medicationData.frequency || 'daily';

        const savedMedication = await prisma.patientMedication.create({
          data: {
            patientId: patient.id,
            name: medicationData.medication_name || medicationData.generic_name || 'Unknown Medication',
            dosage: dosage,
            frequency: frequency,
            route: 'oral',
            start_date: medicationData.start_date ? new Date(medicationData.start_date) : new Date(),
            status: 'active',
            manually_entered: true,
          },
        });

        res.status(201).json({
          success: true,
          message: 'Medication saved successfully',
          medication: {
            id: savedMedication.id,
            medication_name: savedMedication.name,
            generic_name: savedMedication.name,
            strength: strength,
            unit: unit,
            dosage: savedMedication.dosage,
            frequency: savedMedication.frequency,
            frequency_display: medicationData.frequency_display || frequency,
            drug_class: medicationData.drug_class || null,
            start_date: savedMedication.start_date.toISOString().split('T')[0],
            status: savedMedication.status,
            createdAt: savedMedication.created_at.toISOString(),
          },
        });
      } catch (error: any) {
        console.error('Error saving medication:', error);
        res.status(500).json({ success: false, error: 'Failed to save medication', details: error.message });
      }
      return;
    }
  }

  // Route: /api/meds/schedule
  if ((routePath === 'schedule' || path.includes('/meds/schedule')) && method === 'GET') {
    res.json({ schedule: [] });
    return;
  }

  // Route: /api/meds/cycles
  if ((routePath === 'cycles' || path.includes('/meds/cycles')) && method === 'GET') {
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
          return res.json({ cycles: [] });
        }
        userId = latestUser.id;
      }

      const medications = await prisma.medication.findMany({
        where: { userId },
      });

      const cycles = medications.map(med => ({
        medicationId: med.id,
        medicationName: med.name,
        cycles: [],
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

import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from '../lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method === 'GET') {
    try {
      // Get user ID from query params or find the most recent user (for demo)
      let userId = req.query.userId as string | undefined;
      console.log('GET /api/meds/user - userId from query:', userId);

      if (!userId) {
        // For demo: get the most recent user
        const latestUser = await prisma.user.findFirst({
          orderBy: { createdAt: 'desc' },
        });
        if (!latestUser) {
          return res.json({ medications: [] });
        }
        userId = latestUser.id;
      }

      // Find patient associated with user
      const patient = await prisma.patient.findFirst({
        where: { userId },
      });

      console.log('Patient found:', patient ? patient.id : 'NOT FOUND');

      if (!patient) {
        console.log('No patient found for user:', userId);
        return res.json({ medications: [] });
      }

      // Get all medications for this patient (including inactive for debugging)
      const allMeds = await prisma.patientMedication.findMany({
        where: {
          patientId: patient.id,
        },
        orderBy: {
          start_date: 'desc',
        },
      });

      console.log('Total medications in DB for patient:', allMeds.length);
      allMeds.forEach((m) => console.log('  -', m.name, m.status, m.id));

      // Filter to active medications
      const medications = allMeds.filter((m) => m.status === 'active');
      console.log('Active medications:', medications.length);

      // Transform to match frontend expected format
      const transformedMedications = medications.map((med) => {
        // Parse dosage to extract strength and unit if possible
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
            med.frequency === 'daily'
              ? 'Once daily'
              : med.frequency === 'twice_daily'
                ? 'Twice daily'
                : med.frequency === 'three_times_daily'
                  ? 'Three times daily'
                  : med.frequency === 'four_times_daily'
                    ? 'Four times daily'
                    : med.frequency === 'weekly'
                      ? 'Once weekly'
                      : med.frequency === 'monthly'
                        ? 'Once monthly'
                        : med.frequency === 'as_needed'
                          ? 'As needed'
                          : med.frequency,
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
  } else if (req.method === 'POST') {
    try {
      const medicationData = req.body;
      console.log('Saving medication:', medicationData);

      // Get user ID from request body or find the most recent user (for demo)
      let userId = medicationData.userId;
      if (!userId) {
        // For demo: get the most recent user
        const latestUser = await prisma.user.findFirst({
          orderBy: { createdAt: 'desc' },
        });
        if (!latestUser) {
          return res.status(400).json({ success: false, error: 'User not found' });
        }
        userId = latestUser.id;
      }

      // Find patient associated with user
      const patient = await prisma.patient.findFirst({
        where: { userId },
      });

      if (!patient) {
        return res.status(400).json({ success: false, error: 'Patient profile not found for user' });
      }

      // Parse strength and unit
      const strength = medicationData.strength || '';
      const unit = medicationData.unit || '';
      const dosage = strength && unit ? `${strength}${unit}` : medicationData.dosage || 'As directed';

      // Parse frequency
      const frequency = medicationData.frequency || 'daily';

      // Create medication in database
      console.log('Creating medication with data:', {
        patientId: patient.id,
        name: medicationData.medication_name || medicationData.generic_name || 'Unknown Medication',
        dosage: dosage,
        frequency: frequency,
        start_date: medicationData.start_date ? new Date(medicationData.start_date) : new Date(),
      });

      const savedMedication = await prisma.patientMedication.create({
        data: {
          patientId: patient.id,
          name: medicationData.medication_name || medicationData.generic_name || 'Unknown Medication',
          dosage: dosage,
          frequency: frequency,
          route: 'oral', // Default route
          start_date: medicationData.start_date ? new Date(medicationData.start_date) : new Date(),
          status: 'active',
          manually_entered: true,
        },
      });

      console.log('✅ Medication saved successfully:', savedMedication.id, savedMedication.name);

      // Verify it was saved
      const verifyMed = await prisma.patientMedication.findUnique({
        where: { id: savedMedication.id },
      });
      console.log('✅ Verified medication in database:', verifyMed ? 'Found' : 'NOT FOUND');

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
      console.error('Error stack:', error.stack);
      res.status(500).json({ success: false, error: 'Failed to save medication', details: error.message });
    }
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}

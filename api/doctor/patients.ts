import { VercelRequest, VercelResponse } from '@vercel/node';
import { prisma } from '../lib/prisma';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // For demo purposes, get the most recent clinician's hospital code
    // In a real app, you'd get this from the JWT token
    const clinician = await prisma.clinician.findFirst({
      orderBy: { createdAt: 'desc' },
    });

    if (!clinician) {
      return res.json([]);
    }

    const clinicianHospitalCode = clinician.hospitalCode;
    console.log('Clinician hospital code:', clinicianHospitalCode);

    const patients = await prisma.patient.findMany({
      where: {
        user: {
          hospitalCode: clinicianHospitalCode,
          role: 'patient',
        },
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
            hospitalCode: true,
          },
        },
        conditions: true,
      },
    });

    console.log(`Found ${patients.length} patients for hospital code ${clinicianHospitalCode}`);

    // Debug: Log each patient's hospital code
    patients.forEach((patient) => {
      console.log(`Patient ${patient.user.email} has hospital code: ${patient.user.hospitalCode}`);
    });

    const transformedPatients = patients.map((patient) => {
      const age = patient.dob
        ? Math.floor((new Date().getTime() - new Date(patient.dob).getTime()) / (365.25 * 24 * 60 * 60 * 1000))
        : null;

      return {
        id: patient.id,
        userId: patient.userId,
        name: patient.user.name || `Patient ${patient.id}`,
        email: patient.user.email,
        hospitalCode: patient.user.hospitalCode,
        age,
        sex: patient.sex || null,
        ethnic_group: patient.ethnic_group || null,
        ethnicity: patient.ethnicity || null,
        location: patient.location || null,
        postcode: patient.postcode || null,
        nhs_number: patient.nhs_number || null,
        mrn: patient.mrn || null,
        height: patient.height || null,
        baseline_weight: patient.baseline_weight || null,
        baseline_bmi: patient.baseline_bmi || null,
        baseline_weight_date: patient.baseline_weight_date?.toISOString().split('T')[0] || null,
        ascvd: patient.ascvd || false,
        htn: patient.htn || false,
        dyslipidaemia: patient.dyslipidaemia || false,
        osa: patient.osa || false,
        sleep_studies: patient.sleep_studies || false,
        cpap: patient.cpap || false,
        t2dm: patient.t2dm || false,
        prediabetes: patient.prediabetes || false,
        diabetes_type: patient.diabetes_type || null,
        baseline_hba1c: patient.baseline_hba1c || null,
        baseline_hba1c_date: patient.baseline_hba1c_date?.toISOString().split('T')[0] || null,
        hba1cPercent: patient.hba1c_percent || null,
        hba1cMmolMol: patient.hba1c_mmol || null,
        baseline_tc: patient.baseline_tc || null,
        baseline_hdl: patient.baseline_hdl || null,
        baseline_ldl: patient.baseline_ldl || null,
        baseline_tg: patient.baseline_tg || null,
        baseline_lipid_date: patient.baseline_lipid_date?.toISOString().split('T')[0] || null,
        lipid_lowering_treatment: patient.lipid_lowering_treatment || null,
        antihypertensive_medications: patient.antihypertensive_medications || null,
        asthma: patient.asthma || false,
        hypertension: patient.hypertension || false,
        ischaemic_heart_disease: patient.ischaemic_heart_disease || false,
        heart_failure: patient.heart_failure || false,
        cerebrovascular_disease: patient.cerebrovascular_disease || false,
        pulmonary_hypertension: patient.pulmonary_hypertension || false,
        dvt: patient.dvt || false,
        pe: patient.pe || false,
        gord: patient.gord || false,
        ckd: patient.ckd || false,
        kidney_stones: patient.kidney_stones || false,
        masld: patient.masld || false,
        infertility: patient.infertility || false,
        pcos: patient.pcos || false,
        anxiety: patient.anxiety || false,
        depression: patient.depression || false,
        bipolar_disorder: patient.bipolar_disorder || false,
        emotional_eating: patient.emotional_eating || false,
        schizoaffective_disorder: patient.schizoaffective_disorder || false,
        oa_knee: patient.oa_knee || false,
        oa_hip: patient.oa_hip || false,
        limited_mobility: patient.limited_mobility || false,
        lymphoedema: patient.lymphoedema || false,
        thyroid_disorder: patient.thyroid_disorder || false,
        iih: patient.iih || false,
        epilepsy: patient.epilepsy || false,
        functional_neurological_disorder: patient.functional_neurological_disorder || false,
        cancer: patient.cancer || false,
        bariatric_gastric_band: patient.bariatric_gastric_band || false,
        bariatric_sleeve: patient.bariatric_sleeve || false,
        bariatric_bypass: patient.bariatric_bypass || false,
        bariatric_balloon: patient.bariatric_balloon || false,
        diagnoses_coded_in_scr: patient.diagnoses_coded_in_scr || null,
        total_qualifying_comorbidities: patient.total_qualifying_comorbidities || null,
        mes: patient.mes || null,
        notes: patient.notes || null,
        criteria_for_wegovy: patient.criteria_for_wegovy || null,
        conditions: patient.conditions.map((c) => c.normalized),
        lastVisit: null,
        changePercent: null,
      };
    });

    res.json(transformedPatients);
  } catch (error: any) {
    console.error('Error fetching patients:', error);
    res.status(500).json({ error: 'Failed to fetch patients' });
  }
}

/**
 * Comprehensive tests for medical history parsing
 * Tests negation handling, borderline conditions, 0/1/null extraction
 */

const { runOllamaParser, basicTextParser } = require('../utils/ollamaParser');
const { mapConditionsToPatientFields } = require('../src/utils/conditionMapper');

describe('Medical History Parser', () => {
  describe('Negation Handling', () => {
    test('should set t2dm = 0 when explicitly negated', async () => {
      const notes = 'Patient denies T2DM. T2DM: No.';
      const result = await basicTextParser(notes);
      expect(result.t2dm).toBe(0);
    });

    test('should set ckd = 0 when explicitly negated', async () => {
      const notes = 'CKD: No. Chronic kidney disease absent.';
      const result = await basicTextParser(notes);
      expect(result.ckd).toBe(0);
    });

    test('should set hypertension = 0 when explicitly negated', async () => {
      const notes = 'HTN: No. Denies hypertension.';
      const result = await basicTextParser(notes);
      expect(result.hypertension).toBe(0);
      expect(result.htn).toBe(0);
    });

    test('should set ascvd = 0 when explicitly negated', async () => {
      const notes = 'ASCVD: No. No atherosclerotic cardiovascular disease.';
      const result = await basicTextParser(notes);
      expect(result.ascvd).toBe(0);
    });
  });

  describe('Borderline Conditions', () => {
    test('should set hypertension = 0 for borderline hypertension', async () => {
      const notes = 'Hypertension: Borderline.';
      const result = await basicTextParser(notes);
      expect(result.hypertension).toBe(0);
      expect(result.htn).toBe(0);
    });

    test('should set t2dm = 0 and prediabetes = 1 for borderline diabetes', async () => {
      const notes = 'Borderline diabetes. Prediabetes present.';
      const result = await basicTextParser(notes);
      expect(result.t2dm).toBe(0);
      expect(result.prediabetes).toBe(1);
    });
  });

  describe('Possible/Risk Conditions', () => {
    test('should set masld = 0 for "possible MASLD"', async () => {
      const notes = 'Possible MASLD. MASLD: Possible.';
      const result = await basicTextParser(notes);
      expect(result.masld).toBe(0);
    });

    test('should set masld = 1 for confirmed MASLD', async () => {
      const notes = 'MASLD confirmed.';
      const result = await basicTextParser(notes);
      expect(result.masld).toBe(1);
    });

    test('should set masld = 1 when mentioned without "possible"', async () => {
      const notes = 'Patient has MASLD.';
      const result = await basicTextParser(notes);
      expect(result.masld).toBe(1);
    });
  });

  describe('Explicit Diagnoses', () => {
    test('should set osa = 1 for "OSA: Mild"', async () => {
      const notes = 'OSA: Mild. Obstructive sleep apnea present.';
      const result = await basicTextParser(notes);
      expect(result.osa).toBe(1);
    });

    test('should set prediabetes = 1 when explicitly stated', async () => {
      const notes = 'Prediabetes: Yes.';
      const result = await basicTextParser(notes);
      expect(result.prediabetes).toBe(1);
    });

    test('should set t2dm = 1 when explicitly stated', async () => {
      const notes = 'T2DM: Yes. Type 2 diabetes mellitus.';
      const result = await basicTextParser(notes);
      expect(result.t2dm).toBe(1);
    });
  });

  describe('Not Mentioned Conditions', () => {
    test('should set masld = null when not mentioned', async () => {
      const notes = 'Patient is healthy. No liver issues.';
      const result = await basicTextParser(notes);
      expect(result.masld).toBeNull();
    });

    test('should set anxiety = null when not mentioned', async () => {
      const notes = 'No mental health concerns.';
      const result = await basicTextParser(notes);
      expect(result.anxiety).toBeNull();
    });
  });

  describe('Multiple Conditions', () => {
    test('should extract multiple conditions correctly', async () => {
      const notes = `
        T2DM: Yes
        CKD: No
        OSA: Mild
        Prediabetes: Yes
        Hypertension: Borderline
      `;
      const result = await basicTextParser(notes);
      expect(result.t2dm).toBe(1);
      expect(result.ckd).toBe(0);
      expect(result.osa).toBe(1);
      expect(result.prediabetes).toBe(1);
      expect(result.hypertension).toBe(0);
    });
  });

  describe('Condition Mapping', () => {
    test('should map conditions array to Patient boolean fields', () => {
      const conditions = ['Prediabetes', 'Dyslipidaemia', 'Obstructive Sleep Apnea'];
      const fieldMap = mapConditionsToPatientFields(conditions);
      expect(fieldMap.prediabetes).toBe(1);
      expect(fieldMap.dyslipidaemia).toBe(1);
      expect(fieldMap.osa).toBe(1);
    });

    test('should handle condition name variations', () => {
      const conditions = ['Type 2 Diabetes', 'HTN', 'CKD'];
      const fieldMap = mapConditionsToPatientFields(conditions);
      expect(fieldMap.t2dm).toBe(1);
      expect(fieldMap.htn).toBe(1);
      expect(fieldMap.ckd).toBe(1);
    });

    test('should return empty object for unmappable conditions', () => {
      const conditions = ['Unknown Condition', 'Random Text'];
      const fieldMap = mapConditionsToPatientFields(conditions);
      expect(Object.keys(fieldMap).length).toBe(0);
    });
  });

  describe('Lab Value Extraction', () => {
    test('should extract HbA1c correctly', async () => {
      const notes = 'HbA1c: 7.2%';
      const result = await basicTextParser(notes);
      expect(result.hba1c_percent).toBe(7.2);
      expect(result.baseline_hba1c).toBe(7.2);
    });

    test('should extract blood pressure correctly', async () => {
      const notes = 'BP: 140/90';
      const result = await basicTextParser(notes);
      expect(result.systolic_bp).toBe(140);
      expect(result.diastolic_bp).toBe(90);
    });

    test('should extract lipid values correctly', async () => {
      const notes = 'Total cholesterol: 200, LDL: 120, HDL: 50, Triglycerides: 150';
      const result = await basicTextParser(notes);
      expect(result.baseline_tc).toBe(200);
      expect(result.baseline_ldl).toBe(120);
      expect(result.baseline_hdl).toBe(50);
      expect(result.baseline_tg).toBe(150);
    });
  });

  describe('Comorbidity Counting', () => {
    test('should count only conditions = 1', async () => {
      const notes = `
        T2DM: Yes
        Prediabetes: Yes
        CKD: No
        OSA: Mild
        Hypertension: Borderline
      `;
      const result = await basicTextParser(notes);
      // Should count: t2dm=1, prediabetes=1, osa=1 (3 total)
      // Should NOT count: ckd=0, hypertension=0
      expect(result.total_qualifying_comorbidities).toBeGreaterThanOrEqual(3);
    });
  });

  describe('Conditions Array', () => {
    test('should include only positive diagnoses in conditions array', async () => {
      const notes = `
        T2DM: Yes
        CKD: No
        OSA: Mild
        Prediabetes: Yes
      `;
      const result = await basicTextParser(notes);
      expect(result.conditions).toContain('Type 2 Diabetes Mellitus');
      expect(result.conditions).toContain('Obstructive Sleep Apnea');
      expect(result.conditions).toContain('Prediabetes');
      expect(result.conditions).not.toContain('Chronic Kidney Disease');
    });
  });
});



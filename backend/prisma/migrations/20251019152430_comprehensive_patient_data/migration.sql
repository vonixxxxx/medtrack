-- Add comprehensive patient data fields to patients table
ALTER TABLE "patients" ADD COLUMN "patient_audit_id" TEXT;
ALTER TABLE "patients" ADD COLUMN "imd_decile" INTEGER;
ALTER TABLE "patients" ADD COLUMN "location" TEXT;
ALTER TABLE "patients" ADD COLUMN "ethnic_group" TEXT;
ALTER TABLE "patients" ADD COLUMN "postcode" TEXT;
ALTER TABLE "patients" ADD COLUMN "nhs_number" TEXT;
ALTER TABLE "patients" ADD COLUMN "start_date" DATETIME;
ALTER TABLE "patients" ADD COLUMN "height" REAL;
ALTER TABLE "patients" ADD COLUMN "baseline_weight" REAL;
ALTER TABLE "patients" ADD COLUMN "baseline_bmi" REAL;
ALTER TABLE "patients" ADD COLUMN "baseline_weight_date" DATETIME;
ALTER TABLE "patients" ADD COLUMN "ascvd" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "htn" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "dyslipidaemia" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "osa" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "sleep_studies" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "cpap" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "t2dm" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "prediabetes" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "lipid_lowering_treatment" TEXT;
ALTER TABLE "patients" ADD COLUMN "antihypertensive_medications" TEXT;
ALTER TABLE "patients" ADD COLUMN "baseline_tc" REAL;
ALTER TABLE "patients" ADD COLUMN "baseline_hdl" REAL;
ALTER TABLE "patients" ADD COLUMN "baseline_tg" REAL;
ALTER TABLE "patients" ADD COLUMN "baseline_lipid_date" DATETIME;
ALTER TABLE "patients" ADD COLUMN "baseline_hba1c_date" DATETIME;
ALTER TABLE "patients" ADD COLUMN "baseline_hba1c" REAL;
ALTER TABLE "patients" ADD COLUMN "baseline_fasting_glucose" REAL;
ALTER TABLE "patients" ADD COLUMN "random_glucose" REAL;
ALTER TABLE "patients" ADD COLUMN "notes" TEXT;
ALTER TABLE "patients" ADD COLUMN "criteria_for_wegovy" TEXT;

-- Medical conditions (boolean fields)
ALTER TABLE "patients" ADD COLUMN "asthma" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "hypertension" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "ischaemic_heart_disease" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "heart_failure" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "cerebrovascular_disease" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "pulmonary_hypertension" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "dvt" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "pe" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "gord" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "ckd" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "kidney_stones" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "masld" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "infertility" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "pcos" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "anxiety" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "depression" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "oa_knee" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "oa_hip" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "limited_mobility" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "lymphoedema" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "thyroid_disorder" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "bipolar_disorder" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "emotional_eating" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "schizoaffective_disorder" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "iih" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "epilepsy" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "functional_neurological_disorder" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "cancer" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "bariatric_gastric_band" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "bariatric_sleeve" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "bariatric_bypass" BOOLEAN;
ALTER TABLE "patients" ADD COLUMN "bariatric_balloon" BOOLEAN;

-- Clinician-specific fields
ALTER TABLE "patients" ADD COLUMN "mrn" TEXT;
ALTER TABLE "patients" ADD COLUMN "diagnoses_coded_in_scr" TEXT;
ALTER TABLE "patients" ADD COLUMN "total_qualifying_comorbidities" INTEGER;
ALTER TABLE "patients" ADD COLUMN "all_medications_from_scr" TEXT;
ALTER TABLE "patients" ADD COLUMN "baseline_ldl" REAL;

-- Create AI audit log table
CREATE TABLE "ai_audit_log" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patient_id" TEXT NOT NULL,
    "field_name" TEXT NOT NULL,
    "old_value" TEXT,
    "new_value" TEXT,
    "ai_confidence" REAL,
    "ai_suggestion" TEXT,
    "clinician_approved" BOOLEAN DEFAULT false,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "ai_audit_log_patient_id_fkey" FOREIGN KEY ("patient_id") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- Create indexes for better performance
CREATE INDEX "patients_nhs_number_idx" ON "patients"("nhs_number");
CREATE INDEX "patients_mrn_idx" ON "patients"("mrn");
CREATE INDEX "patients_postcode_idx" ON "patients"("postcode");
CREATE INDEX "patients_ethnic_group_idx" ON "patients"("ethnic_group");
CREATE INDEX "ai_audit_log_patient_id_idx" ON "ai_audit_log"("patient_id");
CREATE INDEX "ai_audit_log_created_at_idx" ON "ai_audit_log"("created_at");

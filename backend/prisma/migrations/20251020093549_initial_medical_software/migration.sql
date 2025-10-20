/*
  Warnings:

  - The primary key for the `DoseLog` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `Medication` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `MedicationCycle` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `MedicationLog` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `Metric` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `MetricLog` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `Notification` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to drop the column `created_at` on the `ai_audit_log` table. All the data in the column will be lost.
  - You are about to drop the column `patient_id` on the `ai_audit_log` table. All the data in the column will be lost.
  - You are about to drop the column `updated_at` on the `ai_audit_log` table. All the data in the column will be lost.
  - The primary key for the `user_survey_data` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `users` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - Added the required column `patientId` to the `ai_audit_log` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updatedAt` to the `ai_audit_log` table without a default value. This is not possible if the table is not empty.
  - Made the column `updatedAt` on table `users` required. This step will fail if there are existing NULL values in that column.

*/
-- CreateTable
CREATE TABLE "medical_notes" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "raw_text" TEXT NOT NULL,
    "date_of_entry" DATETIME NOT NULL,
    "source" TEXT,
    "patient_name" TEXT,
    "sex" TEXT,
    "age" INTEGER,
    "conditions" TEXT,
    "medications" TEXT,
    "allergies" TEXT,
    "lab_results" TEXT,
    "vital_signs" TEXT,
    "impression" TEXT,
    "plan" TEXT,
    "ai_confidence" REAL,
    "ai_model_used" TEXT,
    "processing_status" TEXT NOT NULL DEFAULT 'pending',
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "medical_notes_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "patient_metrics" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "noteId" TEXT,
    "metric_name" TEXT NOT NULL,
    "value" REAL NOT NULL,
    "unit" TEXT,
    "date_recorded" DATETIME NOT NULL,
    "source" TEXT,
    "is_baseline" BOOLEAN NOT NULL DEFAULT false,
    "is_ai_generated" BOOLEAN NOT NULL DEFAULT false,
    "confidence_score" REAL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "patient_metrics_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "patient_metrics_noteId_fkey" FOREIGN KEY ("noteId") REFERENCES "medical_notes" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "patient_medications" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "dosage" TEXT,
    "frequency" TEXT,
    "route" TEXT,
    "start_date" DATETIME,
    "end_date" DATETIME,
    "status" TEXT NOT NULL DEFAULT 'active',
    "is_ai_generated" BOOLEAN NOT NULL DEFAULT false,
    "source_note_id" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "patient_medications_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_DoseLog" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "cycleId" TEXT NOT NULL,
    "date" DATETIME NOT NULL,
    "taken" BOOLEAN NOT NULL DEFAULT false,
    CONSTRAINT "DoseLog_cycleId_fkey" FOREIGN KEY ("cycleId") REFERENCES "MedicationCycle" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_DoseLog" ("cycleId", "date", "id", "taken") SELECT "cycleId", "date", "id", "taken" FROM "DoseLog";
DROP TABLE "DoseLog";
ALTER TABLE "new_DoseLog" RENAME TO "DoseLog";
CREATE UNIQUE INDEX "DoseLog_cycleId_date_key" ON "DoseLog"("cycleId", "date");
CREATE TABLE "new_Medication" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "startDate" DATETIME NOT NULL,
    "endDate" DATETIME NOT NULL,
    "dosage" TEXT NOT NULL,
    "frequency" TEXT NOT NULL,
    CONSTRAINT "Medication_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_Medication" ("dosage", "endDate", "frequency", "id", "name", "startDate", "userId") SELECT "dosage", "endDate", "frequency", "id", "name", "startDate", "userId" FROM "Medication";
DROP TABLE "Medication";
ALTER TABLE "new_Medication" RENAME TO "Medication";
CREATE TABLE "new_MedicationCycle" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "dosage" TEXT NOT NULL,
    "startDate" DATETIME NOT NULL,
    "endDate" DATETIME,
    "frequencyDays" INTEGER NOT NULL,
    "dosesPerDay" INTEGER NOT NULL DEFAULT 1,
    "metricsToMonitor" TEXT,
    CONSTRAINT "MedicationCycle_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_MedicationCycle" ("dosage", "dosesPerDay", "endDate", "frequencyDays", "id", "metricsToMonitor", "name", "startDate", "userId") SELECT "dosage", "dosesPerDay", "endDate", "frequencyDays", "id", "metricsToMonitor", "name", "startDate", "userId" FROM "MedicationCycle";
DROP TABLE "MedicationCycle";
ALTER TABLE "new_MedicationCycle" RENAME TO "MedicationCycle";
CREATE TABLE "new_MedicationLog" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "medicationId" TEXT NOT NULL,
    "date" DATETIME NOT NULL,
    "dosage" TEXT NOT NULL,
    CONSTRAINT "MedicationLog_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "Medication" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_MedicationLog" ("date", "dosage", "id", "medicationId") SELECT "date", "dosage", "id", "medicationId" FROM "MedicationLog";
DROP TABLE "MedicationLog";
ALTER TABLE "new_MedicationLog" RENAME TO "MedicationLog";
CREATE TABLE "new_Metric" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "date" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "weight" REAL NOT NULL,
    "height" REAL NOT NULL,
    "bmi" REAL NOT NULL,
    "bloodPressure" TEXT NOT NULL,
    "hipCircumference" REAL,
    CONSTRAINT "Metric_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_Metric" ("bloodPressure", "bmi", "date", "height", "hipCircumference", "id", "userId", "weight") SELECT "bloodPressure", "bmi", "date", "height", "hipCircumference", "id", "userId", "weight" FROM "Metric";
DROP TABLE "Metric";
ALTER TABLE "new_Metric" RENAME TO "Metric";
CREATE TABLE "new_MetricLog" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "cycleId" TEXT NOT NULL,
    "date" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "kind" TEXT NOT NULL,
    "valueFloat" REAL,
    "valueText" TEXT,
    "notes" TEXT,
    CONSTRAINT "MetricLog_cycleId_fkey" FOREIGN KEY ("cycleId") REFERENCES "MedicationCycle" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_MetricLog" ("cycleId", "date", "id", "kind", "notes", "valueFloat", "valueText") SELECT "cycleId", "date", "id", "kind", "notes", "valueFloat", "valueText" FROM "MetricLog";
DROP TABLE "MetricLog";
ALTER TABLE "new_MetricLog" RENAME TO "MetricLog";
CREATE TABLE "new_Notification" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "cycleId" TEXT,
    "date" DATETIME NOT NULL,
    "message" TEXT NOT NULL,
    "sent" BOOLEAN NOT NULL DEFAULT false,
    CONSTRAINT "Notification_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "Notification_cycleId_fkey" FOREIGN KEY ("cycleId") REFERENCES "MedicationCycle" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "new_Notification" ("cycleId", "date", "id", "message", "sent", "userId") SELECT "cycleId", "date", "id", "message", "sent", "userId" FROM "Notification";
DROP TABLE "Notification";
ALTER TABLE "new_Notification" RENAME TO "Notification";
CREATE TABLE "new_ai_audit_log" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "field_name" TEXT NOT NULL,
    "old_value" TEXT,
    "new_value" TEXT,
    "ai_confidence" REAL,
    "ai_suggestion" TEXT,
    "clinician_approved" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "ai_audit_log_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_ai_audit_log" ("ai_confidence", "ai_suggestion", "clinician_approved", "field_name", "id", "new_value", "old_value") SELECT "ai_confidence", "ai_suggestion", coalesce("clinician_approved", false) AS "clinician_approved", "field_name", "id", "new_value", "old_value" FROM "ai_audit_log";
DROP TABLE "ai_audit_log";
ALTER TABLE "new_ai_audit_log" RENAME TO "ai_audit_log";
CREATE INDEX "ai_audit_log_patientId_idx" ON "ai_audit_log"("patientId");
CREATE INDEX "ai_audit_log_createdAt_idx" ON "ai_audit_log"("createdAt");
CREATE TABLE "new_clinicians" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "hospitalCode" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "clinicians_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_clinicians" ("createdAt", "hospitalCode", "id", "updatedAt", "userId") SELECT "createdAt", "hospitalCode", "id", "updatedAt", "userId" FROM "clinicians";
DROP TABLE "clinicians";
ALTER TABLE "new_clinicians" RENAME TO "clinicians";
CREATE UNIQUE INDEX "clinicians_userId_key" ON "clinicians"("userId");
CREATE TABLE "new_patients" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "patient_audit_id" TEXT,
    "imd_decile" INTEGER,
    "dob" DATETIME,
    "sex" TEXT,
    "ethnicity" TEXT,
    "ethnic_group" TEXT,
    "location" TEXT,
    "postcode" TEXT,
    "nhs_number" TEXT,
    "mrn" TEXT,
    "start_date" DATETIME,
    "height" REAL,
    "baseline_weight" REAL,
    "baseline_bmi" REAL,
    "baseline_weight_date" DATETIME,
    "ascvd" BOOLEAN,
    "htn" BOOLEAN,
    "hypertension" BOOLEAN,
    "dyslipidaemia" BOOLEAN,
    "ischaemic_heart_disease" BOOLEAN,
    "heart_failure" BOOLEAN,
    "cerebrovascular_disease" BOOLEAN,
    "pulmonary_hypertension" BOOLEAN,
    "dvt" BOOLEAN,
    "pe" BOOLEAN,
    "osa" BOOLEAN,
    "sleep_studies" BOOLEAN,
    "cpap" BOOLEAN,
    "asthma" BOOLEAN,
    "t2dm" BOOLEAN,
    "prediabetes" BOOLEAN,
    "diabetes_type" TEXT,
    "hba1c_percent" REAL,
    "hba1c_mmol" REAL,
    "baseline_hba1c" REAL,
    "baseline_hba1c_date" DATETIME,
    "baseline_fasting_glucose" REAL,
    "random_glucose" REAL,
    "baseline_tc" REAL,
    "baseline_hdl" REAL,
    "baseline_ldl" REAL,
    "baseline_tg" REAL,
    "baseline_lipid_date" DATETIME,
    "lipid_lowering_treatment" TEXT,
    "antihypertensive_medications" TEXT,
    "all_medications_from_scr" TEXT,
    "gord" BOOLEAN,
    "ckd" BOOLEAN,
    "kidney_stones" BOOLEAN,
    "masld" BOOLEAN,
    "infertility" BOOLEAN,
    "pcos" BOOLEAN,
    "anxiety" BOOLEAN,
    "depression" BOOLEAN,
    "bipolar_disorder" BOOLEAN,
    "emotional_eating" BOOLEAN,
    "schizoaffective_disorder" BOOLEAN,
    "oa_knee" BOOLEAN,
    "oa_hip" BOOLEAN,
    "limited_mobility" BOOLEAN,
    "lymphoedema" BOOLEAN,
    "thyroid_disorder" BOOLEAN,
    "iih" BOOLEAN,
    "epilepsy" BOOLEAN,
    "functional_neurological_disorder" BOOLEAN,
    "cancer" BOOLEAN,
    "bariatric_gastric_band" BOOLEAN,
    "bariatric_sleeve" BOOLEAN,
    "bariatric_bypass" BOOLEAN,
    "bariatric_balloon" BOOLEAN,
    "diagnoses_coded_in_scr" TEXT,
    "total_qualifying_comorbidities" INTEGER,
    "mes" REAL,
    "notes" TEXT,
    "criteria_for_wegovy" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "patients_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_patients" ("all_medications_from_scr", "antihypertensive_medications", "anxiety", "ascvd", "asthma", "bariatric_balloon", "bariatric_bypass", "bariatric_gastric_band", "bariatric_sleeve", "baseline_bmi", "baseline_fasting_glucose", "baseline_hba1c", "baseline_hba1c_date", "baseline_hdl", "baseline_ldl", "baseline_lipid_date", "baseline_tc", "baseline_tg", "baseline_weight", "baseline_weight_date", "bipolar_disorder", "cancer", "cerebrovascular_disease", "ckd", "cpap", "createdAt", "criteria_for_wegovy", "depression", "diabetes_type", "diagnoses_coded_in_scr", "dob", "dvt", "dyslipidaemia", "emotional_eating", "epilepsy", "ethnic_group", "ethnicity", "functional_neurological_disorder", "gord", "hba1c_mmol", "hba1c_percent", "heart_failure", "height", "htn", "hypertension", "id", "iih", "imd_decile", "infertility", "ischaemic_heart_disease", "kidney_stones", "limited_mobility", "lipid_lowering_treatment", "location", "lymphoedema", "masld", "mes", "mrn", "nhs_number", "notes", "oa_hip", "oa_knee", "osa", "patient_audit_id", "pcos", "pe", "postcode", "prediabetes", "pulmonary_hypertension", "random_glucose", "schizoaffective_disorder", "sex", "sleep_studies", "start_date", "t2dm", "thyroid_disorder", "total_qualifying_comorbidities", "updatedAt", "userId") SELECT "all_medications_from_scr", "antihypertensive_medications", "anxiety", "ascvd", "asthma", "bariatric_balloon", "bariatric_bypass", "bariatric_gastric_band", "bariatric_sleeve", "baseline_bmi", "baseline_fasting_glucose", "baseline_hba1c", "baseline_hba1c_date", "baseline_hdl", "baseline_ldl", "baseline_lipid_date", "baseline_tc", "baseline_tg", "baseline_weight", "baseline_weight_date", "bipolar_disorder", "cancer", "cerebrovascular_disease", "ckd", "cpap", "createdAt", "criteria_for_wegovy", "depression", "diabetes_type", "diagnoses_coded_in_scr", "dob", "dvt", "dyslipidaemia", "emotional_eating", "epilepsy", "ethnic_group", "ethnicity", "functional_neurological_disorder", "gord", "hba1c_mmol", "hba1c_percent", "heart_failure", "height", "htn", "hypertension", "id", "iih", "imd_decile", "infertility", "ischaemic_heart_disease", "kidney_stones", "limited_mobility", "lipid_lowering_treatment", "location", "lymphoedema", "masld", "mes", "mrn", "nhs_number", "notes", "oa_hip", "oa_knee", "osa", "patient_audit_id", "pcos", "pe", "postcode", "prediabetes", "pulmonary_hypertension", "random_glucose", "schizoaffective_disorder", "sex", "sleep_studies", "start_date", "t2dm", "thyroid_disorder", "total_qualifying_comorbidities", "updatedAt", "userId" FROM "patients";
DROP TABLE "patients";
ALTER TABLE "new_patients" RENAME TO "patients";
CREATE UNIQUE INDEX "patients_userId_key" ON "patients"("userId");
CREATE UNIQUE INDEX "patients_patient_audit_id_key" ON "patients"("patient_audit_id");
CREATE UNIQUE INDEX "patients_nhs_number_key" ON "patients"("nhs_number");
CREATE UNIQUE INDEX "patients_mrn_key" ON "patients"("mrn");
CREATE INDEX "patients_nhs_number_idx" ON "patients"("nhs_number");
CREATE INDEX "patients_mrn_idx" ON "patients"("mrn");
CREATE INDEX "patients_postcode_idx" ON "patients"("postcode");
CREATE INDEX "patients_ethnic_group_idx" ON "patients"("ethnic_group");
CREATE TABLE "new_user_survey_data" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "dateOfBirth" DATETIME,
    "biologicalSex" TEXT,
    "ethnicity" TEXT,
    "hasMenses" BOOLEAN,
    "ageAtMenarche" INTEGER,
    "menstrualRegularity" TEXT,
    "lastMenstrualPeriod" DATETIME,
    "cycleLength" INTEGER,
    "periodDuration" INTEGER,
    "usesContraception" BOOLEAN,
    "contraceptionType" TEXT,
    "hasPreviousPregnancies" BOOLEAN,
    "isPerimenopausal" BOOLEAN,
    "isPostmenopausal" BOOLEAN,
    "ageAtMenopause" INTEGER,
    "menopauseType" TEXT,
    "isOnHRT" BOOLEAN,
    "hrtType" TEXT,
    "iiefScore" INTEGER,
    "lowTestosteroneSymptoms" TEXT,
    "redFlagQuestions" TEXT,
    "auditScore" INTEGER,
    "smokingStatus" TEXT,
    "smokingStartAge" INTEGER,
    "cigarettesPerDay" INTEGER,
    "vapingDevice" TEXT,
    "nicotineMg" REAL,
    "pgVgRatio" TEXT,
    "usagePattern" TEXT,
    "psecdiScore" INTEGER,
    "readinessToQuit" INTEGER,
    "ipaqScore" INTEGER,
    "weight" REAL,
    "height" REAL,
    "waistCircumference" REAL,
    "hipCircumference" REAL,
    "neckCircumference" REAL,
    "systolicBP" INTEGER,
    "diastolicBP" INTEGER,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "user_survey_data_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_user_survey_data" ("ageAtMenarche", "ageAtMenopause", "auditScore", "biologicalSex", "cigarettesPerDay", "contraceptionType", "createdAt", "cycleLength", "dateOfBirth", "diastolicBP", "ethnicity", "hasMenses", "hasPreviousPregnancies", "height", "hipCircumference", "hrtType", "id", "iiefScore", "ipaqScore", "isOnHRT", "isPerimenopausal", "isPostmenopausal", "lastMenstrualPeriod", "lowTestosteroneSymptoms", "menopauseType", "menstrualRegularity", "neckCircumference", "nicotineMg", "periodDuration", "pgVgRatio", "psecdiScore", "readinessToQuit", "redFlagQuestions", "smokingStartAge", "smokingStatus", "systolicBP", "updatedAt", "usagePattern", "userId", "usesContraception", "vapingDevice", "waistCircumference", "weight") SELECT "ageAtMenarche", "ageAtMenopause", "auditScore", "biologicalSex", "cigarettesPerDay", "contraceptionType", "createdAt", "cycleLength", "dateOfBirth", "diastolicBP", "ethnicity", "hasMenses", "hasPreviousPregnancies", "height", "hipCircumference", "hrtType", "id", "iiefScore", "ipaqScore", "isOnHRT", "isPerimenopausal", "isPostmenopausal", "lastMenstrualPeriod", "lowTestosteroneSymptoms", "menopauseType", "menstrualRegularity", "neckCircumference", "nicotineMg", "periodDuration", "pgVgRatio", "psecdiScore", "readinessToQuit", "redFlagQuestions", "smokingStartAge", "smokingStatus", "systolicBP", "updatedAt", "usagePattern", "userId", "usesContraception", "vapingDevice", "waistCircumference", "weight" FROM "user_survey_data";
DROP TABLE "user_survey_data";
ALTER TABLE "new_user_survey_data" RENAME TO "user_survey_data";
CREATE UNIQUE INDEX "user_survey_data_userId_key" ON "user_survey_data"("userId");
CREATE TABLE "new_users" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "role" TEXT NOT NULL DEFAULT 'patient',
    "hospitalCode" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "resetToken" TEXT,
    "resetTokenExpiry" DATETIME,
    "is2FAEnabled" BOOLEAN NOT NULL DEFAULT false,
    "twoFASecret" TEXT,
    "surveyCompleted" BOOLEAN NOT NULL DEFAULT false
);
INSERT INTO "new_users" ("createdAt", "email", "hospitalCode", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "role", "surveyCompleted", "twoFASecret", "updatedAt") SELECT coalesce("createdAt", CURRENT_TIMESTAMP) AS "createdAt", "email", "hospitalCode", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "role", "surveyCompleted", "twoFASecret", "updatedAt" FROM "users";
DROP TABLE "users";
ALTER TABLE "new_users" RENAME TO "users";
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");
CREATE INDEX "users_hospitalCode_idx" ON "users"("hospitalCode");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

-- CreateIndex
CREATE INDEX "medical_notes_patientId_idx" ON "medical_notes"("patientId");

-- CreateIndex
CREATE INDEX "medical_notes_date_of_entry_idx" ON "medical_notes"("date_of_entry");

-- CreateIndex
CREATE INDEX "medical_notes_processing_status_idx" ON "medical_notes"("processing_status");

-- CreateIndex
CREATE INDEX "patient_metrics_patientId_idx" ON "patient_metrics"("patientId");

-- CreateIndex
CREATE INDEX "patient_metrics_metric_name_idx" ON "patient_metrics"("metric_name");

-- CreateIndex
CREATE INDEX "patient_metrics_date_recorded_idx" ON "patient_metrics"("date_recorded");

-- CreateIndex
CREATE INDEX "patient_medications_patientId_idx" ON "patient_medications"("patientId");

-- CreateIndex
CREATE INDEX "patient_medications_name_idx" ON "patient_medications"("name");

-- CreateIndex
CREATE INDEX "patient_medications_status_idx" ON "patient_medications"("status");

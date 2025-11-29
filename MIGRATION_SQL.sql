-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "role" TEXT NOT NULL DEFAULT 'patient',
    "hospitalCode" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "resetToken" TEXT,
    "resetTokenExpiry" TIMESTAMP(3),
    "is2FAEnabled" BOOLEAN NOT NULL DEFAULT false,
    "twoFASecret" TEXT,
    "surveyCompleted" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "patients" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "patient_audit_id" TEXT,
    "imd_decile" INTEGER,
    "dob" TIMESTAMP(3),
    "sex" TEXT,
    "ethnicity" TEXT,
    "ethnic_group" TEXT,
    "location" TEXT,
    "postcode" TEXT,
    "nhs_number" TEXT,
    "mrn" TEXT,
    "start_date" TIMESTAMP(3),
    "height" DOUBLE PRECISION,
    "baseline_weight" DOUBLE PRECISION,
    "baseline_bmi" DOUBLE PRECISION,
    "baseline_weight_date" TIMESTAMP(3),
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
    "hba1c_percent" DOUBLE PRECISION,
    "hba1c_mmol" DOUBLE PRECISION,
    "baseline_hba1c" DOUBLE PRECISION,
    "baseline_hba1c_date" TIMESTAMP(3),
    "baseline_fasting_glucose" DOUBLE PRECISION,
    "random_glucose" DOUBLE PRECISION,
    "baseline_tc" DOUBLE PRECISION,
    "baseline_hdl" DOUBLE PRECISION,
    "baseline_ldl" DOUBLE PRECISION,
    "baseline_tg" DOUBLE PRECISION,
    "baseline_lipid_date" TIMESTAMP(3),
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
    "mes" DOUBLE PRECISION,
    "notes" TEXT,
    "criteria_for_wegovy" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "patients_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "clinicians" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "hospitalCode" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "clinicians_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "conditions" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "normalized" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "conditions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ai_audit_log" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "field_name" TEXT NOT NULL,
    "old_value" TEXT,
    "new_value" TEXT,
    "ai_confidence" DOUBLE PRECISION,
    "ai_suggestion" TEXT,
    "clinician_approved" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ai_audit_log_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Medication" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "patientId" TEXT,
    "name" TEXT NOT NULL,
    "genericName" TEXT,
    "brandName" TEXT,
    "startDate" TIMESTAMP(3) NOT NULL,
    "endDate" TIMESTAMP(3),
    "dosage" TEXT NOT NULL,
    "unit" TEXT,
    "frequency" TEXT NOT NULL,
    "customFrequency" TEXT,
    "route" TEXT,
    "strength" TEXT,
    "drugClass" TEXT,
    "ndcCode" TEXT,
    "rxnormCode" TEXT,
    "reminderEnabled" BOOLEAN NOT NULL DEFAULT true,
    "reminderTimes" TEXT,
    "reminderDays" TEXT,
    "intervalHours" INTEGER,
    "reminderChainId" TEXT,
    "stockQuantity" DOUBLE PRECISION,
    "stockUnit" TEXT,
    "lowStockThreshold" DOUBLE PRECISION,
    "outOfStock" BOOLEAN NOT NULL DEFAULT false,
    "weekendMode" BOOLEAN NOT NULL DEFAULT false,
    "weekendDelayDays" TEXT,
    "notes" TEXT,
    "specialInstructions" TEXT,
    "status" TEXT NOT NULL DEFAULT 'active',
    "taken" BOOLEAN NOT NULL DEFAULT false,
    "takenAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Medication_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "medication_logs" (
    "id" TEXT NOT NULL,
    "medicationId" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "time" TIMESTAMP(3),
    "dosage" TEXT NOT NULL,
    "taken" BOOLEAN NOT NULL DEFAULT false,
    "skipped" BOOLEAN NOT NULL DEFAULT false,
    "notes" TEXT,

    CONSTRAINT "medication_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "medication_side_effects" (
    "id" TEXT NOT NULL,
    "medicationId" TEXT NOT NULL,
    "symptom" TEXT NOT NULL,
    "severity" TEXT,
    "onsetDate" TIMESTAMP(3) NOT NULL,
    "resolvedDate" TIMESTAMP(3),
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "medication_side_effects_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "medication_adherence_logs" (
    "id" TEXT NOT NULL,
    "medicationId" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "scheduledTime" TIMESTAMP(3),
    "takenTime" TIMESTAMP(3),
    "status" TEXT NOT NULL,
    "delayMinutes" INTEGER,
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "medication_adherence_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "reminder_chains" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "name" TEXT,
    "firstMedicationId" TEXT NOT NULL,
    "secondMedicationId" TEXT NOT NULL,
    "delayHours" INTEGER NOT NULL,
    "active" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "reminder_chains_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "patient_profiles" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "relationship" TEXT,
    "isPrimary" BOOLEAN NOT NULL DEFAULT false,
    "color" TEXT,
    "avatar" TEXT,
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "patient_profiles_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "diary_entries" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "patientId" TEXT,
    "date" TIMESTAMP(3) NOT NULL,
    "entryType" TEXT NOT NULL,
    "title" TEXT,
    "content" TEXT,
    "attributes" TEXT,
    "tags" TEXT,
    "notebookId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "diary_entries_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "custom_attributes" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "options" TEXT,
    "unit" TEXT,
    "color" TEXT,
    "icon" TEXT,
    "active" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "custom_attributes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "drug_interactions" (
    "id" TEXT NOT NULL,
    "medication1Id" TEXT NOT NULL,
    "medication2Id" TEXT NOT NULL,
    "interactionType" TEXT NOT NULL,
    "severity" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "clinicalSignificance" TEXT,
    "management" TEXT,
    "source" TEXT,
    "verified" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "drug_interactions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "pill_recognitions" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "patientId" TEXT,
    "imagePath" TEXT NOT NULL,
    "imageUrl" TEXT,
    "recognized" BOOLEAN NOT NULL DEFAULT false,
    "confidence" DOUBLE PRECISION,
    "medicationName" TEXT,
    "imprint" TEXT,
    "shape" TEXT,
    "color" TEXT,
    "size" TEXT,
    "mlModel" TEXT,
    "rawResult" TEXT,
    "verified" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "pill_recognitions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "health_reports" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "patientId" TEXT,
    "reportType" TEXT NOT NULL,
    "periodStart" TIMESTAMP(3) NOT NULL,
    "periodEnd" TIMESTAMP(3) NOT NULL,
    "data" TEXT NOT NULL,
    "charts" TEXT,
    "insights" TEXT,
    "generatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "health_reports_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "data_exports" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "exportType" TEXT NOT NULL,
    "format" TEXT NOT NULL,
    "filePath" TEXT,
    "fileUrl" TEXT,
    "expiresAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "data_exports_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Metric" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "weight" DOUBLE PRECISION NOT NULL,
    "height" DOUBLE PRECISION NOT NULL,
    "bmi" DOUBLE PRECISION NOT NULL,
    "bloodPressure" TEXT NOT NULL,
    "hipCircumference" DOUBLE PRECISION,

    CONSTRAINT "Metric_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MedicationCycle" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "dosage" TEXT NOT NULL,
    "startDate" TIMESTAMP(3) NOT NULL,
    "endDate" TIMESTAMP(3),
    "frequencyDays" INTEGER NOT NULL,
    "dosesPerDay" INTEGER NOT NULL DEFAULT 1,
    "metricsToMonitor" TEXT,

    CONSTRAINT "MedicationCycle_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MetricLog" (
    "id" TEXT NOT NULL,
    "cycleId" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "kind" TEXT NOT NULL,
    "valueFloat" DOUBLE PRECISION,
    "valueText" TEXT,
    "notes" TEXT,

    CONSTRAINT "MetricLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "DoseLog" (
    "id" TEXT NOT NULL,
    "cycleId" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "taken" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "DoseLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Notification" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "cycleId" TEXT,
    "date" TIMESTAMP(3) NOT NULL,
    "message" TEXT NOT NULL,
    "sent" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "Notification_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "user_survey_data" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "dateOfBirth" TIMESTAMP(3),
    "biologicalSex" TEXT,
    "ethnicity" TEXT,
    "hasMenses" BOOLEAN,
    "ageAtMenarche" INTEGER,
    "menstrualRegularity" TEXT,
    "lastMenstrualPeriod" TIMESTAMP(3),
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
    "nicotineMg" DOUBLE PRECISION,
    "pgVgRatio" TEXT,
    "usagePattern" TEXT,
    "psecdiScore" INTEGER,
    "readinessToQuit" INTEGER,
    "ipaqScore" INTEGER,
    "weight" DOUBLE PRECISION,
    "height" DOUBLE PRECISION,
    "waistCircumference" DOUBLE PRECISION,
    "hipCircumference" DOUBLE PRECISION,
    "neckCircumference" DOUBLE PRECISION,
    "systolicBP" INTEGER,
    "diastolicBP" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "user_survey_data_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "medical_notes" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "raw_text" TEXT NOT NULL,
    "note_type" TEXT,
    "source" TEXT,
    "ai_processed" BOOLEAN NOT NULL DEFAULT false,
    "ai_confidence" DOUBLE PRECISION,
    "ai_model_used" TEXT,
    "processing_errors" TEXT,
    "extracted_data" TEXT,
    "patient_name" TEXT,
    "age" INTEGER,
    "sex" TEXT,
    "conditions" TEXT,
    "medications" TEXT,
    "allergies" TEXT,
    "lab_results" TEXT,
    "vital_signs" TEXT,
    "impression" TEXT,
    "plan" TEXT,
    "created_by" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "medical_notes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "lab_results" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "metric_name" TEXT NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "unit" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "reference_range" TEXT,
    "status" TEXT,
    "source_note_id" TEXT,
    "manually_entered" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "lab_results_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "vital_signs" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "encounterId" TEXT,
    "vital_type" TEXT NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "unit" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "value_secondary" DOUBLE PRECISION,
    "source_note_id" TEXT,
    "manually_entered" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "vital_signs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "patient_medications" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "dosage" TEXT NOT NULL,
    "frequency" TEXT NOT NULL,
    "route" TEXT,
    "start_date" TIMESTAMP(3) NOT NULL,
    "end_date" TIMESTAMP(3),
    "status" TEXT NOT NULL DEFAULT 'active',
    "source_note_id" TEXT,
    "manually_entered" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "patient_medications_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "metric_trends" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "metric_name" TEXT NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "unit" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "change_from_previous" DOUBLE PRECISION,
    "change_percentage" DOUBLE PRECISION,
    "trend_direction" TEXT,
    "source_type" TEXT NOT NULL,
    "source_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "metric_trends_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ai_processing_config" (
    "id" TEXT NOT NULL,
    "model_name" TEXT NOT NULL,
    "model_provider" TEXT NOT NULL,
    "system_prompt" TEXT NOT NULL,
    "max_tokens" INTEGER NOT NULL DEFAULT 2048,
    "temperature" DOUBLE PRECISION NOT NULL DEFAULT 0.1,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ai_processing_config_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "appointments" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "providerId" TEXT,
    "facilityId" TEXT,
    "appointmentDate" TIMESTAMP(3) NOT NULL,
    "appointmentTime" TIMESTAMP(3) NOT NULL,
    "duration" INTEGER NOT NULL DEFAULT 30,
    "appointmentType" TEXT,
    "status" TEXT NOT NULL DEFAULT 'scheduled',
    "reason" TEXT,
    "notes" TEXT,
    "reminderSent" BOOLEAN NOT NULL DEFAULT false,
    "reminderSentAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT,

    CONSTRAINT "appointments_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "encounters" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "providerId" TEXT,
    "facilityId" TEXT,
    "encounterDate" TIMESTAMP(3) NOT NULL,
    "encounterTime" TIMESTAMP(3),
    "encounterType" TEXT,
    "reason" TEXT,
    "status" TEXT NOT NULL DEFAULT 'finished',
    "priority" TEXT,
    "billingFacilityId" TEXT,
    "classCode" TEXT,
    "classDisplay" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT,
    "updatedBy" TEXT,

    CONSTRAINT "encounters_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "soap_notes" (
    "id" TEXT NOT NULL,
    "encounterId" TEXT NOT NULL,
    "subjective" TEXT,
    "objective" TEXT,
    "assessment" TEXT,
    "plan" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT,
    "updatedBy" TEXT,

    CONSTRAINT "soap_notes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "problems" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "encounterId" TEXT,
    "title" TEXT NOT NULL,
    "code" TEXT,
    "codeType" TEXT NOT NULL DEFAULT 'ICD10',
    "beginDate" TIMESTAMP(3),
    "endDate" TIMESTAMP(3),
    "status" TEXT NOT NULL DEFAULT 'active',
    "severity" TEXT,
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT,

    CONSTRAINT "problems_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "allergies" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "allergen" TEXT NOT NULL,
    "allergenType" TEXT,
    "reaction" TEXT,
    "severity" TEXT,
    "onsetDate" TIMESTAMP(3),
    "status" TEXT NOT NULL DEFAULT 'active',
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT,

    CONSTRAINT "allergies_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "immunizations" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "vaccineName" TEXT NOT NULL,
    "vaccineCode" TEXT,
    "administrationDate" TIMESTAMP(3) NOT NULL,
    "lotNumber" TEXT,
    "manufacturer" TEXT,
    "route" TEXT,
    "site" TEXT,
    "dose" TEXT,
    "provider" TEXT,
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT,

    CONSTRAINT "immunizations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "documents" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "encounterId" TEXT,
    "documentType" TEXT,
    "category" TEXT,
    "title" TEXT NOT NULL,
    "filePath" TEXT NOT NULL,
    "fileName" TEXT NOT NULL,
    "fileSize" BIGINT,
    "mimeType" TEXT,
    "date" TIMESTAMP(3),
    "status" TEXT NOT NULL DEFAULT 'active',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT,

    CONSTRAINT "documents_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "prescriptions" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "encounterId" TEXT,
    "providerId" TEXT,
    "medicationName" TEXT NOT NULL,
    "ndcCode" TEXT,
    "rxnormCode" TEXT,
    "dosage" TEXT,
    "unit" TEXT,
    "route" TEXT,
    "frequency" TEXT,
    "quantity" DOUBLE PRECISION,
    "refills" INTEGER NOT NULL DEFAULT 0,
    "startDate" TIMESTAMP(3) NOT NULL,
    "endDate" TIMESTAMP(3),
    "status" TEXT NOT NULL DEFAULT 'active',
    "instructions" TEXT,
    "pharmacyId" TEXT,
    "datePrescribed" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "dateFilled" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT,

    CONSTRAINT "prescriptions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "pharmacies" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "npi" TEXT,
    "address" TEXT,
    "phone" TEXT,
    "fax" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "pharmacies_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "charges" (
    "id" TEXT NOT NULL,
    "encounterId" TEXT,
    "patientId" TEXT NOT NULL,
    "code" TEXT NOT NULL,
    "codeType" TEXT NOT NULL DEFAULT 'CPT',
    "description" TEXT,
    "units" DOUBLE PRECISION NOT NULL DEFAULT 1,
    "fee" DOUBLE PRECISION NOT NULL,
    "dateOfService" TIMESTAMP(3) NOT NULL,
    "providerId" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdBy" TEXT,

    CONSTRAINT "charges_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "payments" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "encounterId" TEXT,
    "amount" DOUBLE PRECISION NOT NULL,
    "paymentMethod" TEXT,
    "paymentDate" TIMESTAMP(3) NOT NULL,
    "checkNumber" TEXT,
    "creditCardLast4" TEXT,
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdBy" TEXT,

    CONSTRAINT "payments_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "payment_allocations" (
    "id" TEXT NOT NULL,
    "paymentId" TEXT NOT NULL,
    "chargeId" TEXT NOT NULL,
    "amount" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "payment_allocations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "claims" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "insuranceId" TEXT,
    "claimNumber" TEXT,
    "totalCharges" DOUBLE PRECISION,
    "totalPaid" DOUBLE PRECISION,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "submittedDate" TIMESTAMP(3),
    "paidDate" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "claims_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "claim_lines" (
    "id" TEXT NOT NULL,
    "claimId" TEXT NOT NULL,
    "chargeId" TEXT,
    "lineNumber" INTEGER,
    "code" TEXT,
    "units" DOUBLE PRECISION,
    "fee" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "claim_lines_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "patient_insurance" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "insuranceType" TEXT NOT NULL,
    "insuranceCompanyId" TEXT,
    "policyNumber" TEXT,
    "groupNumber" TEXT,
    "subscriberName" TEXT,
    "subscriberSsn" TEXT,
    "subscriberDob" TIMESTAMP(3),
    "copayAmount" DOUBLE PRECISION,
    "effectiveDate" TIMESTAMP(3),
    "expirationDate" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "patient_insurance_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "insurance_companies" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "payerId" TEXT,
    "address" TEXT,
    "phone" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "insurance_companies_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "facilities" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "address" TEXT,
    "city" TEXT,
    "state" TEXT,
    "zipCode" TEXT,
    "phone" TEXT,
    "email" TEXT,
    "npi" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "facilities_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "appointment_categories" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "duration" INTEGER NOT NULL DEFAULT 30,
    "color" TEXT,
    "facilityId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "appointment_categories_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "recurring_appointments" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "providerId" TEXT,
    "facilityId" TEXT,
    "startDate" TIMESTAMP(3) NOT NULL,
    "endDate" TIMESTAMP(3),
    "frequency" TEXT,
    "dayOfWeek" INTEGER,
    "dayOfMonth" INTEGER,
    "appointmentType" TEXT,
    "duration" INTEGER NOT NULL DEFAULT 30,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "recurring_appointments_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE INDEX "users_hospitalCode_idx" ON "users"("hospitalCode");

-- CreateIndex
CREATE UNIQUE INDEX "patients_userId_key" ON "patients"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "patients_patient_audit_id_key" ON "patients"("patient_audit_id");

-- CreateIndex
CREATE UNIQUE INDEX "patients_nhs_number_key" ON "patients"("nhs_number");

-- CreateIndex
CREATE UNIQUE INDEX "patients_mrn_key" ON "patients"("mrn");

-- CreateIndex
CREATE INDEX "patients_nhs_number_idx" ON "patients"("nhs_number");

-- CreateIndex
CREATE INDEX "patients_mrn_idx" ON "patients"("mrn");

-- CreateIndex
CREATE INDEX "patients_postcode_idx" ON "patients"("postcode");

-- CreateIndex
CREATE INDEX "patients_ethnic_group_idx" ON "patients"("ethnic_group");

-- CreateIndex
CREATE UNIQUE INDEX "clinicians_userId_key" ON "clinicians"("userId");

-- CreateIndex
CREATE INDEX "ai_audit_log_patientId_idx" ON "ai_audit_log"("patientId");

-- CreateIndex
CREATE INDEX "ai_audit_log_createdAt_idx" ON "ai_audit_log"("createdAt");

-- CreateIndex
CREATE INDEX "Medication_userId_idx" ON "Medication"("userId");

-- CreateIndex
CREATE INDEX "Medication_patientId_idx" ON "Medication"("patientId");

-- CreateIndex
CREATE INDEX "Medication_status_idx" ON "Medication"("status");

-- CreateIndex
CREATE INDEX "medication_logs_medicationId_date_idx" ON "medication_logs"("medicationId", "date");

-- CreateIndex
CREATE INDEX "medication_side_effects_medicationId_idx" ON "medication_side_effects"("medicationId");

-- CreateIndex
CREATE INDEX "medication_adherence_logs_medicationId_idx" ON "medication_adherence_logs"("medicationId");

-- CreateIndex
CREATE INDEX "medication_adherence_logs_date_idx" ON "medication_adherence_logs"("date");

-- CreateIndex
CREATE UNIQUE INDEX "medication_adherence_logs_medicationId_date_key" ON "medication_adherence_logs"("medicationId", "date");

-- CreateIndex
CREATE INDEX "reminder_chains_userId_idx" ON "reminder_chains"("userId");

-- CreateIndex
CREATE INDEX "patient_profiles_userId_idx" ON "patient_profiles"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "patient_profiles_userId_patientId_key" ON "patient_profiles"("userId", "patientId");

-- CreateIndex
CREATE INDEX "diary_entries_userId_idx" ON "diary_entries"("userId");

-- CreateIndex
CREATE INDEX "diary_entries_patientId_idx" ON "diary_entries"("patientId");

-- CreateIndex
CREATE INDEX "diary_entries_date_idx" ON "diary_entries"("date");

-- CreateIndex
CREATE INDEX "diary_entries_entryType_idx" ON "diary_entries"("entryType");

-- CreateIndex
CREATE INDEX "custom_attributes_userId_idx" ON "custom_attributes"("userId");

-- CreateIndex
CREATE INDEX "drug_interactions_medication1Id_idx" ON "drug_interactions"("medication1Id");

-- CreateIndex
CREATE INDEX "drug_interactions_medication2Id_idx" ON "drug_interactions"("medication2Id");

-- CreateIndex
CREATE UNIQUE INDEX "drug_interactions_medication1Id_medication2Id_key" ON "drug_interactions"("medication1Id", "medication2Id");

-- CreateIndex
CREATE INDEX "pill_recognitions_userId_idx" ON "pill_recognitions"("userId");

-- CreateIndex
CREATE INDEX "pill_recognitions_patientId_idx" ON "pill_recognitions"("patientId");

-- CreateIndex
CREATE INDEX "health_reports_userId_idx" ON "health_reports"("userId");

-- CreateIndex
CREATE INDEX "health_reports_patientId_idx" ON "health_reports"("patientId");

-- CreateIndex
CREATE INDEX "health_reports_generatedAt_idx" ON "health_reports"("generatedAt");

-- CreateIndex
CREATE INDEX "data_exports_userId_idx" ON "data_exports"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "DoseLog_cycleId_date_key" ON "DoseLog"("cycleId", "date");

-- CreateIndex
CREATE UNIQUE INDEX "user_survey_data_userId_key" ON "user_survey_data"("userId");

-- CreateIndex
CREATE INDEX "medical_notes_patientId_idx" ON "medical_notes"("patientId");

-- CreateIndex
CREATE INDEX "medical_notes_created_at_idx" ON "medical_notes"("created_at");

-- CreateIndex
CREATE INDEX "medical_notes_ai_processed_idx" ON "medical_notes"("ai_processed");

-- CreateIndex
CREATE INDEX "lab_results_patientId_idx" ON "lab_results"("patientId");

-- CreateIndex
CREATE INDEX "lab_results_metric_name_idx" ON "lab_results"("metric_name");

-- CreateIndex
CREATE INDEX "lab_results_date_idx" ON "lab_results"("date");

-- CreateIndex
CREATE INDEX "vital_signs_patientId_idx" ON "vital_signs"("patientId");

-- CreateIndex
CREATE INDEX "vital_signs_encounterId_idx" ON "vital_signs"("encounterId");

-- CreateIndex
CREATE INDEX "vital_signs_vital_type_idx" ON "vital_signs"("vital_type");

-- CreateIndex
CREATE INDEX "vital_signs_date_idx" ON "vital_signs"("date");

-- CreateIndex
CREATE INDEX "patient_medications_patientId_idx" ON "patient_medications"("patientId");

-- CreateIndex
CREATE INDEX "patient_medications_status_idx" ON "patient_medications"("status");

-- CreateIndex
CREATE INDEX "patient_medications_start_date_idx" ON "patient_medications"("start_date");

-- CreateIndex
CREATE INDEX "metric_trends_patientId_idx" ON "metric_trends"("patientId");

-- CreateIndex
CREATE INDEX "metric_trends_metric_name_idx" ON "metric_trends"("metric_name");

-- CreateIndex
CREATE INDEX "metric_trends_date_idx" ON "metric_trends"("date");

-- CreateIndex
CREATE INDEX "appointments_patientId_idx" ON "appointments"("patientId");

-- CreateIndex
CREATE INDEX "appointments_providerId_idx" ON "appointments"("providerId");

-- CreateIndex
CREATE INDEX "appointments_appointmentDate_idx" ON "appointments"("appointmentDate");

-- CreateIndex
CREATE INDEX "appointments_status_idx" ON "appointments"("status");

-- CreateIndex
CREATE INDEX "encounters_patientId_idx" ON "encounters"("patientId");

-- CreateIndex
CREATE INDEX "encounters_providerId_idx" ON "encounters"("providerId");

-- CreateIndex
CREATE INDEX "encounters_encounterDate_idx" ON "encounters"("encounterDate");

-- CreateIndex
CREATE INDEX "encounters_status_idx" ON "encounters"("status");

-- CreateIndex
CREATE INDEX "soap_notes_encounterId_idx" ON "soap_notes"("encounterId");

-- CreateIndex
CREATE INDEX "problems_patientId_idx" ON "problems"("patientId");

-- CreateIndex
CREATE INDEX "problems_status_idx" ON "problems"("status");

-- CreateIndex
CREATE INDEX "problems_code_idx" ON "problems"("code");

-- CreateIndex
CREATE INDEX "allergies_patientId_idx" ON "allergies"("patientId");

-- CreateIndex
CREATE INDEX "allergies_status_idx" ON "allergies"("status");

-- CreateIndex
CREATE INDEX "immunizations_patientId_idx" ON "immunizations"("patientId");

-- CreateIndex
CREATE INDEX "immunizations_administrationDate_idx" ON "immunizations"("administrationDate");

-- CreateIndex
CREATE INDEX "documents_patientId_idx" ON "documents"("patientId");

-- CreateIndex
CREATE INDEX "documents_encounterId_idx" ON "documents"("encounterId");

-- CreateIndex
CREATE INDEX "documents_documentType_idx" ON "documents"("documentType");

-- CreateIndex
CREATE INDEX "prescriptions_patientId_idx" ON "prescriptions"("patientId");

-- CreateIndex
CREATE INDEX "prescriptions_status_idx" ON "prescriptions"("status");

-- CreateIndex
CREATE INDEX "prescriptions_datePrescribed_idx" ON "prescriptions"("datePrescribed");

-- CreateIndex
CREATE INDEX "charges_patientId_idx" ON "charges"("patientId");

-- CreateIndex
CREATE INDEX "charges_encounterId_idx" ON "charges"("encounterId");

-- CreateIndex
CREATE INDEX "charges_status_idx" ON "charges"("status");

-- CreateIndex
CREATE INDEX "charges_dateOfService_idx" ON "charges"("dateOfService");

-- CreateIndex
CREATE INDEX "payments_patientId_idx" ON "payments"("patientId");

-- CreateIndex
CREATE INDEX "payments_paymentDate_idx" ON "payments"("paymentDate");

-- CreateIndex
CREATE INDEX "payment_allocations_paymentId_idx" ON "payment_allocations"("paymentId");

-- CreateIndex
CREATE INDEX "payment_allocations_chargeId_idx" ON "payment_allocations"("chargeId");

-- CreateIndex
CREATE UNIQUE INDEX "claims_claimNumber_key" ON "claims"("claimNumber");

-- CreateIndex
CREATE INDEX "claims_patientId_idx" ON "claims"("patientId");

-- CreateIndex
CREATE INDEX "claims_status_idx" ON "claims"("status");

-- CreateIndex
CREATE INDEX "claims_claimNumber_idx" ON "claims"("claimNumber");

-- CreateIndex
CREATE INDEX "claim_lines_claimId_idx" ON "claim_lines"("claimId");

-- CreateIndex
CREATE INDEX "patient_insurance_patientId_idx" ON "patient_insurance"("patientId");

-- CreateIndex
CREATE INDEX "recurring_appointments_patientId_idx" ON "recurring_appointments"("patientId");

-- AddForeignKey
ALTER TABLE "patients" ADD CONSTRAINT "patients_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "clinicians" ADD CONSTRAINT "clinicians_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "conditions" ADD CONSTRAINT "conditions_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ai_audit_log" ADD CONSTRAINT "ai_audit_log_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Medication" ADD CONSTRAINT "Medication_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Medication" ADD CONSTRAINT "Medication_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "medication_logs" ADD CONSTRAINT "medication_logs_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "Medication"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "medication_side_effects" ADD CONSTRAINT "medication_side_effects_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "Medication"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "medication_adherence_logs" ADD CONSTRAINT "medication_adherence_logs_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "Medication"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "reminder_chains" ADD CONSTRAINT "reminder_chains_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "patient_profiles" ADD CONSTRAINT "patient_profiles_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "patient_profiles" ADD CONSTRAINT "patient_profiles_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "diary_entries" ADD CONSTRAINT "diary_entries_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "custom_attributes" ADD CONSTRAINT "custom_attributes_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "pill_recognitions" ADD CONSTRAINT "pill_recognitions_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "health_reports" ADD CONSTRAINT "health_reports_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "data_exports" ADD CONSTRAINT "data_exports_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Metric" ADD CONSTRAINT "Metric_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MedicationCycle" ADD CONSTRAINT "MedicationCycle_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MetricLog" ADD CONSTRAINT "MetricLog_cycleId_fkey" FOREIGN KEY ("cycleId") REFERENCES "MedicationCycle"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DoseLog" ADD CONSTRAINT "DoseLog_cycleId_fkey" FOREIGN KEY ("cycleId") REFERENCES "MedicationCycle"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Notification" ADD CONSTRAINT "Notification_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Notification" ADD CONSTRAINT "Notification_cycleId_fkey" FOREIGN KEY ("cycleId") REFERENCES "MedicationCycle"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "user_survey_data" ADD CONSTRAINT "user_survey_data_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "medical_notes" ADD CONSTRAINT "medical_notes_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "lab_results" ADD CONSTRAINT "lab_results_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "vital_signs" ADD CONSTRAINT "vital_signs_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "vital_signs" ADD CONSTRAINT "vital_signs_encounterId_fkey" FOREIGN KEY ("encounterId") REFERENCES "encounters"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "patient_medications" ADD CONSTRAINT "patient_medications_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "metric_trends" ADD CONSTRAINT "metric_trends_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "appointments" ADD CONSTRAINT "appointments_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "appointments" ADD CONSTRAINT "appointments_providerId_fkey" FOREIGN KEY ("providerId") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "encounters" ADD CONSTRAINT "encounters_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "encounters" ADD CONSTRAINT "encounters_providerId_fkey" FOREIGN KEY ("providerId") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "soap_notes" ADD CONSTRAINT "soap_notes_encounterId_fkey" FOREIGN KEY ("encounterId") REFERENCES "encounters"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "problems" ADD CONSTRAINT "problems_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "problems" ADD CONSTRAINT "problems_encounterId_fkey" FOREIGN KEY ("encounterId") REFERENCES "encounters"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "allergies" ADD CONSTRAINT "allergies_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "immunizations" ADD CONSTRAINT "immunizations_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "documents" ADD CONSTRAINT "documents_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "documents" ADD CONSTRAINT "documents_encounterId_fkey" FOREIGN KEY ("encounterId") REFERENCES "encounters"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "prescriptions" ADD CONSTRAINT "prescriptions_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "charges" ADD CONSTRAINT "charges_encounterId_fkey" FOREIGN KEY ("encounterId") REFERENCES "encounters"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "charges" ADD CONSTRAINT "charges_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "payments" ADD CONSTRAINT "payments_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "payment_allocations" ADD CONSTRAINT "payment_allocations_paymentId_fkey" FOREIGN KEY ("paymentId") REFERENCES "payments"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "payment_allocations" ADD CONSTRAINT "payment_allocations_chargeId_fkey" FOREIGN KEY ("chargeId") REFERENCES "charges"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "claims" ADD CONSTRAINT "claims_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "claim_lines" ADD CONSTRAINT "claim_lines_claimId_fkey" FOREIGN KEY ("claimId") REFERENCES "claims"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "patient_insurance" ADD CONSTRAINT "patient_insurance_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients"("id") ON DELETE CASCADE ON UPDATE CASCADE;


-- CreateEnum
CREATE TYPE "UserRole" AS ENUM ('patient', 'clinician');

-- CreateEnum
CREATE TYPE "MetricType" AS ENUM ('blood_pressure', 'heart_rate', 'blood_glucose', 'weight', 'temperature', 'pain_level', 'sleep_quality', 'mood', 'energy_level', 'side_effects', 'blood_sugar', 'cholesterol', 'blood_oxygen', 'general_health', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'bmi', 'waist_circumference', 'hip_circumference', 'body_fat_percentage', 'muscle_mass', 'bone_density', 'vitamin_d_level', 'iron_level', 'thyroid_function', 'kidney_function', 'liver_function', 'blood_count', 'inflammation_markers', 'allergy_symptoms', 'digestive_health', 'mental_health', 'cognitive_function', 'physical_activity', 'exercise_duration', 'exercise_intensity', 'steps_count', 'calories_burned', 'water_intake', 'alcohol_consumption', 'caffeine_intake', 'smoking_status', 'stress_level', 'anxiety_level', 'depression_score', 'quality_of_life', 'medication_adherence', 'drug_interactions', 'allergic_reactions', 'emergency_symptoms');

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "role" "UserRole" NOT NULL DEFAULT 'patient',
    "hospitalCode" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Patient" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "hospitalCode" TEXT NOT NULL,
    "patientAuditId" TEXT,
    "imdDecile" INTEGER,
    "name" TEXT,
    "dob" TIMESTAMP(3),
    "sex" TEXT,
    "ethnicGroup" TEXT,
    "location" TEXT,
    "postcode" TEXT,
    "height" DECIMAL(65,30),
    "weight" DECIMAL(65,30),
    "bmi" DECIMAL(65,30),
    "baselineHbA1c" DECIMAL(65,30),
    "baselineHbA1cDate" TIMESTAMP(3),
    "t2dm" BOOLEAN,
    "ascvd" BOOLEAN,
    "htn" BOOLEAN,
    "mes" INTEGER,
    "conditions" TEXT[],
    "nhsNumber" TEXT,
    "mrn" TEXT,
    "scrDiagnoses" TEXT[],
    "comorbidities" TEXT[],
    "medications" TEXT[],
    "baselineLabs" TEXT[],
    "notes" TEXT,
    "systolicBp" DECIMAL(65,30),
    "diastolicBp" DECIMAL(65,30),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Patient_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MedicalNote" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "dateOfEntry" TIMESTAMP(3) NOT NULL,
    "impression" TEXT,
    "plan" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MedicalNote_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LabResult" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "testName" TEXT NOT NULL,
    "value" DECIMAL(65,30),
    "unit" TEXT,
    "referenceRange" TEXT,
    "date" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LabResult_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "VitalSign" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "value" DECIMAL(65,30),
    "unit" TEXT,
    "date" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "VitalSign_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PatientMedication" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "medicationName" TEXT NOT NULL,
    "dosage" TEXT,
    "frequency" TEXT,
    "startDate" TIMESTAMP(3),
    "endDate" TIMESTAMP(3),
    "isActive" BOOLEAN DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "PatientMedication_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MetricTrend" (
    "id" TEXT NOT NULL,
    "patientId" TEXT NOT NULL,
    "metricType" "MetricType" NOT NULL,
    "value" DECIMAL(65,30) NOT NULL,
    "unit" TEXT,
    "date" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MetricTrend_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AIProcessingConfig" (
    "id" TEXT NOT NULL,
    "modelName" TEXT NOT NULL,
    "promptTemplate" TEXT NOT NULL,
    "temperature" DECIMAL(65,30) DEFAULT 0.1,
    "maxTokens" INTEGER DEFAULT 1000,
    "isActive" BOOLEAN DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "AIProcessingConfig_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- CreateIndex
CREATE INDEX "Patient_hospitalCode_idx" ON "Patient"("hospitalCode");

-- CreateIndex
CREATE INDEX "Patient_userId_idx" ON "Patient"("userId");

-- CreateIndex
CREATE INDEX "MedicalNote_patientId_idx" ON "MedicalNote"("patientId");

-- CreateIndex
CREATE INDEX "LabResult_patientId_idx" ON "LabResult"("patientId");

-- CreateIndex
CREATE INDEX "VitalSign_patientId_idx" ON "VitalSign"("patientId");

-- CreateIndex
CREATE INDEX "PatientMedication_patientId_idx" ON "PatientMedication"("patientId");

-- CreateIndex
CREATE INDEX "MetricTrend_patientId_idx" ON "MetricTrend"("patientId");

-- AddForeignKey
ALTER TABLE "Patient" ADD CONSTRAINT "Patient_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MedicalNote" ADD CONSTRAINT "MedicalNote_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "Patient"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LabResult" ADD CONSTRAINT "LabResult_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "Patient"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "VitalSign" ADD CONSTRAINT "VitalSign_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "Patient"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PatientMedication" ADD CONSTRAINT "PatientMedication_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "Patient"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MetricTrend" ADD CONSTRAINT "MetricTrend_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "Patient"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

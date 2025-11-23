-- Rename tables to match Prisma schema @@map directives
ALTER TABLE "User" RENAME TO "users";
ALTER TABLE "Patient" RENAME TO "patients";
ALTER TABLE "MedicalNote" RENAME TO "medical_notes";
ALTER TABLE "LabResult" RENAME TO "lab_results";
ALTER TABLE "VitalSign" RENAME TO "vital_signs";
ALTER TABLE "PatientMedication" RENAME TO "patient_medications";
ALTER TABLE "MetricTrend" RENAME TO "metric_trends";
ALTER TABLE "AIProcessingConfig" RENAME TO "ai_processing_configs";

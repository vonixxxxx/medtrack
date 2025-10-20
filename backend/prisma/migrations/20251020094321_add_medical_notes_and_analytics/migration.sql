/*
  Warnings:

  - You are about to drop the `patient_metrics` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the column `createdAt` on the `medical_notes` table. All the data in the column will be lost.
  - You are about to drop the column `date_of_entry` on the `medical_notes` table. All the data in the column will be lost.
  - You are about to drop the column `processing_status` on the `medical_notes` table. All the data in the column will be lost.
  - You are about to drop the column `updatedAt` on the `medical_notes` table. All the data in the column will be lost.
  - You are about to drop the column `createdAt` on the `patient_medications` table. All the data in the column will be lost.
  - You are about to drop the column `is_ai_generated` on the `patient_medications` table. All the data in the column will be lost.
  - You are about to drop the column `updatedAt` on the `patient_medications` table. All the data in the column will be lost.
  - Added the required column `updated_at` to the `medical_notes` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updated_at` to the `patient_medications` table without a default value. This is not possible if the table is not empty.
  - Made the column `dosage` on table `patient_medications` required. This step will fail if there are existing NULL values in that column.
  - Made the column `frequency` on table `patient_medications` required. This step will fail if there are existing NULL values in that column.
  - Made the column `start_date` on table `patient_medications` required. This step will fail if there are existing NULL values in that column.

*/
-- DropIndex
DROP INDEX "patient_metrics_date_recorded_idx";

-- DropIndex
DROP INDEX "patient_metrics_metric_name_idx";

-- DropIndex
DROP INDEX "patient_metrics_patientId_idx";

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "patient_metrics";
PRAGMA foreign_keys=on;

-- CreateTable
CREATE TABLE "lab_results" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "metric_name" TEXT NOT NULL,
    "value" REAL NOT NULL,
    "unit" TEXT NOT NULL,
    "date" DATETIME NOT NULL,
    "reference_range" TEXT,
    "status" TEXT,
    "source_note_id" TEXT,
    "manually_entered" BOOLEAN NOT NULL DEFAULT false,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "lab_results_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "vital_signs" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "vital_type" TEXT NOT NULL,
    "value" REAL NOT NULL,
    "unit" TEXT NOT NULL,
    "date" DATETIME NOT NULL,
    "value_secondary" REAL,
    "source_note_id" TEXT,
    "manually_entered" BOOLEAN NOT NULL DEFAULT false,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "vital_signs_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "metric_trends" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "metric_name" TEXT NOT NULL,
    "value" REAL NOT NULL,
    "unit" TEXT NOT NULL,
    "date" DATETIME NOT NULL,
    "change_from_previous" REAL,
    "change_percentage" REAL,
    "trend_direction" TEXT,
    "source_type" TEXT NOT NULL,
    "source_id" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "metric_trends_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "ai_processing_config" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "model_name" TEXT NOT NULL,
    "model_provider" TEXT NOT NULL,
    "system_prompt" TEXT NOT NULL,
    "max_tokens" INTEGER NOT NULL DEFAULT 2048,
    "temperature" REAL NOT NULL DEFAULT 0.1,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL
);

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_medical_notes" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "raw_text" TEXT NOT NULL,
    "note_type" TEXT,
    "source" TEXT,
    "ai_processed" BOOLEAN NOT NULL DEFAULT false,
    "ai_confidence" REAL,
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
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "medical_notes_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_medical_notes" ("age", "ai_confidence", "ai_model_used", "allergies", "conditions", "id", "impression", "lab_results", "medications", "patientId", "patient_name", "plan", "raw_text", "sex", "source", "vital_signs") SELECT "age", "ai_confidence", "ai_model_used", "allergies", "conditions", "id", "impression", "lab_results", "medications", "patientId", "patient_name", "plan", "raw_text", "sex", "source", "vital_signs" FROM "medical_notes";
DROP TABLE "medical_notes";
ALTER TABLE "new_medical_notes" RENAME TO "medical_notes";
CREATE INDEX "medical_notes_patientId_idx" ON "medical_notes"("patientId");
CREATE INDEX "medical_notes_created_at_idx" ON "medical_notes"("created_at");
CREATE INDEX "medical_notes_ai_processed_idx" ON "medical_notes"("ai_processed");
CREATE TABLE "new_patient_medications" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "dosage" TEXT NOT NULL,
    "frequency" TEXT NOT NULL,
    "route" TEXT,
    "start_date" DATETIME NOT NULL,
    "end_date" DATETIME,
    "status" TEXT NOT NULL DEFAULT 'active',
    "source_note_id" TEXT,
    "manually_entered" BOOLEAN NOT NULL DEFAULT false,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "patient_medications_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_patient_medications" ("dosage", "end_date", "frequency", "id", "name", "patientId", "route", "source_note_id", "start_date", "status") SELECT "dosage", "end_date", "frequency", "id", "name", "patientId", "route", "source_note_id", "start_date", "status" FROM "patient_medications";
DROP TABLE "patient_medications";
ALTER TABLE "new_patient_medications" RENAME TO "patient_medications";
CREATE INDEX "patient_medications_patientId_idx" ON "patient_medications"("patientId");
CREATE INDEX "patient_medications_status_idx" ON "patient_medications"("status");
CREATE INDEX "patient_medications_start_date_idx" ON "patient_medications"("start_date");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

-- CreateIndex
CREATE INDEX "lab_results_patientId_idx" ON "lab_results"("patientId");

-- CreateIndex
CREATE INDEX "lab_results_metric_name_idx" ON "lab_results"("metric_name");

-- CreateIndex
CREATE INDEX "lab_results_date_idx" ON "lab_results"("date");

-- CreateIndex
CREATE INDEX "vital_signs_patientId_idx" ON "vital_signs"("patientId");

-- CreateIndex
CREATE INDEX "vital_signs_vital_type_idx" ON "vital_signs"("vital_type");

-- CreateIndex
CREATE INDEX "vital_signs_date_idx" ON "vital_signs"("date");

-- CreateIndex
CREATE INDEX "metric_trends_patientId_idx" ON "metric_trends"("patientId");

-- CreateIndex
CREATE INDEX "metric_trends_metric_name_idx" ON "metric_trends"("metric_name");

-- CreateIndex
CREATE INDEX "metric_trends_date_idx" ON "metric_trends"("date");

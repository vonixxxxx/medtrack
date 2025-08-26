/*
  Warnings:

  - You are about to drop the `DoseLog` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `MetricLog` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `screening_results` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the column `dosage` on the `MedicationCycle` table. All the data in the column will be lost.
  - You are about to drop the column `dosesPerDay` on the `MedicationCycle` table. All the data in the column will be lost.
  - You are about to drop the column `frequencyDays` on the `MedicationCycle` table. All the data in the column will be lost.
  - You are about to drop the column `metricsToMonitor` on the `MedicationCycle` table. All the data in the column will be lost.
  - You are about to drop the column `name` on the `MedicationCycle` table. All the data in the column will be lost.
  - You are about to drop the column `date` on the `MedicationLog` table. All the data in the column will be lost.
  - You are about to drop the column `dosage` on the `MedicationLog` table. All the data in the column will be lost.
  - You are about to drop the column `bloodPressure` on the `Metric` table. All the data in the column will be lost.
  - You are about to drop the column `bmi` on the `Metric` table. All the data in the column will be lost.
  - You are about to drop the column `date` on the `Metric` table. All the data in the column will be lost.
  - You are about to drop the column `height` on the `Metric` table. All the data in the column will be lost.
  - You are about to drop the column `hipCircumference` on the `Metric` table. All the data in the column will be lost.
  - You are about to drop the column `weight` on the `Metric` table. All the data in the column will be lost.
  - You are about to drop the column `cycleId` on the `Notification` table. All the data in the column will be lost.
  - You are about to drop the column `date` on the `Notification` table. All the data in the column will be lost.
  - You are about to drop the column `sent` on the `Notification` table. All the data in the column will be lost.
  - Added the required column `medicationId` to the `MedicationCycle` table without a default value. This is not possible if the table is not empty.
  - Added the required column `name` to the `Metric` table without a default value. This is not possible if the table is not empty.
  - Added the required column `unit` to the `Metric` table without a default value. This is not possible if the table is not empty.
  - Added the required column `value` to the `Metric` table without a default value. This is not possible if the table is not empty.
  - Added the required column `title` to the `Notification` table without a default value. This is not possible if the table is not empty.
  - Added the required column `type` to the `Notification` table without a default value. This is not possible if the table is not empty.

*/
-- DropIndex
DROP INDEX "DoseLog_cycleId_date_key";

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "DoseLog";
PRAGMA foreign_keys=on;

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "MetricLog";
PRAGMA foreign_keys=on;

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "screening_results";
PRAGMA foreign_keys=on;

-- CreateTable
CREATE TABLE "medication_validations" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "genericName" TEXT NOT NULL,
    "atcClass" TEXT,
    "synonyms" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "medication_products" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "medicationId" TEXT NOT NULL,
    "brandName" TEXT NOT NULL,
    "route" TEXT NOT NULL,
    "form" TEXT NOT NULL,
    "allowedIntakeType" TEXT NOT NULL,
    "defaultPlace" TEXT NOT NULL,
    "allowedFrequencies" TEXT NOT NULL,
    "notes" TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    CONSTRAINT "medication_products_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "medication_validations" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "medication_strengths" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "productId" TEXT NOT NULL,
    "strengthValue" REAL NOT NULL,
    "strengthUnit" TEXT NOT NULL,
    "frequency" TEXT NOT NULL,
    "label" TEXT NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    CONSTRAINT "medication_strengths_productId_fkey" FOREIGN KEY ("productId") REFERENCES "medication_products" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "medication_validation_rules" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "productId" TEXT NOT NULL,
    "maxDosePerPeriod" TEXT,
    "minDosePerPeriod" TEXT,
    "contraindications" TEXT,
    "warnings" TEXT,
    "version" INTEGER NOT NULL DEFAULT 1,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "medication_validation_rules_productId_fkey" FOREIGN KEY ("productId") REFERENCES "medication_products" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "user_medication_cycles" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" INTEGER NOT NULL,
    "medicationId" TEXT NOT NULL,
    "productId" TEXT NOT NULL,
    "strengthValue" REAL NOT NULL,
    "strengthUnit" TEXT NOT NULL,
    "frequency" TEXT NOT NULL,
    "intakeType" TEXT NOT NULL,
    "intakePlace" TEXT NOT NULL,
    "startDate" DATETIME NOT NULL,
    "endDate" DATETIME,
    "customFlags" TEXT NOT NULL,
    "notes" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "user_medication_cycles_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "user_medication_cycles_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "medication_validations" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "user_medication_cycles_productId_fkey" FOREIGN KEY ("productId") REFERENCES "medication_products" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "ScreeningResult" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "type" TEXT NOT NULL,
    "score" INTEGER,
    "result" TEXT NOT NULL,
    "details" TEXT,
    "completedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "ScreeningResult_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;

-- Handle MedicationCycle table
CREATE TABLE "new_MedicationCycle" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "medicationId" INTEGER NOT NULL DEFAULT 1,
    "startDate" DATETIME NOT NULL,
    "endDate" DATETIME,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    CONSTRAINT "MedicationCycle_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "MedicationCycle_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "Medication" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_MedicationCycle" ("endDate", "id", "startDate", "userId") SELECT "endDate", "id", "startDate", "userId" FROM "MedicationCycle";
DROP TABLE "MedicationCycle";
ALTER TABLE "new_MedicationCycle" RENAME TO "MedicationCycle";

-- Handle MedicationLog table
CREATE TABLE "new_MedicationLog" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "medicationId" INTEGER NOT NULL,
    "takenAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "notes" TEXT,
    CONSTRAINT "MedicationLog_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "Medication" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_MedicationLog" ("id", "medicationId") SELECT "id", "medicationId" FROM "MedicationLog";
DROP TABLE "MedicationLog";
ALTER TABLE "new_MedicationLog" RENAME TO "MedicationLog";

-- Handle Metric table - preserve existing data
CREATE TABLE "new_Metric" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "name" TEXT NOT NULL DEFAULT 'Weight',
    "value" REAL NOT NULL DEFAULT 0,
    "unit" TEXT NOT NULL DEFAULT 'kg',
    "recordedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "notes" TEXT,
    CONSTRAINT "Metric_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_Metric" ("id", "userId", "name", "value", "unit", "recordedAt", "notes") 
SELECT "id", "userId", 
       CASE 
         WHEN "weight" IS NOT NULL THEN 'Weight'
         WHEN "height" IS NOT NULL THEN 'Height'
         WHEN "bmi" IS NOT NULL THEN 'BMI'
         WHEN "bloodPressure" IS NOT NULL THEN 'Blood Pressure'
         WHEN "hipCircumference" IS NOT NULL THEN 'Hip Circumference'
         ELSE 'Metric'
       END,
       COALESCE("weight", "height", "bmi", "hipCircumference", 0),
       CASE 
         WHEN "weight" IS NOT NULL THEN 'kg'
         WHEN "height" IS NOT NULL THEN 'm'
         WHEN "bmi" IS NOT NULL THEN 'kg/m²'
         WHEN "bloodPressure" IS NOT NULL THEN 'mmHg'
         WHEN "hipCircumference" IS NOT NULL THEN 'cm'
         ELSE 'unit'
       END,
       COALESCE("date", CURRENT_TIMESTAMP),
       NULL
FROM "Metric";
DROP TABLE "Metric";
ALTER TABLE "new_Metric" RENAME TO "Metric";

-- Handle Notification table - preserve existing data
CREATE TABLE "new_Notification" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "type" TEXT NOT NULL DEFAULT 'general',
    "title" TEXT NOT NULL DEFAULT 'Notification',
    "message" TEXT NOT NULL,
    "isRead" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Notification_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_Notification" ("id", "message", "userId", "type", "title") 
SELECT "id", "message", "userId", 'general', 'Notification' FROM "Notification";
DROP TABLE "Notification";
ALTER TABLE "new_Notification" RENAME TO "Notification";

PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

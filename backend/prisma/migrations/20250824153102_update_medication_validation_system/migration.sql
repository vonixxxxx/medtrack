/*
  Warnings:

  - You are about to drop the `medication_validations` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `users` table. If the table is not empty, all the data it contains will be lost.
  - The primary key for the `Medication` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `MedicationCycle` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `MedicationLog` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `Metric` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to drop the column `notes` on the `Metric` table. All the data in the column will be lost.
  - You are about to drop the column `recordedAt` on the `Metric` table. All the data in the column will be lost.
  - The primary key for the `Notification` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The primary key for the `ScreeningResult` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to drop the column `completedAt` on the `ScreeningResult` table. All the data in the column will be lost.
  - You are about to drop the column `details` on the `ScreeningResult` table. All the data in the column will be lost.
  - You are about to drop the column `allowedFrequencies` on the `medication_products` table. All the data in the column will be lost.
  - You are about to drop the column `allowedIntakeType` on the `medication_products` table. All the data in the column will be lost.
  - You are about to drop the column `brandName` on the `medication_products` table. All the data in the column will be lost.
  - You are about to drop the column `defaultPlace` on the `medication_products` table. All the data in the column will be lost.
  - You are about to drop the column `isActive` on the `medication_products` table. All the data in the column will be lost.
  - You are about to drop the column `medicationId` on the `medication_products` table. All the data in the column will be lost.
  - You are about to drop the column `isActive` on the `medication_strengths` table. All the data in the column will be lost.
  - You are about to drop the column `productId` on the `medication_strengths` table. All the data in the column will be lost.
  - You are about to drop the column `strengthUnit` on the `medication_strengths` table. All the data in the column will be lost.
  - You are about to drop the column `strengthValue` on the `medication_strengths` table. All the data in the column will be lost.
  - You are about to drop the column `createdAt` on the `medication_validation_rules` table. All the data in the column will be lost.
  - You are about to drop the column `maxDosePerPeriod` on the `medication_validation_rules` table. All the data in the column will be lost.
  - You are about to drop the column `minDosePerPeriod` on the `medication_validation_rules` table. All the data in the column will be lost.
  - You are about to drop the column `productId` on the `medication_validation_rules` table. All the data in the column will be lost.
  - You are about to drop the column `createdAt` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `customFlags` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `endDate` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `intakePlace` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `intakeType` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `medicationId` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `productId` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `startDate` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `strengthUnit` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `strengthValue` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `updatedAt` on the `user_medication_cycles` table. All the data in the column will be lost.
  - You are about to drop the column `userId` on the `user_medication_cycles` table. All the data in the column will be lost.
  - Added the required column `updatedAt` to the `Medication` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updatedAt` to the `MedicationCycle` table without a default value. This is not possible if the table is not empty.
  - Added the required column `userId` to the `MedicationLog` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updatedAt` to the `Metric` table without a default value. This is not possible if the table is not empty.
  - Made the column `score` on table `ScreeningResult` required. This step will fail if there are existing NULL values in that column.
  - Added the required column `allowed_intake_type` to the `medication_products` table without a default value. This is not possible if the table is not empty.
  - Added the required column `brand_name` to the `medication_products` table without a default value. This is not possible if the table is not empty.
  - Added the required column `medication_id` to the `medication_products` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updated_at` to the `medication_products` table without a default value. This is not possible if the table is not empty.
  - Added the required column `product_id` to the `medication_strengths` table without a default value. This is not possible if the table is not empty.
  - Added the required column `strength_unit` to the `medication_strengths` table without a default value. This is not possible if the table is not empty.
  - Added the required column `strength_value` to the `medication_strengths` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updated_at` to the `medication_strengths` table without a default value. This is not possible if the table is not empty.
  - Added the required column `product_id` to the `medication_validation_rules` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updated_at` to the `medication_validation_rules` table without a default value. This is not possible if the table is not empty.
  - Added the required column `intake_place` to the `user_medication_cycles` table without a default value. This is not possible if the table is not empty.
  - Added the required column `intake_type` to the `user_medication_cycles` table without a default value. This is not possible if the table is not empty.
  - Added the required column `medication_id` to the `user_medication_cycles` table without a default value. This is not possible if the table is not empty.
  - Added the required column `product_id` to the `user_medication_cycles` table without a default value. This is not possible if the table is not empty.
  - Added the required column `start_date` to the `user_medication_cycles` table without a default value. This is not possible if the table is not empty.
  - Added the required column `strength_unit` to the `user_medication_cycles` table without a default value. This is not possible if the table is not empty.
  - Added the required column `strength_value` to the `user_medication_cycles` table without a default value. This is not possible if the table is not empty.
  - Added the required column `updated_at` to the `user_medication_cycles` table without a default value. This is not possible if the table is not empty.
  - Added the required column `user_id` to the `user_medication_cycles` table without a default value. This is not possible if the table is not empty.

*/
-- DropIndex
DROP INDEX "users_email_key";

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "medication_validations";
PRAGMA foreign_keys=on;

-- DropTable
PRAGMA foreign_keys=off;
DROP TABLE "users";
PRAGMA foreign_keys=on;

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "medications" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "generic_name" TEXT NOT NULL,
    "atc_class" TEXT,
    "class_human" TEXT,
    "synonyms" TEXT NOT NULL DEFAULT '[]',
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL
);

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_Medication" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "dosage" TEXT NOT NULL,
    "frequency" TEXT NOT NULL,
    "startDate" DATETIME NOT NULL,
    "endDate" DATETIME,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "userId" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "Medication_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_Medication" ("dosage", "endDate", "frequency", "id", "name", "startDate", "userId") SELECT "dosage", "endDate", "frequency", "id", "name", "startDate", "userId" FROM "Medication";
DROP TABLE "Medication";
ALTER TABLE "new_Medication" RENAME TO "Medication";
CREATE TABLE "new_MedicationCycle" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "medicationId" TEXT NOT NULL,
    "startDate" DATETIME NOT NULL,
    "endDate" DATETIME,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "userId" TEXT NOT NULL,
    CONSTRAINT "MedicationCycle_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "Medication" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "MedicationCycle_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_MedicationCycle" ("endDate", "id", "isActive", "medicationId", "startDate", "userId") SELECT "endDate", "id", "isActive", "medicationId", "startDate", "userId" FROM "MedicationCycle";
DROP TABLE "MedicationCycle";
ALTER TABLE "new_MedicationCycle" RENAME TO "MedicationCycle";
CREATE TABLE "new_MedicationLog" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "medicationId" TEXT NOT NULL,
    "takenAt" DATETIME NOT NULL,
    "notes" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    CONSTRAINT "MedicationLog_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "Medication" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "MedicationLog_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_MedicationLog" ("id", "medicationId", "notes", "takenAt") SELECT "id", "medicationId", "notes", "takenAt" FROM "MedicationLog";
DROP TABLE "MedicationLog";
ALTER TABLE "new_MedicationLog" RENAME TO "MedicationLog";
CREATE TABLE "new_Metric" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL DEFAULT '',
    "value" REAL NOT NULL DEFAULT 0,
    "unit" TEXT NOT NULL DEFAULT '',
    "userId" TEXT NOT NULL,
    "date" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "Metric_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_Metric" ("id", "name", "unit", "userId", "value") SELECT "id", "name", "unit", "userId", "value" FROM "Metric";
DROP TABLE "Metric";
ALTER TABLE "new_Metric" RENAME TO "Metric";
CREATE TABLE "new_Notification" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "type" TEXT NOT NULL DEFAULT 'reminder',
    "title" TEXT NOT NULL DEFAULT '',
    "message" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "isRead" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Notification_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_Notification" ("createdAt", "id", "isRead", "message", "title", "type", "userId") SELECT "createdAt", "id", "isRead", "message", "title", "type", "userId" FROM "Notification";
DROP TABLE "Notification";
ALTER TABLE "new_Notification" RENAME TO "Notification";
CREATE TABLE "new_ScreeningResult" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "type" TEXT NOT NULL,
    "score" INTEGER NOT NULL,
    "result" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "ScreeningResult_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_ScreeningResult" ("id", "result", "score", "type", "userId") SELECT "id", "result", "score", "type", "userId" FROM "ScreeningResult";
DROP TABLE "ScreeningResult";
ALTER TABLE "new_ScreeningResult" RENAME TO "ScreeningResult";
CREATE TABLE "new_medication_products" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "medication_id" TEXT NOT NULL,
    "brand_name" TEXT NOT NULL,
    "route" TEXT NOT NULL,
    "form" TEXT NOT NULL,
    "allowed_intake_type" TEXT NOT NULL,
    "default_places" TEXT NOT NULL DEFAULT '[]',
    "allowed_frequencies" TEXT NOT NULL DEFAULT '[]',
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "notes" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "medication_products_medication_id_fkey" FOREIGN KEY ("medication_id") REFERENCES "medications" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_medication_products" ("form", "id", "notes", "route") SELECT "form", "id", "notes", "route" FROM "medication_products";
DROP TABLE "medication_products";
ALTER TABLE "new_medication_products" RENAME TO "medication_products";
CREATE TABLE "new_medication_strengths" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "product_id" TEXT NOT NULL,
    "strength_value" REAL NOT NULL,
    "strength_unit" TEXT NOT NULL,
    "frequency" TEXT NOT NULL,
    "label" TEXT,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "medication_strengths_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "medication_products" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_medication_strengths" ("frequency", "id", "label") SELECT "frequency", "id", "label" FROM "medication_strengths";
DROP TABLE "medication_strengths";
ALTER TABLE "new_medication_strengths" RENAME TO "medication_strengths";
CREATE TABLE "new_medication_validation_rules" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "product_id" TEXT NOT NULL,
    "max_dose_per_period" TEXT,
    "min_dose_per_period" TEXT,
    "contraindications" TEXT,
    "warnings" TEXT,
    "version" INTEGER NOT NULL DEFAULT 1,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "medication_validation_rules_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "medication_products" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_medication_validation_rules" ("contraindications", "id", "version", "warnings") SELECT "contraindications", "id", "version", "warnings" FROM "medication_validation_rules";
DROP TABLE "medication_validation_rules";
ALTER TABLE "new_medication_validation_rules" RENAME TO "medication_validation_rules";
CREATE TABLE "new_user_medication_cycles" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "user_id" TEXT NOT NULL,
    "medication_id" TEXT NOT NULL,
    "product_id" TEXT NOT NULL,
    "strength_value" REAL NOT NULL,
    "strength_unit" TEXT NOT NULL,
    "frequency" TEXT NOT NULL,
    "intake_type" TEXT NOT NULL,
    "intake_place" TEXT NOT NULL,
    "start_date" DATETIME NOT NULL,
    "end_date" DATETIME,
    "custom_flags" TEXT NOT NULL DEFAULT '{}',
    "notes" TEXT,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "user_medication_cycles_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "user_medication_cycles_medication_id_fkey" FOREIGN KEY ("medication_id") REFERENCES "medications" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "user_medication_cycles_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "medication_products" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_user_medication_cycles" ("frequency", "id", "notes") SELECT "frequency", "id", "notes" FROM "user_medication_cycles";
DROP TABLE "user_medication_cycles";
ALTER TABLE "new_user_medication_cycles" RENAME TO "user_medication_cycles";
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- CreateIndex
CREATE UNIQUE INDEX "medications_generic_name_key" ON "medications"("generic_name");

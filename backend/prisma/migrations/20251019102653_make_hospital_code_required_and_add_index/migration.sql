/*
  Warnings:

  - Made the column `hospitalCode` on table `users` required. This step will fail if there are existing NULL values in that column.

*/
-- First, update existing NULL hospitalCode values to a default value
UPDATE "users" SET "hospitalCode" = '123456789' WHERE "hospitalCode" IS NULL;

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_users" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "role" TEXT NOT NULL DEFAULT 'patient',
    "hospitalCode" TEXT NOT NULL,
    "resetToken" TEXT,
    "resetTokenExpiry" DATETIME,
    "is2FAEnabled" BOOLEAN NOT NULL DEFAULT false,
    "twoFASecret" TEXT,
    "surveyCompleted" BOOLEAN NOT NULL DEFAULT false
);
INSERT INTO "new_users" ("email", "hospitalCode", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "role", "surveyCompleted", "twoFASecret") SELECT "email", "hospitalCode", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "role", "surveyCompleted", "twoFASecret" FROM "users";
DROP TABLE "users";
ALTER TABLE "new_users" RENAME TO "users";
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");
CREATE INDEX "users_hospitalCode_idx" ON "users"("hospitalCode");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

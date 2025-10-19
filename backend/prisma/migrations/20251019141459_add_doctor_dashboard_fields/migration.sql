-- CreateTable
CREATE TABLE "patients" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "dob" DATETIME,
    "sex" TEXT,
    "ethnicity" TEXT,
    "hba1c_percent" REAL,
    "hba1c_mmol" REAL,
    "diabetes_type" TEXT,
    "mes" REAL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "patients_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "clinicians" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "hospitalCode" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "clinicians_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "conditions" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "patientId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "normalized" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "conditions_patientId_fkey" FOREIGN KEY ("patientId") REFERENCES "patients" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- Add new columns to users table
ALTER TABLE "users" ADD COLUMN "createdAt" DATETIME;
ALTER TABLE "users" ADD COLUMN "updatedAt" DATETIME;

-- Update existing rows with current timestamp
UPDATE "users" SET "createdAt" = CURRENT_TIMESTAMP, "updatedAt" = CURRENT_TIMESTAMP WHERE "createdAt" IS NULL;

-- CreateIndex
CREATE INDEX "patients_userId_key" ON "patients"("userId");

-- CreateIndex
CREATE INDEX "clinicians_userId_key" ON "clinicians"("userId");

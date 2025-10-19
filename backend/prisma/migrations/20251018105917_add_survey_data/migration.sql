-- CreateTable
CREATE TABLE "user_survey_data" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
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

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_users" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "resetToken" TEXT,
    "resetTokenExpiry" DATETIME,
    "is2FAEnabled" BOOLEAN NOT NULL DEFAULT false,
    "twoFASecret" TEXT,
    "surveyCompleted" BOOLEAN NOT NULL DEFAULT false
);
INSERT INTO "new_users" ("email", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "twoFASecret") SELECT "email", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "twoFASecret" FROM "users";
DROP TABLE "users";
ALTER TABLE "new_users" RENAME TO "users";
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

-- CreateIndex
CREATE UNIQUE INDEX "user_survey_data_userId_key" ON "user_survey_data"("userId");

-- CreateTable
CREATE TABLE "screening_results" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "assessmentType" TEXT NOT NULL,
    "answers" TEXT NOT NULL,
    "totalScore" INTEGER NOT NULL,
    "severity" TEXT NOT NULL,
    "severityColor" TEXT,
    "severityBgColor" TEXT,
    "completedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "metadata" TEXT,
    CONSTRAINT "screening_results_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
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
    "onHRT" BOOLEAN,
    "hrtType" TEXT,
    "iiefScore" INTEGER,
    "lowTestosteroneSymptoms" TEXT,
    "redFlagQuestions" TEXT,
    "auditScore" INTEGER,
    "smokingStatus" TEXT,
    "smokingStartAge" INTEGER,
    "cigarettesPerDay" INTEGER,
    "packYears" REAL,
    "vapingInfo" TEXT,
    "ipaqScore" INTEGER,
    "weight" REAL,
    "height" REAL,
    "waistCircumference" REAL,
    "hipCircumference" REAL,
    "neckCircumference" REAL,
    "bloodPressure" TEXT,
    "bmi" REAL,
    "whr" REAL,
    "whtr" REAL,
    "bri" REAL,
    "isRegistrationComplete" BOOLEAN NOT NULL DEFAULT false
);
INSERT INTO "new_users" ("email", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "twoFASecret") SELECT "email", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "twoFASecret" FROM "users";
DROP TABLE "users";
ALTER TABLE "new_users" RENAME TO "users";
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

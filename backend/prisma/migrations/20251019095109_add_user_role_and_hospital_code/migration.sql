-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_users" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "name" TEXT,
    "role" TEXT NOT NULL DEFAULT 'patient',
    "hospitalCode" TEXT,
    "resetToken" TEXT,
    "resetTokenExpiry" DATETIME,
    "is2FAEnabled" BOOLEAN NOT NULL DEFAULT false,
    "twoFASecret" TEXT,
    "surveyCompleted" BOOLEAN NOT NULL DEFAULT false
);
INSERT INTO "new_users" ("email", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "surveyCompleted", "twoFASecret") SELECT "email", "id", "is2FAEnabled", "name", "password", "resetToken", "resetTokenExpiry", "surveyCompleted", "twoFASecret" FROM "users";
DROP TABLE "users";
ALTER TABLE "new_users" RENAME TO "users";
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

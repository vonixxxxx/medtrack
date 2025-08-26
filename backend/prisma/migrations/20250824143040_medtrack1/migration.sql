-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_MedicationCycle" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "medicationId" INTEGER NOT NULL,
    "startDate" DATETIME NOT NULL,
    "endDate" DATETIME,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    CONSTRAINT "MedicationCycle_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "MedicationCycle_medicationId_fkey" FOREIGN KEY ("medicationId") REFERENCES "Medication" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_MedicationCycle" ("endDate", "id", "isActive", "medicationId", "startDate", "userId") SELECT "endDate", "id", "isActive", "medicationId", "startDate", "userId" FROM "MedicationCycle";
DROP TABLE "MedicationCycle";
ALTER TABLE "new_MedicationCycle" RENAME TO "MedicationCycle";
CREATE TABLE "new_Metric" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    "value" REAL NOT NULL,
    "unit" TEXT NOT NULL,
    "recordedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "notes" TEXT,
    CONSTRAINT "Metric_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_Metric" ("id", "name", "notes", "recordedAt", "unit", "userId", "value") SELECT "id", "name", "notes", "recordedAt", "unit", "userId", "value" FROM "Metric";
DROP TABLE "Metric";
ALTER TABLE "new_Metric" RENAME TO "Metric";
CREATE TABLE "new_Notification" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "userId" INTEGER NOT NULL,
    "type" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "isRead" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Notification_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
INSERT INTO "new_Notification" ("createdAt", "id", "isRead", "message", "title", "type", "userId") SELECT "createdAt", "id", "isRead", "message", "title", "type", "userId" FROM "Notification";
DROP TABLE "Notification";
ALTER TABLE "new_Notification" RENAME TO "Notification";
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

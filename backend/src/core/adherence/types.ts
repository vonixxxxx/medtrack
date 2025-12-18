/**
 * Types for Medication Adherence Engine
 */

export interface MedicationAdherenceData {
  medicationId: string;
  expectedDosesPerDay: number;
  actualDosesTaken: Array<{
    timestamp: Date;
    taken: boolean;
  }>;
  adherencePercentage: number;
  streakCount: number;
  weeklyAdherenceHistory: Array<{
    weekStart: Date;
    weekEnd: Date;
    adherencePercentage: number;
    dosesTaken: number;
    dosesExpected: number;
  }>;
}

export interface AdherenceCalculationOptions {
  startDate?: Date;
  endDate?: Date;
  includeMissedDosePenalty?: boolean;
  penaltyWeight?: number; // 0-1, default 0.1
}

export interface StreakData {
  medicationId: string;
  currentStreak: number;
  longestStreak: number;
  streakStartDate: Date | null;
  lastMissedDate: Date | null;
  consecutiveMissedDays: number;
}

export interface AdherencePattern {
  medicationId: string;
  pattern: 'improving' | 'declining' | 'stable' | 'volatile';
  trend: number; // -1 to 1, negative = declining, positive = improving
  volatility: number; // 0-1, higher = more volatile
  averageAdherence: number;
  recentAdherence: number; // Last 7 days
}








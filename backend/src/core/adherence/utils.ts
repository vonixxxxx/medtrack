/**
 * Utility functions for adherence calculations
 */

import { startOfDay, endOfDay, differenceInDays, isSameDay, addDays, subDays } from 'date-fns';

/**
 * Calculate adherence percentage
 */
export function calculateAdherence(
  dosesTaken: number,
  dosesExpected: number,
  missedDosePenalty: number = 0
): number {
  if (dosesExpected === 0) return 100;
  
  const baseAdherence = (dosesTaken / dosesExpected) * 100;
  const penalty = missedDosePenalty * 100;
  
  return Math.max(0, Math.min(100, baseAdherence - penalty));
}

/**
 * Get all dates in a range
 */
export function getDatesInRange(startDate: Date, endDate: Date): Date[] {
  const dates: Date[] = [];
  let currentDate = startOfDay(startDate);
  const end = startOfDay(endDate);
  
  while (currentDate <= end) {
    dates.push(new Date(currentDate));
    currentDate = addDays(currentDate, 1);
  }
  
  return dates;
}

/**
 * Group doses by day
 */
export function groupDosesByDay(
  doses: Array<{ timestamp: Date; taken: boolean }>
): Map<string, { taken: number; expected: number }> {
  const grouped = new Map<string, { taken: number; expected: number }>();
  
  for (const dose of doses) {
    const dayKey = startOfDay(dose.timestamp).toISOString();
    const existing = grouped.get(dayKey) || { taken: 0, expected: 0 };
    
    existing.expected += 1;
    if (dose.taken) {
      existing.taken += 1;
    }
    
    grouped.set(dayKey, existing);
  }
  
  return grouped;
}

/**
 * Calculate missed dose penalty
 */
export function calculateMissedDosePenalty(
  missedDays: number,
  totalDays: number,
  penaltyWeight: number = 0.1
): number {
  if (totalDays === 0) return 0;
  return (missedDays / totalDays) * penaltyWeight;
}

/**
 * Check if a day is fully adherent (all expected doses taken)
 */
export function isDayFullyAdherent(
  dosesTaken: number,
  dosesExpected: number
): boolean {
  return dosesTaken >= dosesExpected;
}








/**
 * Medication Adherence Engine
 * 
 * Computes adherence percentages, streaks, and patterns
 * Inspired by MyTherapy/Medisafe adherence models
 */

import { PrismaClient } from '@prisma/client';
import { startOfDay, endOfDay, subDays, addDays, differenceInDays, startOfWeek, endOfWeek } from 'date-fns';
import {
  MedicationAdherenceData,
  AdherenceCalculationOptions,
  StreakData,
  AdherencePattern
} from './types';
import {
  calculateAdherence,
  getDatesInRange,
  groupDosesByDay,
  calculateMissedDosePenalty,
  isDayFullyAdherent
} from './utils';

export class AdherenceEngine {
  constructor(private prisma: PrismaClient) {}

  /**
   * Get daily adherence for a medication
   */
  async getDailyAdherence(
    medicationId: string,
    userId: string,
    date?: Date
  ): Promise<MedicationAdherenceData> {
    const targetDate = date ? startOfDay(date) : startOfDay(new Date());
    const endDate = endOfDay(targetDate);

    // Get medication cycle
    const cycle = await this.prisma.medicationCycle.findFirst({
      where: {
        id: medicationId,
        userId: userId
      },
      include: {
        doseLogs: {
          where: {
            date: {
              gte: targetDate,
              lte: endDate
            }
          }
        }
      }
    });

    if (!cycle) {
      throw new Error('Medication cycle not found');
    }

    const expectedDoses = cycle.dosesPerDay;
    const actualDoses = cycle.doseLogs.filter(log => log.taken).length;

    const adherencePercentage = calculateAdherence(actualDoses, expectedDoses);

    return {
      medicationId,
      expectedDosesPerDay: expectedDoses,
      actualDosesTaken: cycle.doseLogs.map(log => ({
        timestamp: log.date,
        taken: log.taken
      })),
      adherencePercentage,
      streakCount: 0, // Will be calculated separately
      weeklyAdherenceHistory: []
    };
  }

  /**
   * Get weekly adherence for a medication
   */
  async getWeeklyAdherence(
    medicationId: string,
    userId: string,
    weeks: number = 4
  ): Promise<MedicationAdherenceData> {
    const endDate = endOfDay(new Date());
    const startDate = subDays(endDate, weeks * 7);

    const cycle = await this.prisma.medicationCycle.findFirst({
      where: {
        id: medicationId,
        userId: userId
      },
      include: {
        doseLogs: {
          where: {
            date: {
              gte: startDate,
              lte: endDate
            }
          },
          orderBy: {
            date: 'asc'
          }
        }
      }
    });

    if (!cycle) {
      throw new Error('Medication cycle not found');
    }

    // Group doses by week
    const weeklyHistory: MedicationAdherenceData['weeklyAdherenceHistory'] = [];
    const dosesByDay = groupDosesByDay(
      cycle.doseLogs.map(log => ({
        timestamp: log.date,
        taken: log.taken
      }))
    );

    // Calculate weekly adherence
    for (let weekOffset = 0; weekOffset < weeks; weekOffset++) {
      const weekStart = startOfWeek(subDays(endDate, weekOffset * 7), { weekStartsOn: 1 });
      const weekEnd = endOfWeek(weekStart, { weekStartsOn: 1 });

      let dosesTaken = 0;
      let dosesExpected = 0;

      const datesInWeek = getDatesInRange(weekStart, weekEnd);
      for (const date of datesInWeek) {
        const dayKey = date.toISOString();
        const dayData = dosesByDay.get(dayKey);
        
        if (dayData) {
          dosesTaken += dayData.taken;
          dosesExpected += dayData.expected;
        } else {
          // If no log exists, assume expected doses per day
          dosesExpected += cycle.dosesPerDay;
        }
      }

      const weekAdherence = calculateAdherence(dosesTaken, dosesExpected);

      weeklyHistory.push({
        weekStart,
        weekEnd,
        adherencePercentage: weekAdherence,
        dosesTaken,
        dosesExpected
      });
    }

    // Calculate overall adherence
    const allDoses = cycle.doseLogs;
    const totalTaken = allDoses.filter(log => log.taken).length;
    const totalExpected = allDoses.length || (differenceInDays(endDate, startDate) + 1) * cycle.dosesPerDay;

    const adherencePercentage = calculateAdherence(totalTaken, totalExpected);

    return {
      medicationId,
      expectedDosesPerDay: cycle.dosesPerDay,
      actualDosesTaken: allDoses.map(log => ({
        timestamp: log.date,
        taken: log.taken
      })),
      adherencePercentage,
      streakCount: 0, // Will be calculated separately
      weeklyAdherenceHistory: weeklyHistory.reverse() // Most recent first
    };
  }

  /**
   * Get medication streaks
   */
  async getMedicationStreaks(
    medicationId: string,
    userId: string,
    days: number = 30
  ): Promise<StreakData> {
    const endDate = endOfDay(new Date());
    const startDate = subDays(endDate, days);

    const cycle = await this.prisma.medicationCycle.findFirst({
      where: {
        id: medicationId,
        userId: userId
      },
      include: {
        doseLogs: {
          where: {
            date: {
              gte: startDate,
              lte: endDate
            }
          },
          orderBy: {
            date: 'asc'
          }
        }
      }
    });

    if (!cycle) {
      throw new Error('Medication cycle not found');
    }

    const dosesByDay = groupDosesByDay(
      cycle.doseLogs.map(log => ({
        timestamp: log.date,
        taken: log.taken
      }))
    );

    // Calculate streaks
    let currentStreak = 0;
    let longestStreak = 0;
    let tempStreak = 0;
    let streakStartDate: Date | null = null;
    let lastMissedDate: Date | null = null;
    let consecutiveMissedDays = 0;
    let maxConsecutiveMissed = 0;

    const datesInRange = getDatesInRange(startDate, endDate);
    
    for (const date of datesInRange) {
      const dayKey = date.toISOString();
      const dayData = dosesByDay.get(dayKey);
      
      const expectedDoses = cycle.dosesPerDay;
      const takenDoses = dayData?.taken || 0;
      const isFullyAdherent = isDayFullyAdherent(takenDoses, expectedDoses);

      if (isFullyAdherent) {
        tempStreak++;
        consecutiveMissedDays = 0;
        
        if (tempStreak > longestStreak) {
          longestStreak = tempStreak;
        }
      } else {
        if (tempStreak > currentStreak) {
          currentStreak = tempStreak;
          streakStartDate = subDays(date, tempStreak);
        }
        tempStreak = 0;
        consecutiveMissedDays++;
        
        if (consecutiveMissedDays > maxConsecutiveMissed) {
          maxConsecutiveMissed = consecutiveMissedDays;
          lastMissedDate = date;
        }
      }
    }

    // Update current streak if still active
    if (tempStreak > currentStreak) {
      currentStreak = tempStreak;
      streakStartDate = subDays(endDate, tempStreak);
    }

    return {
      medicationId,
      currentStreak,
      longestStreak,
      streakStartDate,
      lastMissedDate,
      consecutiveMissedDays: maxConsecutiveMissed
    };
  }

  /**
   * Analyze adherence patterns
   */
  async analyzeAdherencePatterns(
    medicationId: string,
    userId: string,
    days: number = 30
  ): Promise<AdherencePattern> {
    const adherenceData = await this.getWeeklyAdherence(medicationId, userId, Math.ceil(days / 7));
    
    if (adherenceData.weeklyAdherenceHistory.length < 2) {
      return {
        medicationId,
        pattern: 'stable',
        trend: 0,
        volatility: 0,
        averageAdherence: adherenceData.adherencePercentage,
        recentAdherence: adherenceData.adherencePercentage
      };
    }

    const weeklyAdherence = adherenceData.weeklyAdherenceHistory.map(w => w.adherencePercentage);
    const averageAdherence = weeklyAdherence.reduce((a, b) => a + b, 0) / weeklyAdherence.length;
    const recentAdherence = weeklyAdherence.slice(0, 2).reduce((a, b) => a + b, 0) / Math.min(2, weeklyAdherence.length);

    // Calculate trend (comparing first half to second half)
    const midPoint = Math.floor(weeklyAdherence.length / 2);
    const firstHalf = weeklyAdherence.slice(midPoint);
    const secondHalf = weeklyAdherence.slice(0, midPoint);
    
    const firstHalfAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondHalfAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    
    const trend = (secondHalfAvg - firstHalfAvg) / 100; // Normalize to -1 to 1

    // Calculate volatility (standard deviation)
    const variance = weeklyAdherence.reduce((acc, val) => {
      return acc + Math.pow(val - averageAdherence, 2);
    }, 0) / weeklyAdherence.length;
    const volatility = Math.sqrt(variance) / 100; // Normalize to 0-1

    // Determine pattern
    let pattern: AdherencePattern['pattern'] = 'stable';
    if (volatility > 0.2) {
      pattern = 'volatile';
    } else if (trend > 0.1) {
      pattern = 'improving';
    } else if (trend < -0.1) {
      pattern = 'declining';
    }

    return {
      medicationId,
      pattern,
      trend,
      volatility,
      averageAdherence,
      recentAdherence
    };
  }

  /**
   * Get adherence for all medications
   */
  async getAllMedicationsAdherence(
    userId: string,
    options: AdherenceCalculationOptions = {}
  ): Promise<MedicationAdherenceData[]> {
    const cycles = await this.prisma.medicationCycle.findMany({
      where: {
        userId: userId,
        ...(options.startDate && options.endDate ? {
          startDate: { lte: options.endDate },
          OR: [
            { endDate: null },
            { endDate: { gte: options.startDate } }
          ]
        } : {})
      }
    });

    const adherencePromises = cycles.map(cycle =>
      this.getWeeklyAdherence(cycle.id, userId, 4)
    );

    return Promise.all(adherencePromises);
  }
}








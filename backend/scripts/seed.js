/* eslint-disable no-console */
const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcrypt');

const prisma = new PrismaClient();

async function main() {
  // Clean existing data (optional during development) - order matters for foreign keys
  await prisma.notification.deleteMany();
  await prisma.doseLog.deleteMany();
  await prisma.metricLog.deleteMany();
  await prisma.medicationCycle.deleteMany();
  // Delete old schema data if exists
  try {
    await prisma.medicationLog?.deleteMany();
    await prisma.medication?.deleteMany();
    await prisma.metric?.deleteMany();
  } catch (e) {
    // Ignore if tables don't exist
  }
  await prisma.user.deleteMany();

  // Create a user
  const passwordHash = await bcrypt.hash('password123', 10);
  const user = await prisma.user.create({
    data: {
      email: 'testuser@example.com',
      password: passwordHash,
    },
  });

  // Dates
  const today = new Date();
  const threeDaysAgo = new Date(today);
  threeDaysAgo.setDate(today.getDate() - 3);
  const nextWeek = new Date(today);
  nextWeek.setDate(today.getDate() + 7);

  // Create medication cycles
  const amoxicillin = await prisma.medicationCycle.create({
    data: {
      userId: user.id,
      name: 'Amoxicillin',
      startDate: threeDaysAgo,
      endDate: nextWeek,
      dosage: '500mg',
      frequencyDays: 1, // daily
      dosesPerDay: 3,
    },
  });

  const ibuprofen = await prisma.medicationCycle.create({
    data: {
      userId: user.id,
      name: 'Ibuprofen',
      startDate: today,
      endDate: nextWeek,
      dosage: '200mg',
      frequencyDays: 1, // daily
      dosesPerDay: 2,
    },
  });

  // Dose logs (for amoxicillin - past doses)
  await prisma.doseLog.createMany({
    data: [
      { cycleId: amoxicillin.id, date: threeDaysAgo, taken: true },
      { cycleId: amoxicillin.id, date: new Date(today.getTime() - 24*60*60*1000), taken: true }, // yesterday
    ],
  });

  // Today's doses (not taken yet)
  const todayDate = new Date();
  todayDate.setHours(0, 0, 0, 0);
  await prisma.doseLog.createMany({
    data: [
      { cycleId: amoxicillin.id, date: todayDate, taken: false },
      { cycleId: ibuprofen.id, date: todayDate, taken: false },
    ],
  });

  // Metric logs
  await prisma.metricLog.createMany({
    data: [
      { 
        cycleId: amoxicillin.id,
        kind: 'Weight',
        valueFloat: 70.5,
        date: threeDaysAgo,
        notes: 'Starting weight'
      },
      { 
        cycleId: amoxicillin.id,
        kind: 'Blood Pressure',
        valueText: '120/80',
        date: threeDaysAgo,
        notes: 'Normal range'
      },
      { 
        cycleId: ibuprofen.id,
        kind: 'Weight',
        valueFloat: 70.2,
        date: today,
        notes: 'Current weight'
      },
    ],
  });

  // Future dose schedules (for upcoming intakes)
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);
  tomorrow.setHours(0, 0, 0, 0);
  
  await prisma.doseLog.createMany({
    data: [
      { cycleId: amoxicillin.id, date: tomorrow, taken: false },
      { cycleId: ibuprofen.id, date: tomorrow, taken: false },
    ],
  });

  console.log('🌱  Seed data generated successfully');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

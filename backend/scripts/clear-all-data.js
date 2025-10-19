const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

async function clearAllData() {
  try {
    console.log('🗑️  Clearing all data from database...');
    
    // Clear all data in the correct order (respecting foreign key constraints)
    await prisma.aiAuditLog.deleteMany();
    console.log('✅ Cleared AI audit logs');
    
    await prisma.condition.deleteMany();
    console.log('✅ Cleared conditions');
    
    await prisma.metricLog.deleteMany();
    console.log('✅ Cleared metric logs');
    
    await prisma.medicationLog.deleteMany();
    console.log('✅ Cleared medication logs');
    
    await prisma.doseLog.deleteMany();
    console.log('✅ Cleared dose logs');
    
    await prisma.medicationCycle.deleteMany();
    console.log('✅ Cleared medication cycles');
    
    await prisma.medication.deleteMany();
    console.log('✅ Cleared medications');
    
    await prisma.metric.deleteMany();
    console.log('✅ Cleared metrics');
    
    await prisma.notification.deleteMany();
    console.log('✅ Cleared notifications');
    
    await prisma.userSurveyData.deleteMany();
    console.log('✅ Cleared user survey data');
    
    await prisma.patient.deleteMany();
    console.log('✅ Cleared patients');
    
    await prisma.clinician.deleteMany();
    console.log('✅ Cleared clinicians');
    
    await prisma.user.deleteMany();
    console.log('✅ Cleared users');
    
    console.log('🎉 All data cleared successfully!');
    console.log('📝 Database is now clean and ready for fresh test data');
    
  } catch (error) {
    console.error('❌ Error clearing data:', error);
  } finally {
    await prisma.$disconnect();
  }
}

clearAllData();

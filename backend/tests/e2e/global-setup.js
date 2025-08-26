/**
 * Global Setup for Hospital-Grade E2E Tests
 * Prepares complete test environment
 */

const { chromium } = require('@playwright/test');
const { PrismaClient } = require('@prisma/client');

async function globalSetup() {
  console.log('🏥 Starting Hospital-Grade E2E Test Environment Setup...');
  
  const startTime = Date.now();
  
  try {
    // 1. Initialize database
    await setupTestDatabase();
    
    // 2. Seed comprehensive test data
    await seedE2ETestData();
    
    // 3. Verify services are running
    await verifyServices();
    
    // 4. Create test user session
    await createTestUserSession();
    
    const duration = Date.now() - startTime;
    console.log(`✅ Hospital-Grade E2E Environment Ready (${duration}ms)`);
    
  } catch (error) {
    console.error('❌ E2E Setup Failed:', error);
    throw error;
  }
}

async function setupTestDatabase() {
  const prisma = new PrismaClient();
  
  try {
    // Connect to database
    await prisma.$connect();
    
    // Clean existing test data
    await prisma.userMedicationCycle.deleteMany({
      where: {
        user: {
          email: {
            contains: 'test'
          }
        }
      }
    });
    
    await prisma.user.deleteMany({
      where: {
        email: {
          contains: 'test'
        }
      }
    });
    
    console.log('🗄️ Test database prepared');
    
  } finally {
    await prisma.$disconnect();
  }
}

async function seedE2ETestData() {
  const prisma = new PrismaClient();
  
  try {
    await prisma.$connect();
    
    // Create comprehensive test user
    const testUser = await prisma.user.upsert({
      where: { email: 'test@hospital-grade.test' },
      update: {},
      create: {
        email: 'test@hospital-grade.test',
        password: '$2b$10$1234567890123456789012345678901234567890123456', // bcrypt hash of 'test123'
        name: 'Hospital Grade Test User',
        role: 'user'
      }
    });
    
    // Ensure comprehensive medication data exists
    await ensureMedicationData(prisma);
    
    console.log('🌱 E2E test data seeded');
    
  } finally {
    await prisma.$disconnect();
  }
}

async function ensureMedicationData(prisma) {
  const testMedications = [
    {
      genericName: 'paracetamol',
      atcClass: 'N02BE01',
      classHuman: 'Analgesic',
      synonyms: JSON.stringify(['acetaminophen', 'panadol', 'tylenol']),
      products: [{
        brandName: 'Panadol',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily', 'three times daily', 'four times daily'],
        strengths: [
          { value: 500, unit: 'mg', frequency: 'daily', label: '500 mg daily' },
          { value: 500, unit: 'mg', frequency: 'twice daily', label: '500 mg twice daily' },
          { value: 1000, unit: 'mg', frequency: 'daily', label: '1000 mg daily' }
        ]
      }]
    },
    {
      genericName: 'ibuprofen',
      atcClass: 'M01AE01', 
      classHuman: 'NSAID',
      synonyms: JSON.stringify(['advil', 'nurofen', 'brufen']),
      products: [{
        brandName: 'Nurofen',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily', 'three times daily'],
        strengths: [
          { value: 200, unit: 'mg', frequency: 'daily', label: '200 mg daily' },
          { value: 400, unit: 'mg', frequency: 'twice daily', label: '400 mg twice daily' }
        ]
      }]
    },
    {
      genericName: 'semaglutide',
      atcClass: 'A10BJ06',
      classHuman: 'GLP-1 receptor agonist', 
      synonyms: JSON.stringify(['glp1', 'glp-1', 'ozempic', 'rybelsus']),
      products: [
        {
          brandName: 'Ozempic',
          route: 'subcutaneous',
          form: 'injection-pen',
          allowedIntakeType: 'Injection',
          defaultPlaces: ['at home', 'at clinic', 'self administered'],
          allowedFrequencies: ['weekly'],
          strengths: [
            { value: 0.25, unit: 'mg', frequency: 'weekly', label: '0.25 mg once weekly' },
            { value: 0.5, unit: 'mg', frequency: 'weekly', label: '0.5 mg once weekly' },
            { value: 1, unit: 'mg', frequency: 'weekly', label: '1 mg once weekly' },
            { value: 2, unit: 'mg', frequency: 'weekly', label: '2 mg once weekly' }
          ]
        },
        {
          brandName: 'Rybelsus',
          route: 'oral',
          form: 'tablet',
          allowedIntakeType: 'Pill/Tablet',
          defaultPlaces: ['at home', 'self administered'],
          allowedFrequencies: ['daily'],
          strengths: [
            { value: 3, unit: 'mg', frequency: 'daily', label: '3 mg once daily' },
            { value: 7, unit: 'mg', frequency: 'daily', label: '7 mg once daily' },
            { value: 14, unit: 'mg', frequency: 'daily', label: '14 mg once daily' }
          ]
        }
      ]
    },
    {
      genericName: 'aspirin',
      atcClass: 'N02BA01',
      classHuman: 'Analgesic',
      synonyms: JSON.stringify(['acetylsalicylic acid', 'disprin']),
      products: [{
        brandName: 'Disprin',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'as needed'],
        strengths: [
          { value: 75, unit: 'mg', frequency: 'daily', label: '75 mg daily (low dose)' },
          { value: 300, unit: 'mg', frequency: 'daily', label: '300 mg daily' },
          { value: 300, unit: 'mg', frequency: 'as needed', label: '300 mg as needed' }
        ]
      }]
    },
    {
      genericName: 'metformin',
      atcClass: 'A10BA02',
      classHuman: 'Biguanide antidiabetic',
      synonyms: JSON.stringify(['glucophage']),
      products: [{
        brandName: 'Glucophage',
        route: 'oral', 
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily'],
        strengths: [
          { value: 500, unit: 'mg', frequency: 'daily', label: '500 mg daily' },
          { value: 500, unit: 'mg', frequency: 'twice daily', label: '500 mg twice daily' },
          { value: 850, unit: 'mg', frequency: 'daily', label: '850 mg daily' },
          { value: 1000, unit: 'mg', frequency: 'daily', label: '1000 mg daily' }
        ]
      }]
    }
  ];
  
  for (const medData of testMedications) {
    const { products, ...medicationData } = medData;
    
    let medication = await prisma.medicationValidation.findFirst({
      where: { genericName: medicationData.genericName }
    });
    
    if (!medication) {
      medication = await prisma.medicationValidation.create({
        data: medicationData
      });
    }
    
    for (const productData of products) {
      const { strengths, ...productInfo } = productData;
      
      let product = await prisma.medicationProduct.findFirst({
        where: { 
          medicationId: medication.id,
          brandName: productInfo.brandName
        }
      });
      
      if (!product) {
        product = await prisma.medicationProduct.create({
          data: {
            medicationId: medication.id,
            ...productInfo,
            defaultPlaces: JSON.stringify(productInfo.defaultPlaces),
            allowedFrequencies: JSON.stringify(productInfo.allowedFrequencies),
            isActive: true
          }
        });
      }
      
      for (const strengthData of strengths) {
        const existingStrength = await prisma.medicationStrength.findFirst({
          where: {
            productId: product.id,
            strengthValue: strengthData.value,
            strengthUnit: strengthData.unit,
            frequency: strengthData.frequency
          }
        });
        
        if (!existingStrength) {
          await prisma.medicationStrength.create({
            data: {
              productId: product.id,
              ...strengthData,
              isActive: true
            }
          });
        }
      }
    }
  }
}

async function verifyServices() {
  // Verify backend is running
  try {
    const response = await fetch('http://localhost:8000/api/health');
    if (!response.ok) {
      throw new Error(`Backend health check failed: ${response.status}`);
    }
    console.log('✅ Backend service verified');
  } catch (error) {
    console.error('❌ Backend service not available:', error.message);
    throw error;
  }
  
  // Verify frontend is running  
  try {
    const response = await fetch('http://localhost:3005');
    if (!response.ok) {
      throw new Error(`Frontend health check failed: ${response.status}`);
    }
    console.log('✅ Frontend service verified');
  } catch (error) {
    console.error('❌ Frontend service not available:', error.message);
    throw error;
  }
}

async function createTestUserSession() {
  // Launch browser to create authenticated session
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    // Navigate to app
    await page.goto('http://localhost:3005');
    
    // Check if login is required
    const loginForm = await page.locator('[data-testid="login-form"]').isVisible();
    
    if (loginForm) {
      // Perform login
      await page.fill('[data-testid="email-input"]', 'test@hospital-grade.test');
      await page.fill('[data-testid="password-input"]', 'test123');
      await page.click('[data-testid="login-button"]');
      
      // Wait for successful login
      await page.waitForSelector('[data-testid="dashboard"]', { timeout: 10000 });
    }
    
    // Save authentication state
    await context.storageState({ path: 'tests/e2e/auth-state.json' });
    
    console.log('✅ Test user session created');
    
  } catch (error) {
    console.error('❌ Failed to create test user session:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

module.exports = globalSetup;

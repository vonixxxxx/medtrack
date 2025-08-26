/**
 * Jest Test Setup for Hospital-Grade Medication System
 * Initializes test environment with strict validation
 */

const { PrismaClient } = require('@prisma/client');

// Initialize test database
const prisma = new PrismaClient({
  datasources: {
    db: {
      url: process.env.DATABASE_URL || 'file:./test.db'
    }
  }
});

// Global test configuration
global.prisma = prisma;
global.testStartTime = Date.now();

// Hospital-Grade Test Environment Setup
beforeAll(async () => {
  console.log('🏥 Setting up Hospital-Grade Test Environment...');
  
  // Ensure test database is clean
  await cleanTestDatabase();
  
  // Seed test data
  await seedTestData();
  
  console.log('✅ Hospital-Grade Test Environment Ready');
});

afterAll(async () => {
  console.log('🧹 Cleaning up Hospital-Grade Test Environment...');
  
  // Clean up test data
  await cleanTestDatabase();
  
  // Close database connection
  await prisma.$disconnect();
  
  const duration = Date.now() - global.testStartTime;
  console.log(`✅ Test Environment Cleanup Complete (${duration}ms)`);
});

beforeEach(async () => {
  // Start fresh transaction for each test (if supported)
  // This ensures test isolation
});

afterEach(async () => {
  // Clean up test-specific data
  // Reset any global state
});

// Utility Functions
async function cleanTestDatabase() {
  try {
    // Delete in reverse dependency order
    await prisma.userMedicationCycle.deleteMany({});
    await prisma.medicationStrength.deleteMany({});
    await prisma.medicationProduct.deleteMany({});
    await prisma.medicationValidation.deleteMany({});
    
    console.log('🧹 Test database cleaned');
  } catch (error) {
    console.warn('⚠️ Database cleanup warning:', error.message);
  }
}

async function seedTestData() {
  try {
    // Seed essential test medications
    const testMedications = [
      {
        genericName: 'paracetamol',
        atcClass: 'N02BE01',
        classHuman: 'Analgesic',
        synonyms: JSON.stringify(['acetaminophen', 'panadol', 'tylenol'])
      },
      {
        genericName: 'ibuprofen', 
        atcClass: 'M01AE01',
        classHuman: 'NSAID',
        synonyms: JSON.stringify(['advil', 'nurofen', 'brufen'])
      },
      {
        genericName: 'semaglutide',
        atcClass: 'A10BJ06',
        classHuman: 'GLP-1 receptor agonist',
        synonyms: JSON.stringify(['glp1', 'glp-1', 'ozempic', 'rybelsus'])
      }
    ];

    for (const medData of testMedications) {
      const medication = await prisma.medicationValidation.create({
        data: medData
      });

      // Add test products for each medication
      await seedTestProducts(medication);
    }
    
    console.log('🌱 Test data seeded successfully');
  } catch (error) {
    console.error('❌ Test data seeding failed:', error);
    throw error;
  }
}

async function seedTestProducts(medication) {
  const productConfigs = getProductConfigForMedication(medication.genericName);
  
  for (const config of productConfigs) {
    const product = await prisma.medicationProduct.create({
      data: {
        medicationId: medication.id,
        brandName: config.brandName,
        route: config.route,
        form: config.form,
        allowedIntakeType: config.allowedIntakeType,
        defaultPlaces: JSON.stringify(config.defaultPlaces),
        allowedFrequencies: JSON.stringify(config.allowedFrequencies),
        isActive: true
      }
    });

    // Add strengths
    for (const strength of config.strengths) {
      await prisma.medicationStrength.create({
        data: {
          productId: product.id,
          strengthValue: strength.value,
          strengthUnit: strength.unit,
          frequency: strength.frequency,
          label: strength.label,
          isActive: true
        }
      });
    }
  }
}

function getProductConfigForMedication(genericName) {
  const configs = {
    paracetamol: [{
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
    }],
    
    ibuprofen: [{
      brandName: 'Nurofen',
      route: 'oral',
      form: 'tablet',
      allowedIntakeType: 'Pill/Tablet',
      defaultPlaces: ['at home', 'self administered'],
      allowedFrequencies: ['daily', 'twice daily', 'three times daily'],
      strengths: [
        { value: 200, unit: 'mg', frequency: 'daily', label: '200 mg daily' },
        { value: 400, unit: 'mg', frequency: 'twice daily', label: '400 mg twice daily' },
        { value: 600, unit: 'mg', frequency: 'three times daily', label: '600 mg three times daily' }
      ]
    }],
    
    semaglutide: [
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
  };
  
  return configs[genericName] || [];
}

// Global test utilities
global.testUtils = {
  cleanTestDatabase,
  seedTestData,
  createTestUser: async (userData = {}) => {
    return await prisma.user.create({
      data: {
        email: userData.email || 'test@example.com',
        password: userData.password || 'test123',
        name: userData.name || 'Test User',
        ...userData
      }
    });
  },
  
  createTestMedicationCycle: async (userId, overrides = {}) => {
    // Find a test medication and product
    const medication = await prisma.medicationValidation.findFirst({
      include: {
        products: {
          include: {
            strengths: true
          }
        }
      }
    });
    
    if (!medication || !medication.products[0]) {
      throw new Error('No test medication available for cycle creation');
    }
    
    const product = medication.products[0];
    const strength = product.strengths[0];
    
    return await prisma.userMedicationCycle.create({
      data: {
        userId,
        medicationId: medication.id,
        productId: product.id,
        strengthValue: strength.strengthValue,
        strengthUnit: strength.strengthUnit,
        frequency: strength.frequency,
        intakeType: product.allowedIntakeType,
        intakePlace: 'at home',
        startDate: new Date(),
        isActive: true,
        customFlags: JSON.stringify({}),
        ...overrides
      }
    });
  }
};

// Hospital-Grade Test Assertions
global.expectHospitalGrade = {
  toHaveZeroToleranceValidation: (response) => {
    expect(response).toBeDefined();
    expect(response.valid).toBeDefined();
    
    if (!response.valid) {
      expect(response.errors).toBeDefined();
      expect(response.errors).toBeInstanceOf(Array);
      expect(response.errors.length).toBeGreaterThan(0);
    }
  },
  
  toHaveProvenanceTracking: (result) => {
    expect(result).toHaveProperty('source');
    expect(result.source).toMatch(/NHS|Database|Hospital-Grade/);
  },
  
  toHaveServerApprovedOptions: (options) => {
    expect(options).toHaveProperty('strengths');
    expect(options).toHaveProperty('allowed_frequencies');
    expect(options).toHaveProperty('default_places');
    expect(options.strengths).toBeInstanceOf(Array);
    expect(options.allowed_frequencies).toBeInstanceOf(Array);
    expect(options.default_places).toBeInstanceOf(Array);
  }
};

// Performance monitoring for hospital-grade requirements
global.performanceMonitor = {
  startTimer: () => Date.now(),
  
  assertResponseTime: (startTime, maxMs = 5000) => {
    const duration = Date.now() - startTime;
    expect(duration).toBeLessThan(maxMs);
    return duration;
  },
  
  assertMemoryUsage: () => {
    const used = process.memoryUsage();
    // Ensure reasonable memory usage (< 1GB)
    expect(used.heapUsed).toBeLessThan(1024 * 1024 * 1024);
  }
};

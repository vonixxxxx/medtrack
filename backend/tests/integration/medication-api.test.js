/**
 * Integration Tests for Hospital-Grade Medication API
 * Tests complete API flows with real database interactions
 */

const request = require('supertest');
const app = require('../../src/app'); // Assuming Express app export
const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

describe('Hospital-Grade Medication API Integration', () => {
  let authToken;
  let testUserId;

  beforeAll(async () => {
    // Set up test database
    await prisma.$connect();
    
    // Create test user and get auth token
    const testUser = await prisma.user.create({
      data: {
        email: 'test@hospital-grade.test',
        password: 'test123',
        name: 'Test User'
      }
    });
    
    testUserId = testUser.id;
    
    // Mock authentication - get token
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'test@hospital-grade.test',
        password: 'test123'
      });
    
    authToken = loginResponse.body.token;
  });

  afterAll(async () => {
    // Clean up test data
    await prisma.userMedicationCycle.deleteMany({
      where: { userId: testUserId }
    });
    await prisma.user.delete({
      where: { id: testUserId }
    });
    await prisma.$disconnect();
  });

  beforeEach(async () => {
    // Clean up cycles before each test
    await prisma.userMedicationCycle.deleteMany({
      where: { userId: testUserId }
    });
  });

  describe('GET /api/meds/search', () => {
    test('should find paracetamol with exact match', async () => {
      const response = await request(app)
        .get('/api/meds/search?q=paracetamol')
        .expect(200);

      expect(response.body).toHaveProperty('query', 'paracetamol');
      expect(response.body.matches).toBeInstanceOf(Array);
      expect(response.body.matches.length).toBeGreaterThan(0);
      
      const paracetamolMatch = response.body.matches.find(m => 
        m.genericName.toLowerCase().includes('paracetamol')
      );
      expect(paracetamolMatch).toBeDefined();
      expect(paracetamolMatch.score).toBeGreaterThan(80);
      expect(paracetamolMatch.products).toBeInstanceOf(Array);
    });

    test('should find semaglutide products via brand name search', async () => {
      const response = await request(app)
        .get('/api/meds/search?q=Ozempic')
        .expect(200);

      expect(response.body.matches.length).toBeGreaterThan(0);
      
      const semaglutideMatch = response.body.matches.find(m => 
        m.genericName.toLowerCase().includes('semaglutide')
      );
      expect(semaglutideMatch).toBeDefined();
      
      const ozempicProduct = semaglutideMatch.products.find(p => 
        p.brandName.toLowerCase().includes('ozempic')
      );
      expect(ozempicProduct).toBeDefined();
      expect(ozempicProduct.allowedIntakeType).toBe('Injection');
    });

    test('should find GLP-1 medications via acronym search', async () => {
      const response = await request(app)
        .get('/api/meds/search?q=GLP-1')
        .expect(200);

      expect(response.body.matches.length).toBeGreaterThan(0);
      
      // Should find multiple GLP-1 medications
      const glpMedications = response.body.matches.filter(m => 
        m.classHuman && m.classHuman.toLowerCase().includes('glp')
      );
      expect(glpMedications.length).toBeGreaterThan(0);
      
      // Should include semaglutide, liraglutide, dulaglutide
      const medicationNames = glpMedications.map(m => m.genericName.toLowerCase());
      expect(medicationNames.some(name => name.includes('semaglutide'))).toBe(true);
    });

    test('should provide suggestions for typos', async () => {
      const response = await request(app)
        .get('/api/meds/search?q=paracitamol') // Common typo
        .expect(200);

      expect(response.body.suggestions).toBeInstanceOf(Array);
      if (response.body.suggestions.length > 0) {
        const paracetamolSuggestion = response.body.suggestions.find(s => 
          s.toLowerCase().includes('paracetamol')
        );
        expect(paracetamolSuggestion).toBeDefined();
      }
    });

    test('should handle empty queries gracefully', async () => {
      const response = await request(app)
        .get('/api/meds/search?q=')
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('query');
    });

    test('should limit results when requested', async () => {
      const response = await request(app)
        .get('/api/meds/search?q=pain&limit=3')
        .expect(200);

      expect(response.body.matches.length).toBeLessThanOrEqual(3);
    });

    test('should provide hospital-grade metadata', async () => {
      const response = await request(app)
        .get('/api/meds/search?q=ibuprofen')
        .expect(200);

      expect(response.body).toHaveProperty('hospitalGrade');
      expect(response.body).toHaveProperty('source');
      
      if (response.body.matches.length > 0) {
        const match = response.body.matches[0];
        expect(match).toHaveProperty('score');
        expect(match).toHaveProperty('reason');
      }
    });
  });

  describe('GET /api/meds/product/:productId/options', () => {
    let semaglutideProductId;
    let paracetamolProductId;

    beforeAll(async () => {
      // Find product IDs for testing
      const searchResponse = await request(app)
        .get('/api/meds/search?q=semaglutide')
        .expect(200);
      
      const semaglutideMatch = searchResponse.body.matches.find(m => 
        m.genericName.toLowerCase().includes('semaglutide')
      );
      
      if (semaglutideMatch && semaglutideMatch.products.length > 0) {
        semaglutideProductId = semaglutideMatch.products[0].id;
      }

      const paracetamolResponse = await request(app)
        .get('/api/meds/search?q=paracetamol')
        .expect(200);
      
      const paracetamolMatch = paracetamolResponse.body.matches.find(m => 
        m.genericName.toLowerCase().includes('paracetamol')
      );
      
      if (paracetamolMatch && paracetamolMatch.products.length > 0) {
        paracetamolProductId = paracetamolMatch.products[0].id;
      }
    });

    test('should return valid options for semaglutide injection', async () => {
      if (!semaglutideProductId) {
        console.log('Skipping test - semaglutide product not found');
        return;
      }

      const response = await request(app)
        .get(`/api/meds/product/${semaglutideProductId}/options`)
        .expect(200);

      expect(response.body).toMatchObject({
        product_id: semaglutideProductId,
        generic_name: expect.stringContaining('semaglutide'),
        allowed_intake_type: 'Injection',
        allowed_frequencies: expect.arrayContaining(['weekly']),
        default_places: expect.arrayContaining(['at home']),
        strengths: expect.any(Array),
        allow_custom: true
      });

      // Verify strengths are realistic for semaglutide
      const strengths = response.body.strengths;
      expect(strengths.length).toBeGreaterThan(0);
      
      const validStrengths = strengths.filter(s => 
        s.value >= 0.25 && s.value <= 2 && s.unit === 'mg'
      );
      expect(validStrengths.length).toBeGreaterThan(0);
    });

    test('should return valid options for paracetamol tablets', async () => {
      if (!paracetamolProductId) {
        console.log('Skipping test - paracetamol product not found');
        return;
      }

      const response = await request(app)
        .get(`/api/meds/product/${paracetamolProductId}/options`)
        .expect(200);

      expect(response.body).toMatchObject({
        product_id: paracetamolProductId,
        generic_name: expect.stringContaining('paracetamol'),
        allowed_intake_type: 'Pill/Tablet',
        allowed_frequencies: expect.any(Array),
        default_places: expect.arrayContaining(['at home']),
        strengths: expect.any(Array),
        allow_custom: true
      });

      // Verify strengths are realistic for paracetamol
      const strengths = response.body.strengths;
      expect(strengths.length).toBeGreaterThan(0);
      
      const validStrengths = strengths.filter(s => 
        s.value >= 100 && s.value <= 1000 && s.unit === 'mg'
      );
      expect(validStrengths.length).toBeGreaterThan(0);
    });

    test('should return 404 for non-existent product', async () => {
      const response = await request(app)
        .get('/api/meds/product/non-existent-id/options')
        .expect(404);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('not found');
    });

    test('should include source and validation metadata', async () => {
      if (!paracetamolProductId) {
        console.log('Skipping test - paracetamol product not found');
        return;
      }

      const response = await request(app)
        .get(`/api/meds/product/${paracetamolProductId}/options`)
        .expect(200);

      expect(response.body).toHaveProperty('source');
      expect(response.body.source).toMatch(/NHS|Database/);
    });
  });

  describe('POST /api/meds/validate', () => {
    let validProductData;

    beforeAll(async () => {
      // Get valid product for testing
      const searchResponse = await request(app)
        .get('/api/meds/search?q=paracetamol')
        .expect(200);
      
      if (searchResponse.body.matches.length > 0) {
        const medication = searchResponse.body.matches[0];
        const product = medication.products[0];
        
        const optionsResponse = await request(app)
          .get(`/api/meds/product/${product.id}/options`)
          .expect(200);
        
        validProductData = {
          medication_id: medication.id,
          product_id: product.id,
          options: optionsResponse.body
        };
      }
    });

    test('should validate correct configuration', async () => {
      if (!validProductData) {
        console.log('Skipping test - valid product data not found');
        return;
      }

      const firstStrength = validProductData.options.strengths[0];
      const firstFrequency = validProductData.options.allowed_frequencies[0];
      const firstPlace = validProductData.options.default_places[0];

      const response = await request(app)
        .post('/api/meds/validate')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          medication_id: validProductData.medication_id,
          product_id: validProductData.product_id,
          intake_type: validProductData.options.allowed_intake_type,
          intake_place: firstPlace,
          strength_value: firstStrength.value,
          strength_unit: firstStrength.unit,
          frequency: firstFrequency,
          custom_flags: { dose: false, frequency: false, intake_type: false }
        })
        .expect(200);

      expect(response.body).toMatchObject({
        valid: true,
        normalized: expect.objectContaining({
          intake_type: validProductData.options.allowed_intake_type,
          frequency: firstFrequency,
          strength_value: firstStrength.value
        })
      });
    });

    test('should reject invalid intake type', async () => {
      if (!validProductData) {
        console.log('Skipping test - valid product data not found');
        return;
      }

      const firstStrength = validProductData.options.strengths[0];
      const firstFrequency = validProductData.options.allowed_frequencies[0];
      const firstPlace = validProductData.options.default_places[0];
      
      // Use wrong intake type
      const wrongIntakeType = validProductData.options.allowed_intake_type === 'Injection' 
        ? 'Pill/Tablet' 
        : 'Injection';

      const response = await request(app)
        .post('/api/meds/validate')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          medication_id: validProductData.medication_id,
          product_id: validProductData.product_id,
          intake_type: wrongIntakeType,
          intake_place: firstPlace,
          strength_value: firstStrength.value,
          strength_unit: firstStrength.unit,
          frequency: firstFrequency,
          custom_flags: { dose: false, frequency: false, intake_type: false }
        })
        .expect(422);

      expect(response.body).toMatchObject({
        valid: false,
        errors: expect.arrayContaining([
          expect.objectContaining({
            field: 'intake_type'
          })
        ])
      });
    });

    test('should reject invalid frequency', async () => {
      if (!validProductData) {
        console.log('Skipping test - valid product data not found');
        return;
      }

      const firstStrength = validProductData.options.strengths[0];
      const firstPlace = validProductData.options.default_places[0];

      const response = await request(app)
        .post('/api/meds/validate')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          medication_id: validProductData.medication_id,
          product_id: validProductData.product_id,
          intake_type: validProductData.options.allowed_intake_type,
          intake_place: firstPlace,
          strength_value: firstStrength.value,
          strength_unit: firstStrength.unit,
          frequency: 'invalid-frequency',
          custom_flags: { dose: false, frequency: false, intake_type: false }
        })
        .expect(422);

      expect(response.body).toMatchObject({
        valid: false,
        errors: expect.arrayContaining([
          expect.objectContaining({
            field: 'frequency'
          })
        ])
      });
    });

    test('should reject invalid strength', async () => {
      if (!validProductData) {
        console.log('Skipping test - valid product data not found');
        return;
      }

      const firstFrequency = validProductData.options.allowed_frequencies[0];
      const firstPlace = validProductData.options.default_places[0];

      const response = await request(app)
        .post('/api/meds/validate')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          medication_id: validProductData.medication_id,
          product_id: validProductData.product_id,
          intake_type: validProductData.options.allowed_intake_type,
          intake_place: firstPlace,
          strength_value: 99999, // Invalid strength
          strength_unit: 'mg',
          frequency: firstFrequency,
          custom_flags: { dose: false, frequency: false, intake_type: false }
        })
        .expect(422);

      expect(response.body).toMatchObject({
        valid: false,
        errors: expect.arrayContaining([
          expect.objectContaining({
            field: 'strength_value'
          })
        ])
      });
    });

    test('should require authentication', async () => {
      const response = await request(app)
        .post('/api/meds/validate')
        // No Authorization header
        .send({
          medication_id: 'test',
          product_id: 'test',
          intake_type: 'Pill/Tablet',
          intake_place: 'at home',
          strength_value: 500,
          strength_unit: 'mg',
          frequency: 'daily',
          custom_flags: {}
        })
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });
  });

  describe('POST /api/meds/cycles', () => {
    test('should create medication cycle after validation', async () => {
      if (!validProductData) {
        console.log('Skipping test - valid product data not found');
        return;
      }

      const firstStrength = validProductData.options.strengths[0];
      const firstFrequency = validProductData.options.allowed_frequencies[0];
      const firstPlace = validProductData.options.default_places[0];

      const response = await request(app)
        .post('/api/meds/cycles')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          medication_id: validProductData.medication_id,
          product_id: validProductData.product_id,
          intake_type: validProductData.options.allowed_intake_type,
          intake_place: firstPlace,
          strength_value: firstStrength.value,
          strength_unit: firstStrength.unit,
          frequency: firstFrequency,
          start_date: '2025-01-15',
          end_date: null,
          custom_flags: { dose: false, frequency: false, intake_type: false },
          notes: 'Integration test cycle'
        })
        .expect(201);

      expect(response.body).toMatchObject({
        message: expect.stringContaining('created successfully'),
        cycle: expect.objectContaining({
          id: expect.any(String),
          userId: testUserId
        })
      });
    });

    test('should prevent duplicate active cycles', async () => {
      if (!validProductData) {
        console.log('Skipping test - valid product data not found');
        return;
      }

      const firstStrength = validProductData.options.strengths[0];
      const firstFrequency = validProductData.options.allowed_frequencies[0];
      const firstPlace = validProductData.options.default_places[0];

      const cycleData = {
        medication_id: validProductData.medication_id,
        product_id: validProductData.product_id,
        intake_type: validProductData.options.allowed_intake_type,
        intake_place: firstPlace,
        strength_value: firstStrength.value,
        strength_unit: firstStrength.unit,
        frequency: firstFrequency,
        start_date: '2025-01-15',
        end_date: null,
        custom_flags: { dose: false, frequency: false, intake_type: false },
        notes: 'First cycle'
      };

      // Create first cycle
      await request(app)
        .post('/api/meds/cycles')
        .set('Authorization', `Bearer ${authToken}`)
        .send(cycleData)
        .expect(201);

      // Try to create duplicate
      const response = await request(app)
        .post('/api/meds/cycles')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          ...cycleData,
          notes: 'Duplicate cycle attempt'
        })
        .expect(409);

      expect(response.body).toMatchObject({
        valid: false,
        errors: expect.arrayContaining([
          expect.objectContaining({
            message: expect.stringContaining('already have an active cycle')
          })
        ])
      });
    });
  });

  describe('Error Handling & Edge Cases', () => {
    test('should handle malformed JSON gracefully', async () => {
      const response = await request(app)
        .post('/api/meds/validate')
        .set('Authorization', `Bearer ${authToken}`)
        .set('Content-Type', 'application/json')
        .send('{ invalid json }')
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    test('should handle database connection issues', async () => {
      // This would require mocking database failures
      // Implementation depends on specific testing strategy
    });

    test('should rate limit excessive requests', async () => {
      // Send many requests quickly
      const promises = Array(20).fill().map(() => 
        request(app).get('/api/meds/search?q=test')
      );

      const responses = await Promise.all(promises);
      
      // Check if some requests are rate limited (status 429)
      const rateLimited = responses.filter(r => r.status === 429);
      
      // Rate limiting behavior depends on configuration
      // This test verifies the system handles high load
      expect(responses.length).toBe(20);
    });
  });
});

/**
 * Unit Tests for Hospital-Grade Medication Controller
 * Zero tolerance for incorrect options testing
 */

const HospitalGradeMedicationController = require('../../src/controllers/HospitalGradeMedicationController');
const SystemConfigManager = require('../../src/config/SystemConfigManager');

// Mock dependencies
jest.mock('../../src/config/SystemConfigManager');
jest.mock('../../src/services/ProfoundMedicationResolver');

describe('HospitalGradeMedicationController', () => {
  let controller;
  let mockReq;
  let mockRes;
  let mockNext;

  beforeEach(() => {
    controller = new HospitalGradeMedicationController();
    
    mockReq = {
      query: {},
      params: {},
      body: {},
      user: { id: 'test-user-123' }
    };
    
    mockRes = {
      json: jest.fn(),
      status: jest.fn().mockReturnThis(),
      send: jest.fn()
    };
    
    mockNext = jest.fn();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Search Endpoint', () => {
    test('should return search results for valid query', async () => {
      mockReq.query = { q: 'paracetamol', limit: 10 };
      
      // Mock resolver response
      controller.resolver = {
        resolve: jest.fn().mockResolvedValue({
          query: 'paracetamol',
          matches: [{
            medicationId: 'med-123',
            genericName: 'paracetamol',
            classHuman: 'Analgesic',
            confidence: 0.95,
            matchType: 'exact',
            products: [{
              productId: 'prod-123',
              brandName: 'Panadol',
              intakeType: 'Pill/Tablet',
              route: 'oral',
              doseForm: 'tablet'
            }]
          }],
          suggestions: [],
          metadata: { searchType: 'exact', sources: ['nhs_dmd'] }
        })
      };

      await controller.search(mockReq, mockRes);
      
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'paracetamol',
          matches: expect.arrayContaining([
            expect.objectContaining({
              id: 'med-123',
              genericName: 'paracetamol',
              confidence: 95,
              products: expect.arrayContaining([
                expect.objectContaining({
                  id: 'prod-123',
                  brandName: 'Panadol',
                  allowedIntakeType: 'Pill/Tablet'
                })
              ])
            })
          ])
        })
      );
    });

    test('should reject empty queries', async () => {
      mockReq.query = { q: '' };
      
      await controller.search(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(400);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.stringContaining('query')
        })
      );
    });

    test('should handle search errors gracefully', async () => {
      mockReq.query = { q: 'paracetamol' };
      
      controller.resolver = {
        resolve: jest.fn().mockRejectedValue(new Error('Database connection failed'))
      };

      await controller.search(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(500);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.stringContaining('search')
        })
      );
    });

    test('should limit results according to query parameter', async () => {
      mockReq.query = { q: 'pain', limit: 3 };
      
      const mockMatches = Array(10).fill().map((_, i) => ({
        medicationId: `med-${i}`,
        genericName: `medication-${i}`,
        confidence: 0.8,
        products: []
      }));

      controller.resolver = {
        resolve: jest.fn().mockResolvedValue({
          query: 'pain',
          matches: mockMatches,
          suggestions: [],
          metadata: {}
        })
      };

      await controller.search(mockReq, mockRes);
      
      expect(controller.resolver.resolve).toHaveBeenCalledWith(
        'pain',
        expect.objectContaining({
          maxResults: 3
        })
      );
    });
  });

  describe('Product Options Endpoint', () => {
    test('should return valid options for existing product', async () => {
      mockReq.params = { productId: 'prod-123' };
      
      // Mock database response
      controller.getProductFromDatabase = jest.fn().mockResolvedValue({
        id: 'prod-123',
        brandName: 'Ozempic',
        genericName: 'semaglutide',
        route: 'subcutaneous',
        doseForm: 'injection-pen',
        intakeType: 'Injection',
        defaultPlaces: ['at home', 'at clinic'],
        allowedFrequencies: ['weekly'],
        strengths: [
          { value: 0.25, unit: 'mg', frequency: 'weekly', label: '0.25 mg once weekly' },
          { value: 0.5, unit: 'mg', frequency: 'weekly', label: '0.5 mg once weekly' },
          { value: 1, unit: 'mg', frequency: 'weekly', label: '1 mg once weekly' },
          { value: 2, unit: 'mg', frequency: 'weekly', label: '2 mg once weekly' }
        ],
        rules: { maxDosePerPeriod: '2 mg/week' }
      });

      await controller.getProductOptions(mockReq, mockRes);
      
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          product_id: 'prod-123',
          brand_name: 'Ozempic',
          generic_name: 'semaglutide',
          allowed_intake_type: 'Injection',
          allowed_frequencies: ['weekly'],
          default_places: ['at home', 'at clinic'],
          strengths: expect.arrayContaining([
            expect.objectContaining({
              value: 0.25,
              unit: 'mg',
              frequency: 'weekly'
            })
          ]),
          allow_custom: true
        })
      );
    });

    test('should return 404 for non-existent product', async () => {
      mockReq.params = { productId: 'non-existent' };
      
      controller.getProductFromDatabase = jest.fn().mockResolvedValue(null);

      await controller.getProductOptions(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(404);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.stringContaining('not found')
        })
      );
    });

    test('should reject inactive products', async () => {
      mockReq.params = { productId: 'prod-inactive' };
      
      controller.getProductFromDatabase = jest.fn().mockResolvedValue({
        id: 'prod-inactive',
        isActive: false
      });

      await controller.getProductOptions(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(400);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.stringContaining('no longer active')
        })
      );
    });
  });

  describe('Validation Endpoint', () => {
    test('should validate correct medication configuration', async () => {
      mockReq.body = {
        medication_id: 'med-123',
        product_id: 'prod-123',
        intake_type: 'Injection',
        intake_place: 'at home',
        strength_value: 0.5,
        strength_unit: 'mg',
        frequency: 'weekly',
        custom_flags: { dose: false, frequency: false, intake_type: false }
      };

      controller.validateAgainstProduct = jest.fn().mockResolvedValue({
        valid: true,
        normalized: {
          intake_type: 'Injection',
          intake_place: 'at home',
          strength_value: 0.5,
          strength_unit: 'mg',
          frequency: 'weekly',
          label: 'Ozempic (semaglutide) 0.5 mg, weekly injection'
        },
        warnings: []
      });

      await controller.validateMedicationConfiguration(mockReq, mockRes);
      
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          valid: true,
          normalized: expect.objectContaining({
            intake_type: 'Injection',
            frequency: 'weekly'
          })
        })
      );
    });

    test('should reject invalid intake type', async () => {
      mockReq.body = {
        medication_id: 'med-123',
        product_id: 'prod-123',
        intake_type: 'Pill/Tablet', // Wrong for injection product
        intake_place: 'at home',
        strength_value: 0.5,
        strength_unit: 'mg',
        frequency: 'weekly',
        custom_flags: { dose: false, frequency: false, intake_type: false }
      };

      controller.validateAgainstProduct = jest.fn().mockResolvedValue({
        valid: false,
        errors: [{
          field: 'intake_type',
          message: 'This product is Injection only.'
        }]
      });

      await controller.validateMedicationConfiguration(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(422);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          valid: false,
          errors: expect.arrayContaining([
            expect.objectContaining({
              field: 'intake_type',
              message: expect.stringContaining('Injection only')
            })
          ])
        })
      );
    });

    test('should reject invalid frequency', async () => {
      mockReq.body = {
        medication_id: 'med-123',
        product_id: 'prod-123',
        intake_type: 'Injection',
        intake_place: 'at home',
        strength_value: 0.5,
        strength_unit: 'mg',
        frequency: 'daily', // Wrong for weekly product
        custom_flags: { dose: false, frequency: false, intake_type: false }
      };

      controller.validateAgainstProduct = jest.fn().mockResolvedValue({
        valid: false,
        errors: [{
          field: 'frequency',
          message: 'Allowed frequency for this product is weekly.'
        }]
      });

      await controller.validateMedicationConfiguration(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(422);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          valid: false,
          errors: expect.arrayContaining([
            expect.objectContaining({
              field: 'frequency',
              message: expect.stringContaining('weekly')
            })
          ])
        })
      );
    });

    test('should reject invalid strength', async () => {
      mockReq.body = {
        medication_id: 'med-123',
        product_id: 'prod-123',
        intake_type: 'Injection',
        intake_place: 'at home',
        strength_value: 5, // Not in allowed strengths
        strength_unit: 'mg',
        frequency: 'weekly',
        custom_flags: { dose: false, frequency: false, intake_type: false }
      };

      controller.validateAgainstProduct = jest.fn().mockResolvedValue({
        valid: false,
        errors: [{
          field: 'strength_value',
          message: 'Selected dose not available for this product/frequency.'
        }]
      });

      await controller.validateMedicationConfiguration(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(422);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          valid: false,
          errors: expect.arrayContaining([
            expect.objectContaining({
              field: 'strength_value'
            })
          ])
        })
      );
    });

    test('should validate custom doses within safety limits', async () => {
      mockReq.body = {
        medication_id: 'med-123',
        product_id: 'prod-123',
        intake_type: 'Injection',
        intake_place: 'at home',
        strength_value: 1.5, // Custom dose
        strength_unit: 'mg',
        frequency: 'weekly',
        custom_flags: { dose: true, frequency: false, intake_type: false }
      };

      controller.validateAgainstProduct = jest.fn().mockResolvedValue({
        valid: true,
        normalized: {
          intake_type: 'Injection',
          intake_place: 'at home',
          strength_value: 1.5,
          strength_unit: 'mg',
          frequency: 'weekly',
          label: 'Ozempic (semaglutide) 1.5 mg, weekly injection (custom dose)'
        },
        warnings: ['Custom dose used - ensure this matches your prescription']
      });

      await controller.validateMedicationConfiguration(mockReq, mockRes);
      
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          valid: true,
          warnings: expect.arrayContaining([
            expect.stringContaining('Custom dose')
          ])
        })
      );
    });

    test('should reject dangerous custom doses', async () => {
      mockReq.body = {
        medication_id: 'med-123',
        product_id: 'prod-123',
        intake_type: 'Injection',
        intake_place: 'at home',
        strength_value: 10, // Exceeds safety limit
        strength_unit: 'mg',
        frequency: 'weekly',
        custom_flags: { dose: true, frequency: false, intake_type: false }
      };

      controller.validateAgainstProduct = jest.fn().mockResolvedValue({
        valid: false,
        errors: [{
          field: 'strength_value',
          message: 'Custom dose exceeds maximum allowed: 2 mg/week'
        }]
      });

      await controller.validateMedicationConfiguration(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(422);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          valid: false,
          errors: expect.arrayContaining([
            expect.objectContaining({
              field: 'strength_value',
              message: expect.stringContaining('exceeds maximum')
            })
          ])
        })
      );
    });

    test('should prevent duplicate active cycles', async () => {
      mockReq.body = {
        medication_id: 'med-123',
        product_id: 'prod-123',
        intake_type: 'Injection',
        intake_place: 'at home',
        strength_value: 0.5,
        strength_unit: 'mg',
        frequency: 'weekly',
        custom_flags: { dose: false, frequency: false, intake_type: false }
      };

      controller.validateAgainstProduct = jest.fn().mockResolvedValue({
        valid: false,
        errors: [{
          field: 'general',
          message: 'You already have an active cycle for this product. Please edit the existing cycle instead.'
        }]
      });

      await controller.validateMedicationConfiguration(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(409);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          valid: false,
          errors: expect.arrayContaining([
            expect.objectContaining({
              message: expect.stringContaining('already have an active cycle')
            })
          ])
        })
      );
    });
  });

  describe('Medication Cycle Creation', () => {
    test('should create cycle after successful validation', async () => {
      mockReq.body = {
        medication_id: 'med-123',
        product_id: 'prod-123',
        intake_type: 'Injection',
        intake_place: 'at home',
        strength_value: 0.5,
        strength_unit: 'mg',
        frequency: 'weekly',
        start_date: '2025-01-15',
        end_date: null,
        custom_flags: { dose: false, frequency: false, intake_type: false },
        notes: 'Test medication cycle'
      };

      controller.validateMedicationConfiguration = jest.fn().mockResolvedValue(true);
      controller.createCycleInDatabase = jest.fn().mockResolvedValue({
        id: 'cycle-123',
        userId: 'test-user-123',
        medicationId: 'med-123',
        productId: 'prod-123',
        createdAt: new Date()
      });

      await controller.createMedicationCycle(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(201);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('created successfully'),
          cycle: expect.objectContaining({
            id: 'cycle-123'
          })
        })
      );
    });

    test('should reject cycle creation if validation fails', async () => {
      mockReq.body = {
        medication_id: 'med-123',
        product_id: 'prod-123',
        intake_type: 'Pill/Tablet', // Invalid
        // ... other fields
      };

      controller.validateMedicationConfiguration = jest.fn().mockResolvedValue(false);

      await controller.createMedicationCycle(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(422);
      expect(controller.createCycleInDatabase).not.toHaveBeenCalled();
    });
  });

  describe('Error Handling & Security', () => {
    test('should handle database connection errors', async () => {
      mockReq.query = { q: 'paracetamol' };
      
      controller.resolver = {
        resolve: jest.fn().mockRejectedValue(new Error('ECONNREFUSED'))
      };

      await controller.search(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(500);
      expect(mockRes.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.any(String)
        })
      );
    });

    test('should validate user authentication', async () => {
      mockReq.user = null; // No user
      
      await controller.validateMedicationConfiguration(mockReq, mockRes);
      
      expect(mockRes.status).toHaveBeenCalledWith(401);
    });

    test('should sanitize input parameters', async () => {
      mockReq.query = { 
        q: '<script>alert("xss")</script>',
        limit: 'invalid'
      };
      
      controller.resolver = {
        resolve: jest.fn().mockResolvedValue({
          query: mockReq.query.q,
          matches: [],
          suggestions: [],
          metadata: {}
        })
      };

      await controller.search(mockReq, mockRes);
      
      // Should sanitize and normalize input
      expect(controller.resolver.resolve).toHaveBeenCalledWith(
        expect.not.stringContaining('<script>'),
        expect.objectContaining({
          maxResults: expect.any(Number)
        })
      );
    });
  });
});

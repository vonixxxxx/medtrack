const { PrismaClient } = require('@prisma/client');
const NHSMedicinesService = require('../services/nhsMedicinesService');
const HospitalGradeMedicationController = require('./HospitalGradeMedicationController');
const AuditLogger = require('../services/AuditLogger');
const SafetyMonitor = require('../services/SafetyMonitor');

const prisma = new PrismaClient();
const nhsService = new NHSMedicinesService();
const hospitalController = new HospitalGradeMedicationController();
const auditLogger = new AuditLogger();
const safetyMonitor = new SafetyMonitor();

// Initialize safety monitoring
safetyMonitor.initialize().catch(error => {
  console.error('Failed to initialize safety monitor:', error);
});

// Helper function to parse JSON strings safely
const parseJsonSafely = (jsonString, defaultValue = []) => {
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    console.warn('Failed to parse JSON string:', jsonString);
    return defaultValue;
  }
};

// Helper function to calculate similarity score for fuzzy matching
const calculateSimilarity = (str1, str2) => {
  const longer = str1.length > str2.length ? str1 : str2;
  const shorter = str1.length > str2.length ? str2 : str1;
  
  if (longer.length === 0) return 1.0;
  if (longer.includes(shorter)) return shorter.length / longer.length;
  
  let commonPrefix = 0;
  for (let i = 0; i < Math.min(str1.length, str2.length); i++) {
    if (str1[i] === str2[i]) {
      commonPrefix++;
    } else {
      break;
    }
  }
  
  let commonSuffix = 0;
  for (let i = 1; i <= Math.min(str1.length, str2.length); i++) {
    if (str1[str1.length - i] === str2[str1.length - i]) {
      commonSuffix++;
    } else {
      break;
    }
  }
  
  return (commonPrefix + commonSuffix) / (str1.length + str2.length);
};

// Enhanced search with NHS integration
exports.searchMedications = async (req, res) => {
  const startTime = Date.now();
  const userId = req.user?.id || 'anonymous';
  
  try {
    const { q } = req.query;
    
    if (!q || q.trim().length < 2) {
      await auditLogger.logSecurityEvent(userId, 'INVALID_SEARCH_QUERY', {
        query: q,
        reason: 'Query too short or empty'
      }, {
        clientIP: req.ip,
        userAgent: req.get('User-Agent')
      });
      
      return res.status(400).json({
        error: 'Search query must be at least 2 characters long'
      });
    }

    const query = q.trim();
    console.log(`🔍 NHS Enhanced Search for: "${query}"`);

    // Use NHS service for comprehensive search
    const nhsResults = await nhsService.searchMedications(query);
    
    // Transform NHS results to match expected format
    const transformedMatches = nhsResults.matches.map(med => ({
      id: med.id,
      genericName: med.genericName,
      classHuman: med.atcClass,
      reason: med.reason || 'nhs_match',
      score: med.score || 100,
      products: med.products.map(product => ({
        id: product.id,
        brandName: product.brandName,
        allowedIntakeType: product.allowedIntakeType,
        route: product.route,
        form: product.form
      }))
    }));

    // Add suggestions for no results
    const suggestions = nhsResults.suggestions || [];
    
    const response = {
      query,
      matches: transformedMatches,
      suggestions,
      total: transformedMatches.length,
      source: 'NHS Medicines A-Z'
    };

    // Log successful search
    await auditLogger.logMedicationSearch(userId, { query }, response, {
      executionTime: Date.now() - startTime,
      clientIP: req.ip,
      userAgent: req.get('User-Agent')
    });

    res.json(response);

  } catch (error) {
    console.error('❌ NHS Enhanced Search failed:', error);
    
    // Log search error
    await auditLogger.logSystemError(userId, error, {
      component: 'medicationController',
      operation: 'searchMedications',
      query
    }, {
      clientIP: req.ip,
      userAgent: req.get('User-Agent')
    });
    
    // Fallback to existing search if NHS service fails
    try {
      const fallbackResults = await performFallbackSearch(query);
      res.json(fallbackResults);
    } catch (fallbackError) {
      console.error('❌ Fallback search also failed:', fallbackError);
      res.status(500).json({
        error: 'Search service temporarily unavailable. Please try again.',
        suggestions: ['paracetamol', 'ibuprofen', 'aspirin', 'metformin']
      });
    }
  }
};

// Fallback search implementation
async function performFallbackSearch(query) {
  const normalizedQuery = query.toLowerCase();
  
  // Stage A: Exact/Prefix matches
  const exactMatches = await prisma.medicationValidation.findMany({
    where: {
      OR: [
        { genericName: { contains: normalizedQuery } },
        {
          products: {
            some: {
              brandName: { contains: normalizedQuery }
            }
          }
        }
      ]
    },
    include: {
      products: {
        where: { isActive: true },
        select: {
          id: true,
          brandName: true,
          allowedIntakeType: true,
          route: true,
          form: true
        }
      }
    }
  });

  // Stage B: Synonyms/Acronyms/Class matches
  const synonymMatches = await prisma.medicationValidation.findMany({
    where: {
      OR: [
        {
          synonyms: {
            contains: normalizedQuery
          }
        },
        {
          classHuman: {
            contains: normalizedQuery
          }
        }
      ]
    },
    include: {
      products: {
        where: { isActive: true },
        select: {
          id: true,
          brandName: true,
          allowedIntakeType: true,
          route: true,
          form: true
        }
      }
    }
  });

  // Stage C: Fuzzy matches
  const allMedications = await prisma.medicationValidation.findMany({
    include: {
      products: {
        where: { isActive: true },
        select: {
          id: true,
          brandName: true,
          allowedIntakeType: true,
          route: true,
          form: true
        }
      }
    }
  });

  const fuzzyMatches = allMedications
    .map(med => {
      const searchTerms = [
        med.genericName,
        ...parseJsonSafely(med.synonyms),
        ...med.products.map(p => p.brandName)
      ];
      
      let maxSimilarity = 0;
      let bestMatch = null;
      
      for (const term of searchTerms) {
        if (term) {
          const similarity = calculateSimilarity(normalizedQuery, term.toLowerCase());
          if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            bestMatch = term;
          }
        }
      }
      
      if (maxSimilarity > 0.6) {
        return {
          ...med,
          reason: 'fuzzy',
          score: Math.round(maxSimilarity * 60),
          matchedTerm: bestMatch
        };
      }
      return null;
    })
    .filter(Boolean);

  // Combine and rank all results
  const allMatches = [
    ...exactMatches.map(m => ({ ...m, reason: 'exact', score: 100 })),
    ...synonymMatches.map(m => ({ ...m, reason: 'synonym', score: 80 })),
    ...fuzzyMatches
  ];

  // Remove duplicates and rank by score
  const uniqueMatches = allMatches
    .filter((match, index, self) => 
      index === self.findIndex(m => m.id === match.id)
    )
    .sort((a, b) => b.score - a.score);

  // Generate suggestions if no results
  const suggestions = uniqueMatches.length === 0 ? 
    await generateSuggestions(normalizedQuery) : [];

  return {
    query: normalizedQuery,
    matches: uniqueMatches,
    suggestions,
    total: uniqueMatches.length,
    source: 'Fallback Database'
  };
}

// Generate search suggestions
async function generateSuggestions(query) {
  try {
    // Find similar medications
    const medications = await prisma.medicationValidation.findMany({
      take: 10,
      orderBy: {
        genericName: 'asc'
      }
    });
    
    const suggestions = [];
    
    for (const med of medications) {
      const distance = calculateLevenshteinDistance(query, med.genericName.toLowerCase());
      if (distance <= 3) { // Close matches
        suggestions.push(med.genericName);
      }
    }
    
    // Add common medications if no close matches
    if (suggestions.length === 0) {
      const commonMeds = [
        'paracetamol', 'ibuprofen', 'aspirin', 'metformin', 'semaglutide',
        'insulin', 'atorvastatin', 'omeprazole', 'amoxicillin', 'warfarin'
      ];
      
      suggestions.push(...commonMeds.slice(0, 5));
    }
    
    return suggestions.slice(0, 5);
  } catch (error) {
    console.warn('⚠️ Failed to get suggestions:', error.message);
    return ['paracetamol', 'ibuprofen', 'aspirin'];
  }
}

// Calculate Levenshtein distance
function calculateLevenshteinDistance(str1, str2) {
  const matrix = [];
  
  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }
  
  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }
  
  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }
  
  return matrix[str2.length][str1.length];
}

// Get product options for UI rendering with NHS integration
exports.getProductOptions = async (req, res) => {
  try {
    const { productId } = req.params;
    
    if (!productId) {
      return res.status(400).json({
        error: 'Product ID is required'
      });
    }

    console.log(`🔍 Getting NHS Enhanced Product Options for: ${productId}`);

    // Try NHS service first
    try {
      const nhsOptions = await nhsService.getProductOptions(productId);
      
      // Enhance with additional metadata
      const enhancedOptions = {
        ...nhsOptions,
        source: 'NHS Medicines A-Z',
        enhanced: true,
        metadata: {
          route: nhsOptions.route || 'oral',
          form: nhsOptions.form || 'tablet',
          notes: nhsOptions.notes || []
        }
      };

      res.json(enhancedOptions);
      return;
    } catch (nhsError) {
      console.warn('⚠️ NHS service failed, falling back to database:', nhsError.message);
    }

    // Fallback to database
    const product = await prisma.medicationProduct.findUnique({
      where: { id: productId },
      include: {
        medication: true,
        strengths: {
          where: { isActive: true },
          orderBy: { strengthValue: 'asc' }
        },
        validationRules: {
          orderBy: { version: 'desc' },
          take: 1
        }
      }
    });

    if (!product) {
      return res.status(404).json({
        error: 'Product not found'
      });
    }

    if (!product.isActive) {
      return res.status(400).json({
        error: 'Product is no longer active'
      });
    }

    // Parse JSON fields
    const defaultPlaces = parseJsonSafely(product.defaultPlaces);
    const allowedFrequencies = parseJsonSafely(product.allowedFrequencies);

    // Format strengths with proper labels
    const strengths = product.strengths.map(strength => ({
      value: strength.strengthValue,
      unit: strength.strengthUnit,
      frequency: strength.frequency,
      label: strength.label || `${strength.strengthValue} ${strength.strengthUnit} ${strength.frequency}`
    }));

    // Get latest validation rules
    const latestRules = product.validationRules[0] || {};

    const options = {
      product_id: product.id,
      brand_name: product.brandName,
      generic_name: product.medication.genericName,
      allowed_intake_type: product.allowedIntakeType,
      allowed_frequencies: allowedFrequencies,
      default_places: defaultPlaces,
      strengths,
      rules: {
        max_dose_per_period: latestRules.maxDosePerPeriod,
        min_dose_per_period: latestRules.minDosePerPeriod,
        contraindications: latestRules.contraindications ? parseJsonSafely(latestRules.contraindications) : [],
        warnings: latestRules.warnings ? parseJsonSafely(latestRules.warnings) : []
      },
      allow_custom: true, // Allow custom values with proper validation
      metadata: {
        route: product.route,
        form: product.form,
        notes: product.notes
      },
      source: 'Database Fallback',
      enhanced: false
    };

    res.json(options);

  } catch (error) {
    console.error('Product options error:', error);
    res.status(500).json({
      error: 'Failed to get product options. Please try again.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Server-side hard validation with NHS integration
exports.validateMedication = async (req, res) => {
  const startTime = Date.now();
  const userId = req.user?.id || 'anonymous';
  
  try {
    const {
      medication_id,
      product_id,
      intake_type,
      intake_place,
      strength_value,
      strength_unit,
      frequency,
      custom_flags = {}
    } = req.body;

    // Validate required fields
    if (!medication_id || !product_id || !intake_type || !intake_place || !strength_value || !strength_unit || !frequency) {
      const error = {
        error: 'Missing required fields',
        required: ['medication_id', 'product_id', 'intake_type', 'intake_place', 'strength_value', 'strength_unit', 'frequency']
      };

      await auditLogger.logSecurityEvent(userId, 'INVALID_VALIDATION_REQUEST', {
        missingFields: error.required,
        providedFields: Object.keys(req.body)
      }, {
        clientIP: req.ip,
        userAgent: req.get('User-Agent')
      });

      return res.status(400).json(error);
    }

    console.log(`🔍 NHS Enhanced Validation for product: ${product_id}`);

    // Try NHS service first for enhanced validation
    try {
      const nhsValidation = await nhsService.validateMedicationConfig({
        medication_id,
        product_id,
        intake_type,
        intake_place,
        strength_value,
        strength_unit,
        frequency,
        custom_flags
      });

      // Log validation attempt
      await auditLogger.logMedicationValidation(userId, req.body, nhsValidation, {
        executionTime: Date.now() - startTime,
        clientIP: req.ip,
        userAgent: req.get('User-Agent'),
        source: 'NHS Medicines A-Z'
      });

      // Monitor for safety issues
      await safetyMonitor.monitorValidation(userId, req.body, nhsValidation, {
        clientIP: req.ip,
        userAgent: req.get('User-Agent')
      });

      if (nhsValidation.valid) {
        res.json({
          ...nhsValidation,
          source: 'NHS Medicines A-Z',
          enhanced: true
        });
        return;
      } else {
        // NHS validation failed, return detailed errors
        res.status(422).json({
          ...nhsValidation,
          source: 'NHS Medicines A-Z',
          enhanced: true
        });
        return;
      }
    } catch (nhsError) {
      console.warn('⚠️ NHS validation failed, falling back to database:', nhsError.message);
      
      await auditLogger.logSystemError(userId, nhsError, {
        component: 'nhsService',
        operation: 'validateMedicationConfig'
      });
    }

    // Fallback to database validation
    const product = await prisma.medicationProduct.findUnique({
      where: { id: product_id },
      include: {
        medication: true,
        strengths: {
          where: { isActive: true }
        },
        validationRules: {
          orderBy: { version: 'desc' },
          take: 1
        }
      }
    });

    if (!product) {
      return res.status(404).json({
        error: 'Product not found'
      });
    }

    if (!product.isActive) {
      return res.status(400).json({
        error: 'Product is no longer active'
      });
    }

    const validationErrors = [];
    const warnings = [];

    // 1. Validate intake type
    if (intake_type !== product.allowedIntakeType && !custom_flags.intake_type) {
      validationErrors.push({
        field: 'intake_type',
        message: `This product is ${product.allowedIntakeType} only.`
      });
    }

    // 2. Validate frequency
    const allowedFrequencies = parseJsonSafely(product.allowedFrequencies);
    if (!allowedFrequencies.includes(frequency) && !custom_flags.frequency) {
      validationErrors.push({
        field: 'frequency',
        message: `Allowed frequency for this product is ${allowedFrequencies.join(', ')}.`
      });
    }

    // 3. Validate strength
    const allowedStrength = product.strengths.find(s => 
      s.strengthValue === strength_value && 
      s.strengthUnit === strength_unit && 
      s.frequency === frequency
    );

    if (!allowedStrength && !custom_flags.dose) {
      validationErrors.push({
        field: 'strength_value',
        message: 'Selected dose not available for this product/frequency.'
      });
    }

    // 4. Validate place
    const defaultPlaces = parseJsonSafely(product.defaultPlaces);
    const placeAllowed = defaultPlaces.includes(intake_place) || intake_place.startsWith('custom:');
    if (!placeAllowed) {
      validationErrors.push({
        field: 'intake_place',
        message: 'Choose one of the allowed places or use Custom.'
      });
    }

    // 5. Check custom dose bounds if applicable
    if (custom_flags.dose && product.validationRules[0]) {
      const rules = product.validationRules[0];
      
      if (rules.maxDosePerPeriod) {
        // Simple validation - could be enhanced with more sophisticated parsing
        const maxDose = parseFloat(rules.maxDosePerPeriod.split(' ')[0]);
        if (strength_value > maxDose) {
          validationErrors.push({
            field: 'strength_value',
            message: `Custom dose exceeds maximum allowed: ${rules.maxDosePerPeriod}`
          });
        }
      }
    }

    // 6. Check for duplicate active cycles (if user is authenticated)
    if (req.user) {
      const existingCycle = await prisma.userMedicationCycle.findFirst({
        where: {
          userId: req.user.id,
          productId: product_id,
          isActive: true,
          OR: [
            { endDate: null },
            { endDate: { gte: new Date() } }
          ]
        }
      });

      if (existingCycle) {
        validationErrors.push({
          field: 'general',
          message: 'You already have an active cycle for this product. Please edit the existing cycle instead.'
        });
      }
    }

    // Return validation result
    if (validationErrors.length > 0) {
      return res.status(422).json({
        valid: false,
        errors: validationErrors,
        suggested_options_endpoint: `/api/meds/product/${product_id}/options`
      });
    }

    // Validation passed - return normalized data
    const normalized = {
      intake_type: intake_type,
      intake_place: intake_place,
      strength_value: strength_value,
      strength_unit: strength_unit,
      frequency: frequency,
      label: `${product.brandName} (${product.medication.genericName}) ${strength_value} ${strength_unit}, ${frequency} ${intake_type.toLowerCase()}`
    };

    // Add warnings for custom values
    if (custom_flags.dose) {
      warnings.push('Custom dose used - ensure this matches your prescription');
    }
    if (custom_flags.frequency) {
      warnings.push('Custom frequency used - ensure this matches your prescription');
    }
    if (custom_flags.intake_type) {
      warnings.push('Custom intake type used - this may affect medication effectiveness');
    }

    const validationResult = {
      valid: true,
      normalized,
      rule_version: product.validationRules[0]?.version || 1,
      warnings,
      source: 'Database Fallback'
    };

    // Log successful validation
    await auditLogger.logMedicationValidation(userId, req.body, validationResult, {
      executionTime: Date.now() - startTime,
      clientIP: req.ip,
      userAgent: req.get('User-Agent'),
      source: 'Database Fallback'
    });

    // Monitor for safety issues
    await safetyMonitor.monitorValidation(userId, req.body, validationResult, {
      clientIP: req.ip,
      userAgent: req.get('User-Agent')
    });

    res.json(validationResult);

  } catch (error) {
    console.error('Validation error:', error);
    
    // Log validation error
    await auditLogger.logSystemError(userId, error, {
      component: 'medicationController',
      operation: 'validateMedication'
    });

    res.status(500).json({
      error: 'Validation failed. Please try again.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Create medication cycle
exports.createMedicationCycle = async (req, res) => {
  const startTime = Date.now();
  const userId = req.user?.id;
  
  try {
    if (!userId) {
      await auditLogger.logSecurityEvent('anonymous', 'UNAUTHORIZED_CYCLE_CREATION', {
        reason: 'No user ID in request'
      }, {
        clientIP: req.ip,
        userAgent: req.get('User-Agent')
      });
      
      return res.status(401).json({
        error: 'Authentication required'
      });
    }

    // First validate the medication configuration
    const validationResult = await this.validateMedication(req, res);
    
    // If validation failed, the response has already been sent
    if (validationResult && !validationResult.valid) {
      return; // Response already sent by validateMedication
    }
    
    const {
      medication_id,
      product_id,
      strength_value,
      strength_unit,
      frequency,
      intake_type,
      intake_place,
      start_date,
      end_date,
      custom_flags,
      notes
    } = req.body;

    // Create the medication cycle
    const medicationCycle = await prisma.userMedicationCycle.create({
      data: {
        userId,
        medicationId: medication_id,
        productId: product_id,
        strengthValue: strength_value,
        strengthUnit: strength_unit,
        frequency,
        intakeType: intake_type,
        intakePlace: intake_place,
        startDate: new Date(start_date),
        endDate: end_date ? new Date(end_date) : null,
        customFlags: JSON.stringify(custom_flags || {}),
        notes,
        isActive: true
      },
      include: {
        medication: true,
        product: true
      }
    });

    const response = {
      message: 'Medication cycle created successfully',
      cycle: medicationCycle
    };

    // Log successful cycle creation
    await auditLogger.logMedicationCycle(userId, 'create', req.body, response, {
      executionTime: Date.now() - startTime,
      clientIP: req.ip,
      userAgent: req.get('User-Agent')
    });

    // Monitor for safety issues
    await safetyMonitor.monitorCycle(userId, 'create', req.body, response, {
      clientIP: req.ip,
      userAgent: req.get('User-Agent')
    });

    res.status(201).json(response);

  } catch (error) {
    console.error('Create cycle error:', error);
    
    // Log cycle creation error
    await auditLogger.logSystemError(userId, error, {
      component: 'medicationController',
      operation: 'createMedicationCycle'
    });

    res.status(500).json({
      error: 'Failed to create medication cycle. Please try again.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Get user medication cycles
exports.getUserMedicationCycles = async (req, res) => {
  try {
    const userId = req.user.id;
    
    const cycles = await prisma.userMedicationCycle.findMany({
      where: { userId },
      include: {
        medication: true,
        product: true
      },
      orderBy: { createdAt: 'desc' }
    });

    res.json({
      cycles: cycles.map(cycle => ({
        ...cycle,
        customFlags: parseJsonSafely(cycle.customFlags)
      }))
    });

  } catch (error) {
    console.error('Get cycles error:', error);
    res.status(500).json({
      error: 'Failed to get medication cycles. Please try again.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Update medication cycle
exports.updateMedicationCycle = async (req, res) => {
  try {
    const { cycleId } = req.params;
    const userId = req.user.id;
    
    // Verify ownership
    const existingCycle = await prisma.userMedicationCycle.findFirst({
      where: { id: cycleId, userId }
    });

    if (!existingCycle) {
      return res.status(404).json({
        error: 'Medication cycle not found'
      });
    }

    const updatedCycle = await prisma.userMedicationCycle.update({
      where: { id: cycleId },
      data: {
        ...req.body,
        updatedAt: new Date()
      },
      include: {
        medication: true,
        product: true
      }
    });

    res.json({
      message: 'Medication cycle updated successfully',
      cycle: updatedCycle
    });

  } catch (error) {
    console.error('Update cycle error:', error);
    res.status(500).json({
      error: 'Failed to update medication cycle. Please try again.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Delete medication cycle
exports.deleteMedicationCycle = async (req, res) => {
  try {
    const { cycleId } = req.params;
    const userId = req.user.id;
    
    // Verify ownership
    const existingCycle = await prisma.userMedicationCycle.findFirst({
      where: { id: cycleId, userId }
    });

    if (!existingCycle) {
      return res.status(404).json({
        error: 'Medication cycle not found'
      });
    }

    await prisma.userMedicationCycle.delete({
      where: { id: cycleId }
    });

    res.json({
      message: 'Medication cycle deleted successfully'
    });

  } catch (error) {
    console.error('Delete cycle error:', error);
    res.status(500).json({
      error: 'Failed to delete medication cycle. Please try again.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Ollama suggestions (advisory only)
exports.getOllamaSuggestions = async (req, res) => {
  try {
    const { medication_id, product_id, context } = req.body;
    
    // This is a placeholder for future Ollama integration
    // For now, return basic advisory information
    res.json({
      suggestions: {
        monitoring_metrics: ['Blood Sugar', 'Weight', 'Side Effects'],
        counseling: 'Always follow your healthcare provider\'s instructions',
        reminders: 'Set reminders for consistent dosing schedule'
      },
      disclaimer: 'These are general suggestions only. Always consult your healthcare provider for specific advice.'
    });

  } catch (error) {
    console.error('Ollama suggestions error:', error);
    res.status(500).json({
      error: 'Failed to get suggestions. Please try again.'
    });
  }
};

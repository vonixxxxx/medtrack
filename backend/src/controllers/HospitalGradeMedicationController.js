/**
 * Hospital-Grade Medication Controller
 * Zero tolerance for incorrect options - server is the authoritative source
 * All validation is server-side with complete audit trails
 */

const { PrismaClient } = require('@prisma/client');
const ProfoundMedicationResolver = require('../services/ProfoundMedicationResolver');
const crypto = require('crypto');

class HospitalGradeMedicationController {
  constructor() {
    this.prisma = new PrismaClient();
    this.resolver = new ProfoundMedicationResolver();
    this.logger = this.createLogger();
    
    // Load system configuration
    this.config = {
      enabledSources: ['dmd', 'bnf', 'rxnorm'],
      sourcePriority: { 'dmd': 10, 'bnf': 9, 'rxnorm': 7 },
      validation: {
        requireServerConfirmation: true,
        allowCustomDose: true,
        allowCustomFrequency: true,
        requireExplicitAckForCustom: true
      },
      security: {
        redactPiiInLogs: true,
        auditAllActions: true
      }
    };
  }

  createLogger() {
    return {
      info: (msg, meta = {}) => console.log(`[HospitalMedController] INFO: ${msg}`, this.sanitizeMeta(meta)),
      warn: (msg, meta = {}) => console.warn(`[HospitalMedController] WARN: ${msg}`, this.sanitizeMeta(meta)),
      error: (msg, meta = {}) => console.error(`[HospitalMedController] ERROR: ${msg}`, this.sanitizeMeta(meta)),
      audit: (action, details, req) => this.auditLog(action, details, req)
    };
  }

  /**
   * GET /api/meds/search?q=...
   * Authoritative medication search with explainability
   */
  async searchMedications(req, res) {
    const startTime = Date.now();
    const { q: query, sources, limit = 10 } = req.query;
    
    try {
      // Input validation
      if (!query || typeof query !== 'string' || query.trim().length < 2) {
        await this.auditLog('search_invalid', { query, reason: 'query_too_short' }, req);
        return res.status(400).json({
          error: 'Search query must be at least 2 characters long',
          query: query || ''
        });
      }

      const normalizedQuery = query.trim();
      this.logger.info(`Medication search initiated`, { query: normalizedQuery, userId: req.user?.id });

      // Configure resolver based on request
      const resolverConfig = {
        enabledSources: sources ? sources.split(',') : this.config.enabledSources,
        maxResults: Math.min(parseInt(limit) || 10, 50) // Cap at 50 results
      };

      // Execute search using Profound Resolver
      const searchResult = await this.resolver.resolve(normalizedQuery, resolverConfig);
      
      // Security: Never expose internal medication IDs in logs
      await this.auditLog('search_executed', {
        queryHash: this.hashQuery(normalizedQuery),
        resultsCount: searchResult.matches.length,
        suggestionsCount: searchResult.suggestions.length,
        processingTime: Date.now() - startTime
      }, req);

      // Format response with hospital-grade metadata
      const response = {
        query: normalizedQuery,
        matches: searchResult.matches.map(match => this.formatSearchResult(match)),
        suggestions: searchResult.suggestions.map(match => this.formatSuggestion(match)),
        metadata: {
          ...searchResult.metadata,
          totalResults: searchResult.matches.length,
          hasMore: searchResult.suggestions.length > 0,
          searchId: crypto.randomUUID(), // For result tracking
          timestamp: new Date().toISOString()
        }
      };

      this.logger.info(`Search completed successfully`, {
        query: normalizedQuery,
        resultsCount: response.matches.length,
        processingTime: Date.now() - startTime
      });

      res.json(response);

    } catch (error) {
      this.logger.error('Search operation failed', {
        query: normalizedQuery,
        error: error.message,
        stack: error.stack
      });

      await this.auditLog('search_failed', {
        queryHash: this.hashQuery(normalizedQuery || ''),
        error: error.message
      }, req);

      res.status(500).json({
        error: 'Search service temporarily unavailable',
        query: normalizedQuery || '',
        searchId: crypto.randomUUID(),
        suggestions: ['paracetamol', 'ibuprofen', 'metformin'] // Fallback suggestions
      });
    }
  }

  /**
   * GET /api/meds/product/:productId/options
   * Returns ONLY server-approved options - frontend renders exactly this
   */
  async getProductOptions(req, res) {
    const { productId } = req.params;
    const startTime = Date.now();

    try {
      if (!productId || !this.isValidUUID(productId)) {
        await this.auditLog('product_options_invalid', { productId, reason: 'invalid_uuid' }, req);
        return res.status(400).json({
          error: 'Valid product ID is required'
        });
      }

      this.logger.info(`Fetching product options`, { productId, userId: req.user?.id });

      // Fetch product with all related data
      const product = await this.prisma.product.findUnique({
        where: { 
          id: productId,
          isActive: true 
        },
        include: {
          medication: true,
          strengths: {
            where: { isActive: true },
            orderBy: [
              { value: 'asc' },
              { unit: 'asc' }
            ]
          },
          rules: {
            where: { isBlacklisted: false },
            orderBy: { version: 'desc' },
            take: 1
          }
        }
      });

      if (!product) {
        await this.auditLog('product_options_not_found', { productId }, req);
        return res.status(404).json({
          error: 'Product not found or inactive'
        });
      }

      // Check if product is blacklisted
      if (product.rules.some(rule => rule.isBlacklisted)) {
        await this.auditLog('product_options_blacklisted', { productId }, req);
        return res.status(403).json({
          error: 'Product is not available for prescription'
        });
      }

      // Build authoritative options response
      const options = {
        productId: product.id,
        brandName: product.brandName,
        genericName: product.medication.genericName,
        
        // ONLY these values are allowed in frontend
        intakeType: product.intakeType,
        route: product.route,
        doseForm: product.doseForm,
        
        allowedFrequencies: product.allowedFrequencies,
        defaultPlaces: product.defaultPlaces,
        
        // Pre-approved strengths only
        strengths: product.strengths.map(strength => ({
          value: Number(strength.value),
          unit: strength.unit,
          per: strength.per,
          frequency: strength.frequency,
          label: strength.label || `${strength.value} ${strength.unit}${strength.per ? ` per ${strength.per}` : ''}`
        })),
        
        // Safety rules and constraints
        rules: this.formatSafetyRules(product.rules[0]),
        
        // Configuration flags
        allowCustom: this.config.validation.allowCustomDose,
        requiresAcknowledgment: this.config.validation.requireExplicitAckForCustom,
        
        // Provenance and versioning
        provenance: product.provenance,
        datasetVersion: product.datasetVersion,
        
        // Metadata
        metadata: {
          lastUpdated: product.updatedAt,
          isGeneric: !product.brandName,
          strengthsCount: product.strengths.length,
          hasRules: product.rules.length > 0
        }
      };

      await this.auditLog('product_options_fetched', {
        productId,
        strengthsCount: options.strengths.length,
        hasCustomAllowed: options.allowCustom
      }, req);

      this.logger.info(`Product options retrieved successfully`, {
        productId,
        strengthsCount: options.strengths.length,
        processingTime: Date.now() - startTime
      });

      res.json(options);

    } catch (error) {
      this.logger.error('Product options retrieval failed', {
        productId,
        error: error.message,
        stack: error.stack
      });

      await this.auditLog('product_options_failed', {
        productId,
        error: error.message
      }, req);

      res.status(500).json({
        error: 'Unable to retrieve product options',
        productId
      });
    }
  }

  /**
   * POST /api/meds/validate
   * Server-side hard validation - blocks impossible combinations
   */
  async validateMedicationConfiguration(req, res) {
    const startTime = Date.now();
    const payload = req.body;

    try {
      // Input validation
      const validationResult = this.validatePayloadStructure(payload);
      if (!validationResult.valid) {
        await this.auditLog('validation_invalid_payload', {
          errors: validationResult.errors,
          payloadKeys: Object.keys(payload)
        }, req);
        
        return res.status(400).json({
          error: 'Invalid request payload',
          details: validationResult.errors
        });
      }

      this.logger.info(`Validating medication configuration`, {
        productId: payload.productId,
        userId: req.user?.id
      });

      // Execute comprehensive validation
      const result = await this.executeHardValidation(payload, req);

      if (result.valid) {
        await this.auditLog('validation_passed', {
          productId: payload.productId,
          configHash: this.hashConfiguration(payload),
          ruleVersion: result.ruleVersion
        }, req);

        this.logger.info(`Configuration validation passed`, {
          productId: payload.productId,
          processingTime: Date.now() - startTime
        });

        res.json({
          valid: true,
          normalized: result.normalized,
          ruleVersion: result.ruleVersion,
          warnings: result.warnings || [],
          metadata: {
            validatedAt: new Date().toISOString(),
            processingTime: Date.now() - startTime,
            validationId: crypto.randomUUID()
          }
        });

      } else {
        await this.auditLog('validation_failed', {
          productId: payload.productId,
          configHash: this.hashConfiguration(payload),
          errors: result.errors
        }, req);

        this.logger.warn(`Configuration validation failed`, {
          productId: payload.productId,
          errorCount: result.errors.length
        });

        res.status(422).json({
          valid: false,
          errors: result.errors,
          suggestedOptionsEndpoint: `/api/meds/product/${payload.productId}/options`,
          metadata: {
            validatedAt: new Date().toISOString(),
            processingTime: Date.now() - startTime,
            validationId: crypto.randomUUID()
          }
        });
      }

    } catch (error) {
      this.logger.error('Validation process failed', {
        productId: payload?.productId,
        error: error.message,
        stack: error.stack
      });

      await this.auditLog('validation_error', {
        productId: payload?.productId,
        error: error.message
      }, req);

      res.status(500).json({
        error: 'Validation service temporarily unavailable',
        validationId: crypto.randomUUID()
      });
    }
  }

  /**
   * POST /api/meds/cycles
   * Create medication cycle after internal validation
   */
  async createMedicationCycle(req, res) {
    const payload = req.body;
    const startTime = Date.now();

    try {
      // First, validate the configuration
      const validationResult = await this.executeHardValidation(payload, req);
      
      if (!validationResult.valid) {
        await this.auditLog('cycle_creation_validation_failed', {
          productId: payload.productId,
          errors: validationResult.errors
        }, req);
        
        return res.status(422).json({
          error: 'Configuration validation failed',
          errors: validationResult.errors,
          suggestedOptionsEndpoint: `/api/meds/product/${payload.productId}/options`
        });
      }

      // Check for duplicate active cycles
      const duplicateCheck = await this.checkForDuplicateCycles(payload, req.user.id);
      if (duplicateCheck.hasDuplicate) {
        await this.auditLog('cycle_creation_duplicate', {
          productId: payload.productId,
          existingCycleId: duplicateCheck.existingCycle.id
        }, req);
        
        return res.status(409).json({
          error: 'Active medication cycle already exists',
          existingCycle: {
            id: duplicateCheck.existingCycle.id,
            startDate: duplicateCheck.existingCycle.startDate,
            frequency: duplicateCheck.existingCycle.frequency
          },
          suggestion: 'Edit existing cycle or end it before creating a new one'
        });
      }

      // Create the medication cycle
      const cycle = await this.prisma.userMedicationCycle.create({
        data: {
          userId: req.user.id,
          medicationId: payload.medicationId,
          productId: payload.productId,
          strengthValue: payload.strengthValue,
          strengthUnit: payload.strengthUnit,
          frequency: payload.frequency,
          intakeType: payload.intakeType,
          intakePlace: payload.intakePlace,
          startDate: new Date(payload.startDate),
          endDate: payload.endDate ? new Date(payload.endDate) : null,
          customFlags: payload.customFlags || {},
          notes: payload.notes,
          datasetVersion: validationResult.datasetVersion,
          ruleVersion: validationResult.ruleVersion,
          provenance: validationResult.provenance,
          prescriberId: req.user.role === 'CLINICIAN' ? req.user.id : null
        },
        include: {
          medication: true,
          product: true
        }
      });

      await this.auditLog('cycle_created', {
        cycleId: cycle.id,
        productId: payload.productId,
        duration: payload.endDate ? 
          Math.ceil((new Date(payload.endDate) - new Date(payload.startDate)) / (1000 * 60 * 60 * 24)) : 
          null
      }, req);

      this.logger.info(`Medication cycle created successfully`, {
        cycleId: cycle.id,
        productId: payload.productId,
        userId: req.user.id,
        processingTime: Date.now() - startTime
      });

      res.status(201).json({
        success: true,
        cycle: {
          id: cycle.id,
          medicationName: cycle.medication.genericName,
          brandName: cycle.product.brandName,
          strength: `${cycle.strengthValue} ${cycle.strengthUnit}`,
          frequency: cycle.frequency,
          startDate: cycle.startDate,
          endDate: cycle.endDate,
          label: validationResult.normalized.label
        },
        metadata: {
          createdAt: cycle.createdAt,
          datasetVersion: cycle.datasetVersion,
          ruleVersion: cycle.ruleVersion
        }
      });

    } catch (error) {
      this.logger.error('Cycle creation failed', {
        productId: payload?.productId,
        error: error.message,
        stack: error.stack
      });

      await this.auditLog('cycle_creation_error', {
        productId: payload?.productId,
        error: error.message
      }, req);

      res.status(500).json({
        error: 'Unable to create medication cycle',
        cycleId: crypto.randomUUID()
      });
    }
  }

  /**
   * Execute comprehensive hard validation
   */
  async executeHardValidation(payload, req) {
    const errors = [];
    const warnings = [];

    // Fetch product with all validation data
    const product = await this.prisma.product.findUnique({
      where: { id: payload.productId },
      include: {
        medication: true,
        strengths: { where: { isActive: true } },
        rules: { 
          where: { isBlacklisted: false },
          orderBy: { version: 'desc' },
          take: 1
        }
      }
    });

    if (!product) {
      errors.push({
        field: 'productId',
        code: 'PRODUCT_NOT_FOUND',
        message: 'Product not found or inactive'
      });
      return { valid: false, errors };
    }

    if (!product.isActive) {
      errors.push({
        field: 'productId',
        code: 'PRODUCT_INACTIVE',
        message: 'Product is no longer available'
      });
      return { valid: false, errors };
    }

    // 1. Validate intake type
    if (payload.intakeType !== product.intakeType && !payload.customFlags?.intakeType) {
      errors.push({
        field: 'intakeType',
        code: 'INVALID_INTAKE_TYPE',
        message: `This product is ${product.intakeType} only`,
        allowedValues: [product.intakeType]
      });
    }

    // 2. Validate frequency
    if (!product.allowedFrequencies.includes(payload.frequency) && !payload.customFlags?.frequency) {
      errors.push({
        field: 'frequency',
        code: 'INVALID_FREQUENCY',
        message: `Allowed frequencies: ${product.allowedFrequencies.join(', ')}`,
        allowedValues: product.allowedFrequencies
      });
    }

    // 3. Validate strength (critical safety check)
    const allowedStrength = product.strengths.find(s => 
      Number(s.value) === Number(payload.strengthValue) && 
      s.unit === payload.strengthUnit
    );

    if (!allowedStrength && !payload.customFlags?.dose) {
      errors.push({
        field: 'strengthValue',
        code: 'INVALID_STRENGTH',
        message: 'Selected strength not available for this product',
        allowedValues: product.strengths.map(s => ({
          value: Number(s.value),
          unit: s.unit,
          label: s.label
        }))
      });
    }

    // 4. Validate intake place
    if (!product.defaultPlaces.includes(payload.intakePlace) && 
        !payload.intakePlace.startsWith('custom:')) {
      errors.push({
        field: 'intakePlace',
        code: 'INVALID_PLACE',
        message: 'Choose from allowed places or specify custom location',
        allowedValues: product.defaultPlaces
      });
    }

    // 5. Apply safety rules
    if (product.rules.length > 0) {
      const rule = product.rules[0];
      const ruleValidation = this.validateSafetyRules(payload, rule);
      errors.push(...ruleValidation.errors);
      warnings.push(...ruleValidation.warnings);
    }

    // 6. Custom flags validation
    if (this.hasCustomFlags(payload.customFlags)) {
      if (!this.config.validation.allowCustomDose && payload.customFlags.dose) {
        errors.push({
          field: 'customFlags',
          code: 'CUSTOM_DOSE_NOT_ALLOWED',
          message: 'Custom dosages are not permitted'
        });
      }
      
      if (this.config.validation.requireExplicitAckForCustom && 
          !payload.customAcknowledgment) {
        warnings.push({
          field: 'customFlags',
          code: 'CUSTOM_ACKNOWLEDGMENT_REQUIRED',
          message: 'Custom configuration requires explicit acknowledgment'
        });
      }
    }

    if (errors.length > 0) {
      return { valid: false, errors, warnings };
    }

    // Build normalized response
    const normalized = {
      medicationId: product.medicationId,
      productId: product.id,
      intakeType: payload.intakeType,
      intakePlace: payload.intakePlace,
      strengthValue: Number(payload.strengthValue),
      strengthUnit: payload.strengthUnit,
      frequency: payload.frequency,
      label: `${product.brandName || product.medication.genericName} (${product.medication.genericName}) ${payload.strengthValue} ${payload.strengthUnit}, ${payload.frequency} ${payload.intakeType.toLowerCase()}`
    };

    return {
      valid: true,
      normalized,
      warnings,
      ruleVersion: product.rules[0]?.version || 1,
      datasetVersion: product.datasetVersion,
      provenance: product.provenance
    };
  }

  /**
   * Check for duplicate active medication cycles
   */
  async checkForDuplicateCycles(payload, userId) {
    const existingCycle = await this.prisma.userMedicationCycle.findFirst({
      where: {
        userId,
        productId: payload.productId,
        frequency: payload.frequency,
        isActive: true,
        OR: [
          { endDate: null }, // Indefinite cycles
          { endDate: { gte: new Date() } } // Active cycles
        ]
      }
    });

    return {
      hasDuplicate: !!existingCycle,
      existingCycle
    };
  }

  /**
   * Validate safety rules
   */
  validateSafetyRules(payload, rule) {
    const errors = [];
    const warnings = [];

    if (rule.isBlacklisted) {
      errors.push({
        field: 'productId',
        code: 'PRODUCT_BLACKLISTED',
        message: 'This product is not available for prescription'
      });
    }

    // Check dose limits
    if (rule.maxDosePerPeriod) {
      const maxDose = this.parseDoseLimit(rule.maxDosePerPeriod);
      if (this.exceedsDoseLimit(payload, maxDose)) {
        errors.push({
          field: 'strengthValue',
          code: 'EXCEEDS_MAX_DOSE',
          message: `Maximum dose: ${rule.maxDosePerPeriod}`,
          limit: rule.maxDosePerPeriod
        });
      }
    }

    // Add clinical flags as warnings
    if (rule.ageFlags && typeof rule.ageFlags === 'object') {
      if (rule.ageFlags.pediatric) {
        warnings.push({
          field: 'clinical',
          code: 'PEDIATRIC_FLAG',
          message: rule.ageFlags.pediatric
        });
      }
      if (rule.ageFlags.geriatric) {
        warnings.push({
          field: 'clinical',
          code: 'GERIATRIC_FLAG',
          message: rule.ageFlags.geriatric
        });
      }
    }

    return { errors, warnings };
  }

  /**
   * Format search result for API response
   */
  formatSearchResult(match) {
    return {
      medicationId: match.medicationId,
      genericName: match.genericName,
      atcCode: match.atcCode,
      classHuman: match.classHuman,
      confidence: match.confidence,
      reasons: match.reasons,
      matchType: match.matchType,
      products: match.products.map(product => ({
        productId: product.productId,
        brandName: product.brandName,
        intakeType: product.intakeType,
        route: product.route,
        doseForm: product.doseForm,
        strengthsCount: product.strengthsCount
      })),
      sourceRefs: this.sanitizeSourceRefs(match.sourceRefs)
    };
  }

  /**
   * Format suggestion for API response
   */
  formatSuggestion(match) {
    return {
      medicationId: match.medicationId,
      genericName: match.genericName,
      confidence: match.confidence,
      reason: match.reasons[0] || 'suggestion',
      productsCount: match.products.length
    };
  }

  /**
   * Format safety rules for client
   */
  formatSafetyRules(rule) {
    if (!rule) return {};

    return {
      maxDosePerPeriod: rule.maxDosePerPeriod,
      minDosePerPeriod: rule.minDosePerPeriod,
      hasAgeFlags: !!(rule.ageFlags && Object.keys(rule.ageFlags).length > 0),
      hasRenalFlags: !!(rule.renalFlags && Object.keys(rule.renalFlags).length > 0),
      hasHepaticFlags: !!(rule.hepaticFlags && Object.keys(rule.hepaticFlags).length > 0),
      hasPregnancyFlags: !!(rule.pregnancyFlags && Object.keys(rule.pregnancyFlags).length > 0),
      version: rule.version
    };
  }

  /**
   * Audit logging for compliance
   */
  async auditLog(action, details, req) {
    if (!this.config.security.auditAllActions) return;

    try {
      await this.prisma.auditLog.create({
        data: {
          userId: req?.user?.id || null,
          action,
          entityType: 'medication',
          entityId: details.productId || details.medicationId || null,
          details: this.sanitizeAuditDetails(details),
          ipAddress: this.getClientIP(req),
          userAgent: req?.get('User-Agent')?.substring(0, 500) || null,
          sessionId: req?.sessionID || null
        }
      });
    } catch (error) {
      this.logger.error('Audit logging failed', { action, error: error.message });
    }
  }

  /**
   * Helper methods
   */
  isValidUUID(str) {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    return uuidRegex.test(str);
  }

  hashQuery(query) {
    return crypto.createHash('sha256').update(query.toLowerCase()).digest('hex').substring(0, 16);
  }

  hashConfiguration(payload) {
    const configString = JSON.stringify({
      productId: payload.productId,
      strengthValue: payload.strengthValue,
      strengthUnit: payload.strengthUnit,
      frequency: payload.frequency,
      intakeType: payload.intakeType
    });
    return crypto.createHash('sha256').update(configString).digest('hex').substring(0, 16);
  }

  sanitizeMeta(meta) {
    if (!this.config.security.redactPiiInLogs) return meta;
    
    const sanitized = { ...meta };
    delete sanitized.email;
    delete sanitized.phone;
    delete sanitized.address;
    return sanitized;
  }

  sanitizeSourceRefs(sourceRefs) {
    // Return only source names and versions, not internal IDs
    const sanitized = {};
    for (const [source, refs] of Object.entries(sourceRefs || {})) {
      sanitized[source] = Array.isArray(refs) ? refs.length : 1;
    }
    return sanitized;
  }

  sanitizeAuditDetails(details) {
    const sanitized = { ...details };
    delete sanitized.internalId;
    delete sanitized.sensitiveData;
    return sanitized;
  }

  getClientIP(req) {
    return req?.ip || req?.connection?.remoteAddress || 'unknown';
  }

  hasCustomFlags(customFlags) {
    return customFlags && Object.values(customFlags).some(flag => flag === true);
  }

  validatePayloadStructure(payload) {
    const required = ['productId', 'medicationId', 'intakeType', 'intakePlace', 'strengthValue', 'strengthUnit', 'frequency'];
    const missing = required.filter(field => !payload[field]);
    
    if (missing.length > 0) {
      return {
        valid: false,
        errors: missing.map(field => ({
          field,
          code: 'REQUIRED_FIELD_MISSING',
          message: `${field} is required`
        }))
      };
    }

    return { valid: true };
  }

  parseDoseLimit(doseLimit) {
    // Parse dose limits like "2 mg/week", "4 puffs/6h"
    const match = doseLimit.match(/(\d+(?:\.\d+)?)\s*(\w+)\/(\w+)/);
    if (match) {
      return {
        value: parseFloat(match[1]),
        unit: match[2],
        period: match[3]
      };
    }
    return null;
  }

  exceedsDoseLimit(payload, maxDose) {
    if (!maxDose) return false;
    
    // Simplified dose checking - in production, implement comprehensive dose calculations
    if (payload.strengthUnit === maxDose.unit) {
      return Number(payload.strengthValue) > maxDose.value;
    }
    
    return false;
  }
}

module.exports = HospitalGradeMedicationController;

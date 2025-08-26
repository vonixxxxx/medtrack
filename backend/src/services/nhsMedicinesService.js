/**
 * Hospital-Grade NHS Medicines Service
 * Integrates with the comprehensive medication knowledge engine
 */

const ProfoundMedicationResolver = require('./ProfoundMedicationResolver');
const SystemConfigManager = require('../config/SystemConfigManager');

class NHSMedicinesService {
  constructor() {
    this.resolver = new ProfoundMedicationResolver();
    this.configManager = new SystemConfigManager();
    this.initialized = false;
    this.logger = this.createLogger();
  }

  createLogger() {
    return {
      info: (msg, meta = {}) => console.log(`[NHSService] INFO: ${msg}`, meta),
      warn: (msg, meta = {}) => console.warn(`[NHSService] WARN: ${msg}`, meta),
      error: (msg, meta = {}) => console.error(`[NHSService] ERROR: ${msg}`, meta)
    };
  }

  async initialize() {
    if (this.initialized) return;

    try {
      await this.configManager.initialize();
      this.initialized = true;
      this.logger.info('NHS Medicines Service initialized with hospital-grade configuration');
    } catch (error) {
      this.logger.error('Failed to initialize NHS service', { error: error.message });
      throw error;
    }
  }

  /**
   * Hospital-grade medication search
   */
  async searchMedications(query, options = {}) {
    await this.initialize();

    try {
      const searchConfig = this.configManager.getSearchConfig();
      const enabledSources = this.configManager.getEnabledSources();
      
      const resolverOptions = {
        enabledSources,
        maxResults: options.limit || searchConfig.maxResults,
        confidenceThresholds: {
          directSelect: searchConfig.minConfidenceDirect,
          suggestions: searchConfig.minConfidenceSuggest,
          fuzzy: searchConfig.trigramThreshold
        }
      };

      const result = await this.resolver.resolve(query, resolverOptions);
      
      return {
        query: result.query,
        matches: result.matches.map(this.formatMatch.bind(this)),
        suggestions: result.suggestions.map(this.formatSuggestion.bind(this)),
        total: result.matches.length,
        metadata: {
          ...result.metadata,
          nhsEnhanced: true,
          sources: enabledSources
        }
      };

    } catch (error) {
      this.logger.error('NHS search failed', { query, error: error.message });
      throw error;
    }
  }

  /**
   * Get product options with NHS validation
   */
  async getProductOptions(productId) {
    await this.initialize();

    try {
      // Use the hospital-grade controller for authoritative options
      const HospitalController = require('../controllers/HospitalGradeMedicationController');
      const controller = new HospitalController();
      
      // Mock request object for internal use
      const mockReq = { params: { productId }, user: { id: 'system' } };
      let response = null;
      const mockRes = {
        json: (data) => { response = data; },
        status: () => mockRes
      };

      await controller.getProductOptions(mockReq, mockRes);
      
      if (!response) {
        throw new Error('Failed to get product options');
      }

      return {
        ...response,
        nhsValidated: true,
        source: 'NHS Hospital-Grade System'
      };

    } catch (error) {
      this.logger.error('NHS product options failed', { productId, error: error.message });
      throw error;
    }
  }

  /**
   * Validate medication configuration using NHS standards
   */
  async validateMedicationConfig(payload) {
    await this.initialize();

    try {
      const HospitalController = require('../controllers/HospitalGradeMedicationController');
      const controller = new HospitalController();
      
      // Mock request object for internal use
      const mockReq = { body: payload, user: { id: 'system' } };
      let response = null;
      let statusCode = 200;
      
      const mockRes = {
        json: (data) => { response = data; },
        status: (code) => { statusCode = code; return mockRes; }
      };

      await controller.validateMedicationConfiguration(mockReq, mockRes);
      
      if (!response) {
        throw new Error('Validation failed');
      }

      return {
        ...response,
        nhsValidated: true,
        statusCode
      };

    } catch (error) {
      this.logger.error('NHS validation failed', { payload, error: error.message });
      throw error;
    }
  }

  /**
   * Format match for NHS compatibility
   */
  formatMatch(match) {
    return {
      id: match.medicationId,
      genericName: match.genericName,
      atcClass: match.atcCode,
      classHuman: match.classHuman,
      reason: match.matchType,
      score: Math.round(match.confidence * 100),
      products: match.products.map(product => ({
        id: product.productId,
        brandName: product.brandName,
        allowedIntakeType: product.intakeType,
        route: product.route,
        form: product.doseForm
      })),
      nhsMetadata: {
        confidence: match.confidence,
        reasons: match.reasons,
        sourceRefs: match.sourceRefs
      }
    };
  }

  /**
   * Format suggestion for NHS compatibility
   */
  formatSuggestion(suggestion) {
    return {
      genericName: suggestion.genericName,
      confidence: Math.round(suggestion.confidence * 100),
      reason: suggestion.reason
    };
  }

  /**
   * Health check with NHS integration status
   */
  async healthCheck() {
    try {
      await this.initialize();
      
      const config = this.configManager.getSearchConfig();
      const sources = this.configManager.getEnabledSources();
      
      return {
        status: 'healthy',
        nhsIntegrated: true,
        enabledSources: sources,
        configuration: {
          minConfidence: config.minConfidenceDirect,
          maxResults: config.maxResults,
          cacheEnabled: config.cacheEnabled
        },
        lastUpdated: new Date().toISOString()
      };

    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message,
        nhsIntegrated: false,
        lastUpdated: new Date().toISOString()
      };
    }
  }

  async cleanup() {
    await this.resolver.cleanup();
    await this.configManager.cleanup();
  }
}

module.exports = NHSMedicinesService;

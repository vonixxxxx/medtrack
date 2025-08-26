/**
 * Hospital-Grade System Configuration Manager
 * Manages source priorities, validation rules, and hospital-specific settings
 */

const { PrismaClient } = require('@prisma/client');
const fs = require('fs').promises;
const path = require('path');
const yaml = require('js-yaml');

class SystemConfigManager {
  constructor() {
    this.prisma = new PrismaClient();
    this.config = null;
    this.defaultConfig = this.getDefaultConfig();
    this.configCache = new Map();
    this.lastConfigLoad = null;
    this.logger = this.createLogger();
  }

  createLogger() {
    return {
      info: (msg, meta = {}) => console.log(`[ConfigManager] INFO: ${msg}`, meta),
      warn: (msg, meta = {}) => console.warn(`[ConfigManager] WARN: ${msg}`, meta),
      error: (msg, meta = {}) => console.error(`[ConfigManager] ERROR: ${msg}`, meta),
      debug: (msg, meta = {}) => console.log(`[ConfigManager] DEBUG: ${msg}`, meta)
    };
  }

  /**
   * Initialize configuration system
   */
  async initialize() {
    try {
      this.logger.info('Initializing system configuration...');

      // Load configuration from multiple sources
      await this.loadConfiguration();
      
      // Validate configuration
      await this.validateConfiguration();
      
      // Apply hospital-specific overrides
      await this.applyHospitalOverrides();
      
      this.logger.info('System configuration initialized successfully');
      
      return this.config;
      
    } catch (error) {
      this.logger.error('Failed to initialize configuration', { error: error.message });
      
      // Fallback to default configuration
      this.config = this.defaultConfig;
      this.logger.warn('Using default configuration as fallback');
      
      return this.config;
    }
  }

  /**
   * Load configuration from multiple sources with priority order
   */
  async loadConfiguration() {
    const sources = [
      { name: 'default', loader: () => this.defaultConfig },
      { name: 'file', loader: () => this.loadFileConfiguration() },
      { name: 'database', loader: () => this.loadDatabaseConfiguration() },
      { name: 'environment', loader: () => this.loadEnvironmentConfiguration() }
    ];

    let mergedConfig = {};

    for (const source of sources) {
      try {
        const sourceConfig = await source.loader();
        if (sourceConfig && Object.keys(sourceConfig).length > 0) {
          mergedConfig = this.deepMerge(mergedConfig, sourceConfig);
          this.logger.debug(`Loaded configuration from ${source.name}`);
        }
      } catch (error) {
        this.logger.warn(`Failed to load configuration from ${source.name}`, { error: error.message });
      }
    }

    this.config = mergedConfig;
    this.lastConfigLoad = new Date();
  }

  /**
   * Get default system configuration
   */
  getDefaultConfig() {
    return {
      // Data source configuration
      medSources: {
        priority: ['dmd', 'bnf', 'snomed', 'rxnorm', 'atc', 'dailymed', 'local'],
        enabled: ['dmd', 'bnf', 'rxnorm', 'local'],
        region: 'UK',
        allowClassSearch: true,
        datasetVersion: 'uk-2025-08',
        weights: {
          dmd: 10,
          bnf: 9,
          snomed: 8,
          rxnorm: 7,
          atc: 6,
          dailymed: 5,
          local: 8
        }
      },

      // Search configuration
      search: {
        minConfidenceDirect: 0.92,
        minConfidenceSuggest: 0.75,
        fuzzy: {
          trigramThreshold: 0.35,
          maxResults: 10,
          enablePhonetic: true,
          enableEditDistance: true
        },
        caching: {
          enabled: true,
          ttl: 600, // 10 minutes
          maxSize: 10000
        },
        rateLimit: {
          search: 20, // per minute
          validate: 10 // per minute
        }
      },

      // UI configuration
      ui: {
        disableAsNeeded: true,
        preloadAllFieldsInPopup: true,
        showProvenance: true,
        enterpriseMode: true,
        maxSearchResults: 10,
        enableAutoComplete: true,
        showConfidenceScores: false
      },

      // Validation configuration
      validation: {
        requireServerConfirmation: true,
        allowCustomDose: true,
        allowCustomFrequency: true,
        requireExplicitAckForCustom: true,
        strictMode: true,
        enableSafetyRules: true,
        maxDoseMultiplier: 2.0,
        duplicatePreventionEnabled: true
      },

      // Security configuration
      security: {
        redactPiiInLogs: true,
        auditAllActions: true,
        encryptSensitiveData: true,
        requireMfa: false,
        sessionTimeout: 3600, // 1 hour
        maxLoginAttempts: 5,
        passwordPolicy: {
          minLength: 12,
          requireUppercase: true,
          requireLowercase: true,
          requireNumbers: true,
          requireSpecialChars: true
        }
      },

      // Clinical configuration
      clinical: {
        enableInteractionChecking: true,
        enableAllergyChecking: true,
        enableDuplicateTherapyChecking: true,
        enableDoseRangeChecking: true,
        enableContraindicationChecking: true,
        defaultUnits: {
          weight: 'kg',
          volume: 'ml',
          mass: 'mg'
        }
      },

      // Integration configuration
      integrations: {
        fhir: {
          enabled: false,
          version: 'R4',
          endpoint: null
        },
        hl7: {
          enabled: false,
          version: '2.5'
        },
        emr: {
          enabled: false,
          type: null,
          endpoint: null
        }
      },

      // Monitoring configuration
      monitoring: {
        metrics: {
          enabled: true,
          interval: 60, // seconds
          retention: 30 // days
        },
        logging: {
          level: 'info',
          retention: 90, // days
          structured: true
        },
        alerts: {
          enabled: true,
          channels: ['email', 'webhook'],
          thresholds: {
            errorRate: 0.05,
            responseTime: 5000,
            availability: 0.99
          }
        }
      }
    };
  }

  /**
   * Load configuration from file (YAML/JSON)
   */
  async loadFileConfiguration() {
    const configPaths = [
      path.join(process.cwd(), 'config', 'medtrack.yml'),
      path.join(process.cwd(), 'config', 'medtrack.yaml'),
      path.join(process.cwd(), 'config', 'medtrack.json'),
      path.join(process.cwd(), 'medtrack.config.yml')
    ];

    for (const configPath of configPaths) {
      try {
        const content = await fs.readFile(configPath, 'utf8');
        const config = configPath.endsWith('.json') ? 
          JSON.parse(content) : 
          yaml.load(content);
        
        this.logger.debug(`Loaded file configuration from ${configPath}`);
        return config;
        
      } catch (error) {
        if (error.code !== 'ENOENT') {
          this.logger.warn(`Failed to load config file ${configPath}`, { error: error.message });
        }
      }
    }

    return {};
  }

  /**
   * Load configuration from database
   */
  async loadDatabaseConfiguration() {
    try {
      const configEntries = await this.prisma.systemConfig.findMany({
        where: { isActive: true }
      });

      const dbConfig = {};
      
      for (const entry of configEntries) {
        this.setNestedValue(dbConfig, entry.key, entry.value);
      }

      return dbConfig;
      
    } catch (error) {
      this.logger.warn('Failed to load database configuration', { error: error.message });
      return {};
    }
  }

  /**
   * Load configuration from environment variables
   */
  loadEnvironmentConfiguration() {
    const envConfig = {};

    // Map environment variables to configuration
    const envMappings = {
      'MEDTRACK_REGION': 'medSources.region',
      'MEDTRACK_SOURCES': 'medSources.enabled',
      'MEDTRACK_SEARCH_CONFIDENCE': 'search.minConfidenceDirect',
      'MEDTRACK_STRICT_MODE': 'validation.strictMode',
      'MEDTRACK_AUDIT_ENABLED': 'security.auditAllActions',
      'MEDTRACK_CACHE_TTL': 'search.caching.ttl',
      'MEDTRACK_MAX_RESULTS': 'search.fuzzy.maxResults'
    };

    for (const [envVar, configPath] of Object.entries(envMappings)) {
      const value = process.env[envVar];
      if (value !== undefined) {
        this.setNestedValue(envConfig, configPath, this.parseEnvValue(value));
      }
    }

    return envConfig;
  }

  /**
   * Apply hospital-specific configuration overrides
   */
  async applyHospitalOverrides() {
    try {
      // Get hospital-specific settings if user context is available
      const hospitalOverrides = await this.getHospitalOverrides();
      
      if (hospitalOverrides && Object.keys(hospitalOverrides).length > 0) {
        this.config = this.deepMerge(this.config, hospitalOverrides);
        this.logger.debug('Applied hospital-specific overrides');
      }
      
    } catch (error) {
      this.logger.warn('Failed to apply hospital overrides', { error: error.message });
    }
  }

  /**
   * Get hospital-specific configuration overrides
   */
  async getHospitalOverrides(hospitalId = null) {
    if (!hospitalId) return {};

    try {
      const hospital = await this.prisma.hospital.findUnique({
        where: { id: hospitalId }
      });

      if (hospital && hospital.sourceConfig) {
        return hospital.sourceConfig;
      }

      return {};
      
    } catch (error) {
      this.logger.warn(`Failed to get hospital overrides for ${hospitalId}`, { error: error.message });
      return {};
    }
  }

  /**
   * Validate configuration structure and values
   */
  async validateConfiguration() {
    const errors = [];

    // Validate required fields
    if (!this.config.medSources || !this.config.medSources.priority) {
      errors.push('medSources.priority is required');
    }

    if (!this.config.search || typeof this.config.search.minConfidenceDirect !== 'number') {
      errors.push('search.minConfidenceDirect must be a number');
    }

    // Validate confidence thresholds
    if (this.config.search.minConfidenceDirect < 0 || this.config.search.minConfidenceDirect > 1) {
      errors.push('search.minConfidenceDirect must be between 0 and 1');
    }

    if (this.config.search.minConfidenceSuggest < 0 || this.config.search.minConfidenceSuggest > 1) {
      errors.push('search.minConfidenceSuggest must be between 0 and 1');
    }

    // Validate source priorities
    if (this.config.medSources.priority && !Array.isArray(this.config.medSources.priority)) {
      errors.push('medSources.priority must be an array');
    }

    // Validate enabled sources
    if (this.config.medSources.enabled) {
      for (const source of this.config.medSources.enabled) {
        if (!this.config.medSources.priority.includes(source)) {
          errors.push(`Enabled source '${source}' not found in priority list`);
        }
      }
    }

    if (errors.length > 0) {
      throw new Error(`Configuration validation failed: ${errors.join(', ')}`);
    }

    this.logger.debug('Configuration validation passed');
  }

  /**
   * Get configuration value by path
   */
  get(path, defaultValue = null) {
    return this.getNestedValue(this.config, path, defaultValue);
  }

  /**
   * Set configuration value by path
   */
  async set(path, value, persist = false) {
    this.setNestedValue(this.config, path, value);

    if (persist) {
      await this.persistConfigValue(path, value);
    }

    // Invalidate cache
    this.configCache.clear();
  }

  /**
   * Get source priority configuration
   */
  getSourcePriority() {
    const priority = this.get('medSources.priority', []);
    const weights = this.get('medSources.weights', {});
    const enabled = this.get('medSources.enabled', []);

    const priorityMap = {};
    
    for (let i = 0; i < priority.length; i++) {
      const source = priority[i];
      if (enabled.includes(source)) {
        priorityMap[source] = weights[source] || (10 - i);
      }
    }

    return priorityMap;
  }

  /**
   * Get enabled sources in priority order
   */
  getEnabledSources() {
    const priority = this.get('medSources.priority', []);
    const enabled = this.get('medSources.enabled', []);

    return priority.filter(source => enabled.includes(source));
  }

  /**
   * Get validation configuration
   */
  getValidationConfig() {
    return {
      requireServerConfirmation: this.get('validation.requireServerConfirmation', true),
      allowCustomDose: this.get('validation.allowCustomDose', true),
      allowCustomFrequency: this.get('validation.allowCustomFrequency', true),
      requireExplicitAckForCustom: this.get('validation.requireExplicitAckForCustom', true),
      strictMode: this.get('validation.strictMode', true),
      enableSafetyRules: this.get('validation.enableSafetyRules', true),
      maxDoseMultiplier: this.get('validation.maxDoseMultiplier', 2.0),
      duplicatePreventionEnabled: this.get('validation.duplicatePreventionEnabled', true)
    };
  }

  /**
   * Get search configuration
   */
  getSearchConfig() {
    return {
      minConfidenceDirect: this.get('search.minConfidenceDirect', 0.92),
      minConfidenceSuggest: this.get('search.minConfidenceSuggest', 0.75),
      trigramThreshold: this.get('search.fuzzy.trigramThreshold', 0.35),
      maxResults: this.get('search.fuzzy.maxResults', 10),
      enablePhonetic: this.get('search.fuzzy.enablePhonetic', true),
      enableEditDistance: this.get('search.fuzzy.enableEditDistance', true),
      cacheEnabled: this.get('search.caching.enabled', true),
      cacheTtl: this.get('search.caching.ttl', 600)
    };
  }

  /**
   * Get security configuration
   */
  getSecurityConfig() {
    return {
      redactPiiInLogs: this.get('security.redactPiiInLogs', true),
      auditAllActions: this.get('security.auditAllActions', true),
      encryptSensitiveData: this.get('security.encryptSensitiveData', true),
      requireMfa: this.get('security.requireMfa', false),
      sessionTimeout: this.get('security.sessionTimeout', 3600),
      maxLoginAttempts: this.get('security.maxLoginAttempts', 5)
    };
  }

  /**
   * Update source priority
   */
  async updateSourcePriority(newPriority, persist = true) {
    await this.set('medSources.priority', newPriority, persist);
    this.logger.info('Source priority updated', { newPriority });
  }

  /**
   * Enable/disable source
   */
  async toggleSource(sourceName, enabled, persist = true) {
    const currentEnabled = this.get('medSources.enabled', []);
    
    let newEnabled;
    if (enabled && !currentEnabled.includes(sourceName)) {
      newEnabled = [...currentEnabled, sourceName];
    } else if (!enabled && currentEnabled.includes(sourceName)) {
      newEnabled = currentEnabled.filter(s => s !== sourceName);
    } else {
      return; // No change needed
    }

    await this.set('medSources.enabled', newEnabled, persist);
    this.logger.info(`Source ${sourceName} ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Reload configuration from all sources
   */
  async reload() {
    this.logger.info('Reloading system configuration...');
    this.configCache.clear();
    await this.loadConfiguration();
    await this.validateConfiguration();
    await this.applyHospitalOverrides();
    this.logger.info('Configuration reloaded successfully');
  }

  /**
   * Export current configuration
   */
  exportConfig(format = 'json') {
    if (format === 'yaml') {
      return yaml.dump(this.config, { indent: 2 });
    }
    return JSON.stringify(this.config, null, 2);
  }

  /**
   * Helper methods
   */
  getNestedValue(obj, path, defaultValue = null) {
    const keys = path.split('.');
    let current = obj;

    for (const key of keys) {
      if (current && typeof current === 'object' && key in current) {
        current = current[key];
      } else {
        return defaultValue;
      }
    }

    return current;
  }

  setNestedValue(obj, path, value) {
    const keys = path.split('.');
    let current = obj;

    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!(key in current) || typeof current[key] !== 'object') {
        current[key] = {};
      }
      current = current[key];
    }

    current[keys[keys.length - 1]] = value;
  }

  parseEnvValue(value) {
    // Try to parse as JSON first
    try {
      return JSON.parse(value);
    } catch {
      // Handle special string values
      if (value.toLowerCase() === 'true') return true;
      if (value.toLowerCase() === 'false') return false;
      if (/^\d+$/.test(value)) return parseInt(value);
      if (/^\d+\.\d+$/.test(value)) return parseFloat(value);
      if (value.includes(',')) return value.split(',').map(s => s.trim());
      
      return value;
    }
  }

  deepMerge(target, source) {
    const result = { ...target };

    for (const key in source) {
      if (source.hasOwnProperty(key)) {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
          result[key] = this.deepMerge(result[key] || {}, source[key]);
        } else {
          result[key] = source[key];
        }
      }
    }

    return result;
  }

  async persistConfigValue(path, value) {
    try {
      await this.prisma.systemConfig.upsert({
        where: { key: path },
        update: { 
          value: value,
          updatedAt: new Date()
        },
        create: {
          key: path,
          value: value,
          category: path.split('.')[0]
        }
      });
    } catch (error) {
      this.logger.error(`Failed to persist config value ${path}`, { error: error.message });
    }
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    await this.prisma.$disconnect();
    this.configCache.clear();
  }
}

module.exports = SystemConfigManager;

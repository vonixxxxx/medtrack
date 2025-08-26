/**
 * Base ETL Adapter for Hospital-Grade Medication Knowledge Engine
 * Provides common interface for all data source adapters
 */

const { PrismaClient } = require('@prisma/client');
const crypto = require('crypto');

class BaseAdapter {
  constructor(sourceName, config = {}) {
    this.sourceName = sourceName;
    this.config = config;
    this.prisma = new PrismaClient();
    this.logger = this.createLogger();
    this.stats = {
      processed: 0,
      skipped: 0,
      errors: 0,
      startTime: null,
      endTime: null
    };
  }

  createLogger() {
    return {
      info: (msg, meta = {}) => console.log(`[${this.sourceName}] INFO: ${msg}`, meta),
      warn: (msg, meta = {}) => console.warn(`[${this.sourceName}] WARN: ${msg}`, meta),
      error: (msg, meta = {}) => console.error(`[${this.sourceName}] ERROR: ${msg}`, meta),
      debug: (msg, meta = {}) => this.config.debug && console.log(`[${this.sourceName}] DEBUG: ${msg}`, meta)
    };
  }

  /**
   * Abstract methods to be implemented by each adapter
   */
  async initialize() {
    throw new Error('initialize() must be implemented by adapter');
  }

  async fetchData() {
    throw new Error('fetchData() must be implemented by adapter');
  }

  async transformData(rawData) {
    throw new Error('transformData() must be implemented by adapter');
  }

  async validateData(transformedData) {
    throw new Error('validateData() must be implemented by adapter');
  }

  /**
   * Main ETL execution pipeline
   */
  async execute(jobType = 'full_sync') {
    const jobId = await this.createETLJob(jobType);
    this.stats.startTime = new Date();

    try {
      this.logger.info(`Starting ETL job ${jobId} for ${this.sourceName}`);
      
      await this.updateJobStatus(jobId, 'running');
      
      // Initialize adapter
      await this.initialize();
      
      // Fetch raw data
      this.logger.info('Fetching data from source...');
      const rawData = await this.fetchData();
      
      // Transform to canonical format
      this.logger.info('Transforming data to canonical format...');
      const transformedData = await this.transformData(rawData);
      
      // Validate transformed data
      this.logger.info('Validating transformed data...');
      const validatedData = await this.validateData(transformedData);
      
      // Load into database
      this.logger.info('Loading data into database...');
      await this.loadData(validatedData);
      
      // Update search indexes
      this.logger.info('Updating search indexes...');
      await this.updateSearchIndexes(validatedData);
      
      this.stats.endTime = new Date();
      
      await this.updateJobStatus(jobId, 'completed', this.stats);
      
      this.logger.info(`ETL job ${jobId} completed successfully`, this.stats);
      
      return {
        success: true,
        jobId,
        stats: this.stats
      };
      
    } catch (error) {
      this.stats.endTime = new Date();
      this.stats.errors++;
      
      this.logger.error(`ETL job ${jobId} failed`, { error: error.message, stack: error.stack });
      
      await this.updateJobStatus(jobId, 'failed', this.stats, { error: error.message });
      
      throw error;
    }
  }

  /**
   * Create ETL job record
   */
  async createETLJob(jobType) {
    const job = await this.prisma.eTLJob.create({
      data: {
        sourceName: this.sourceName,
        jobType,
        status: 'pending',
        metadata: {
          config: this.sanitizeConfig(this.config),
          version: this.getDatasetVersion()
        }
      }
    });
    
    return job.id;
  }

  /**
   * Update job status and statistics
   */
  async updateJobStatus(jobId, status, stats = null, errors = null) {
    const updateData = {
      status,
      ...(status === 'running' && { startedAt: new Date() }),
      ...(status === 'completed' || status === 'failed') && { 
        completedAt: new Date(),
        recordsProcessed: stats?.processed || 0,
        recordsSkipped: stats?.skipped || 0
      },
      ...(errors && { errors })
    };

    await this.prisma.eTLJob.update({
      where: { id: jobId },
      data: updateData
    });
  }

  /**
   * Load validated data into canonical database
   */
  async loadData(validatedData) {
    const { medications, products, strengths, rules } = validatedData;
    
    // Use transaction for consistency
    await this.prisma.$transaction(async (tx) => {
      
      // Load medications
      for (const med of medications) {
        await this.upsertMedication(tx, med);
      }
      
      // Load products
      for (const product of products) {
        await this.upsertProduct(tx, product);
      }
      
      // Load strengths
      for (const strength of strengths) {
        await this.upsertStrength(tx, strength);
      }
      
      // Load rules
      for (const rule of rules) {
        await this.upsertRule(tx, rule);
      }
      
    });
  }

  /**
   * Upsert medication with conflict resolution
   */
  async upsertMedication(tx, medData) {
    try {
      const medication = await tx.medication.upsert({
        where: { genericName: medData.genericName },
        update: {
          atcCode: medData.atcCode,
          classHuman: medData.classHuman,
          synonyms: medData.synonyms,
          sourceRefs: medData.sourceRefs,
          datasetVersion: this.getDatasetVersion(),
          searchTokens: this.generateSearchTokens(medData),
          metaphoneKey: this.generateMetaphoneKey(medData.genericName),
          trigramTokens: this.generateTrigramTokens(medData.genericName),
          updatedAt: new Date()
        },
        create: {
          id: medData.id || crypto.randomUUID(),
          genericName: medData.genericName,
          atcCode: medData.atcCode,
          classHuman: medData.classHuman,
          synonyms: medData.synonyms,
          sourceRefs: medData.sourceRefs,
          datasetVersion: this.getDatasetVersion(),
          searchTokens: this.generateSearchTokens(medData),
          metaphoneKey: this.generateMetaphoneKey(medData.genericName),
          trigramTokens: this.generateTrigramTokens(medData.genericName)
        }
      });
      
      this.stats.processed++;
      return medication;
      
    } catch (error) {
      this.logger.error(`Failed to upsert medication ${medData.genericName}`, { error: error.message });
      this.stats.errors++;
      throw error;
    }
  }

  /**
   * Upsert product with conflict resolution
   */
  async upsertProduct(tx, productData) {
    try {
      const product = await tx.product.upsert({
        where: { 
          medicationId_brandName_doseForm: {
            medicationId: productData.medicationId,
            brandName: productData.brandName || '',
            doseForm: productData.doseForm
          }
        },
        update: {
          route: productData.route,
          intakeType: productData.intakeType,
          defaultPlaces: productData.defaultPlaces,
          allowedFrequencies: productData.allowedFrequencies,
          provenance: productData.provenance,
          datasetVersion: this.getDatasetVersion(),
          notes: productData.notes,
          updatedAt: new Date()
        },
        create: {
          id: productData.id || crypto.randomUUID(),
          medicationId: productData.medicationId,
          brandName: productData.brandName,
          route: productData.route,
          doseForm: productData.doseForm,
          intakeType: productData.intakeType,
          defaultPlaces: productData.defaultPlaces,
          allowedFrequencies: productData.allowedFrequencies,
          provenance: productData.provenance,
          datasetVersion: this.getDatasetVersion(),
          notes: productData.notes
        }
      });
      
      this.stats.processed++;
      return product;
      
    } catch (error) {
      this.logger.error(`Failed to upsert product ${productData.brandName}`, { error: error.message });
      this.stats.errors++;
      throw error;
    }
  }

  /**
   * Generate search optimization tokens
   */
  generateSearchTokens(medData) {
    const tokens = new Set();
    
    // Add generic name variations
    tokens.add(medData.genericName.toLowerCase());
    
    // Add synonyms
    if (medData.synonyms) {
      medData.synonyms.forEach(synonym => tokens.add(synonym.toLowerCase()));
    }
    
    // Add partial tokens for fuzzy search
    const name = medData.genericName.toLowerCase();
    for (let i = 3; i <= name.length; i++) {
      tokens.add(name.substring(0, i));
    }
    
    return Array.from(tokens);
  }

  /**
   * Generate metaphone key for phonetic matching
   */
  generateMetaphoneKey(text) {
    // Simplified metaphone implementation
    // In production, use a proper metaphone library
    return text.toLowerCase()
      .replace(/[aeiou]/g, '')
      .replace(/[^a-z]/g, '')
      .substring(0, 6);
  }

  /**
   * Generate trigram tokens for fuzzy search
   */
  generateTrigramTokens(text) {
    const tokens = new Set();
    const normalized = text.toLowerCase().replace(/[^a-z0-9]/g, '');
    
    for (let i = 0; i <= normalized.length - 3; i++) {
      tokens.add(normalized.substring(i, i + 3));
    }
    
    return Array.from(tokens);
  }

  /**
   * Update search indexes for fast retrieval
   */
  async updateSearchIndexes(validatedData) {
    const { medications } = validatedData;
    
    for (const med of medications) {
      // Update exact match index
      await this.updateSearchIndex(med.genericName, med.id, 'exact');
      
      // Update synonym indexes
      if (med.synonyms) {
        for (const synonym of med.synonyms) {
          await this.updateSearchIndex(synonym, med.id, 'synonym');
        }
      }
      
      // Update metaphone index
      const metaphoneKey = this.generateMetaphoneKey(med.genericName);
      await this.updateSearchIndex(metaphoneKey, med.id, 'metaphone');
      
      // Update trigram indexes
      const trigrams = this.generateTrigramTokens(med.genericName);
      for (const trigram of trigrams) {
        await this.updateSearchIndex(trigram, med.id, 'trigram');
      }
    }
  }

  /**
   * Update individual search index entry
   */
  async updateSearchIndex(term, medicationId, tokenType) {
    try {
      await this.prisma.searchIndex.upsert({
        where: { term },
        update: {
          medicationIds: {
            push: medicationId
          },
          frequency: {
            increment: 1
          },
          updatedAt: new Date()
        },
        create: {
          term,
          medicationIds: [medicationId],
          tokenType,
          frequency: 1
        }
      });
    } catch (error) {
      // Handle array uniqueness constraint violations
      this.logger.debug(`Search index update skipped for term: ${term}`);
    }
  }

  /**
   * Get dataset version identifier
   */
  getDatasetVersion() {
    const date = new Date();
    return `${this.sourceName}-${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
  }

  /**
   * Sanitize config for logging (remove sensitive data)
   */
  sanitizeConfig(config) {
    const sanitized = { ...config };
    
    // Remove sensitive fields
    delete sanitized.apiKey;
    delete sanitized.password;
    delete sanitized.token;
    delete sanitized.secret;
    
    return sanitized;
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    await this.prisma.$disconnect();
  }
}

module.exports = BaseAdapter;

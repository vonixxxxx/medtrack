/**
 * Profound Medication Resolver
 * Hospital-grade medication matching with explainability
 * Handles: exact matches, synonyms, acronyms, classes, fuzzy matching
 */

const { PrismaClient } = require('@prisma/client');
const NodeCache = require('node-cache');

class ProfoundMedicationResolver {
  constructor(config = {}) {
    this.config = {
      enabledSources: ['dmd', 'bnf', 'rxnorm', 'local'],
      sourcePriority: { 'dmd': 10, 'bnf': 9, 'rxnorm': 7, 'local': 8 },
      region: 'UK',
      confidenceThresholds: {
        directSelect: 0.92,
        suggestions: 0.75,
        fuzzy: 0.35
      },
      maxResults: 10,
      enableClassSearch: true,
      enableFuzzySearch: true,
      cacheTimeout: 600, // 10 minutes
      ...config
    };
    
    this.prisma = new PrismaClient();
    this.cache = new NodeCache({ stdTTL: this.config.cacheTimeout });
    this.logger = this.createLogger();
    
    // Pre-compiled regex patterns for performance
    this.patterns = {
      dosage: /(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)/gi,
      frequency: /(daily|weekly|monthly|hourly)/gi,
      route: /(oral|injection|topical|inhalation)/gi
    };
    
    // Metaphone cache for phonetic matching
    this.metaphoneCache = new Map();
  }

  createLogger() {
    return {
      info: (msg, meta = {}) => console.log(`[ProfoundResolver] INFO: ${msg}`, meta),
      warn: (msg, meta = {}) => console.warn(`[ProfoundResolver] WARN: ${msg}`, meta),
      error: (msg, meta = {}) => console.error(`[ProfoundResolver] ERROR: ${msg}`, meta),
      debug: (msg, meta = {}) => this.config.debug && console.log(`[ProfoundResolver] DEBUG: ${msg}`, meta)
    };
  }

  /**
   * Main resolution method - hospital-grade medication matching
   */
  async resolve(query, options = {}) {
    const startTime = Date.now();
    
    try {
      // Input validation
      if (!query || typeof query !== 'string' || query.trim().length < 2) {
        return this.buildEmptyResult(query, 'Query too short or invalid');
      }

      const normalizedQuery = this.normalizeQuery(query);
      const cacheKey = this.buildCacheKey(normalizedQuery, options);
      
      // Check cache first
      const cached = this.cache.get(cacheKey);
      if (cached && !options.skipCache) {
        this.logger.debug(`Cache hit for query: ${normalizedQuery}`);
        return cached;
      }

      this.logger.info(`Resolving medication query: "${normalizedQuery}"`);

      // Multi-stage resolution pipeline
      const results = await this.executeResolutionPipeline(normalizedQuery, options);
      
      // Apply confidence thresholds and ranking
      const rankedResults = this.rankAndFilterResults(results, normalizedQuery);
      
      // Build response with explainability
      const response = this.buildResponse(normalizedQuery, rankedResults, startTime);
      
      // Cache the result
      this.cache.set(cacheKey, response);
      
      this.logger.info(`Resolution completed in ${Date.now() - startTime}ms`, {
        query: normalizedQuery,
        resultsCount: response.matches.length,
        suggestionsCount: response.suggestions.length
      });

      return response;

    } catch (error) {
      this.logger.error('Resolution failed', { query, error: error.message, stack: error.stack });
      return this.buildErrorResult(query, error.message);
    }
  }

  /**
   * Execute the multi-stage resolution pipeline
   */
  async executeResolutionPipeline(query, options = {}) {
    const results = [];
    
    // Stage A: Deterministic matches (high confidence)
    const exactMatches = await this.findExactMatches(query);
    results.push(...exactMatches);
    
    const synonymMatches = await this.findSynonymMatches(query);
    results.push(...synonymMatches);
    
    const acronymMatches = await this.findAcronymMatches(query);
    results.push(...acronymMatches);
    
    // Stage B: Class-based search (if enabled)
    if (this.config.enableClassSearch) {
      const classMatches = await this.findClassMatches(query);
      results.push(...classMatches);
    }
    
    // Stage C: Fuzzy matching (if not enough high-confidence results)
    if (results.length < 3 && this.config.enableFuzzySearch) {
      const fuzzyMatches = await this.findFuzzyMatches(query);
      results.push(...fuzzyMatches);
    }
    
    // Stage D: Phonetic matching (last resort)
    if (results.length < 2) {
      const phoneticMatches = await this.findPhoneticMatches(query);
      results.push(...phoneticMatches);
    }
    
    return results;
  }

  /**
   * Find exact generic/brand name matches
   */
  async findExactMatches(query) {
    try {
      const medications = await this.prisma.medication.findMany({
        where: {
          OR: [
            { genericName: { equals: query, mode: 'insensitive' } },
            { 
              products: {
                some: {
                  brandName: { equals: query, mode: 'insensitive' }
                }
              }
            }
          ],
          isActive: true
        },
        include: {
          products: {
            where: { isActive: true },
            include: {
              strengths: { where: { isActive: true } },
              rules: true
            }
          }
        }
      });

      return medications.map(med => ({
        medication: med,
        confidence: 1.0,
        reasons: ['exact-match'],
        matchType: 'exact',
        matchedField: med.genericName.toLowerCase() === query ? 'genericName' : 'brandName'
      }));

    } catch (error) {
      this.logger.error('Exact match search failed', { query, error: error.message });
      return [];
    }
  }

  /**
   * Find synonym matches
   */
  async findSynonymMatches(query) {
    try {
      const medications = await this.prisma.medication.findMany({
        where: {
          synonyms: {
            has: query
          },
          isActive: true
        },
        include: {
          products: {
            where: { isActive: true },
            include: {
              strengths: { where: { isActive: true } },
              rules: true
            }
          }
        }
      });

      return medications.map(med => ({
        medication: med,
        confidence: 0.95,
        reasons: ['synonym-match'],
        matchType: 'synonym',
        matchedField: 'synonyms'
      }));

    } catch (error) {
      this.logger.error('Synonym search failed', { query, error: error.message });
      return [];
    }
  }

  /**
   * Find acronym matches (e.g., GLP-1, NSAID)
   */
  async findAcronymMatches(query) {
    try {
      // Normalize query for acronym matching
      const normalizedAcronym = query.toLowerCase().replace(/[-\s]/g, '');
      
      const medications = await this.prisma.medication.findMany({
        where: {
          OR: [
            { synonyms: { has: normalizedAcronym } },
            { synonyms: { has: query } },
            { classHuman: { contains: query, mode: 'insensitive' } }
          ],
          isActive: true
        },
        include: {
          products: {
            where: { isActive: true },
            include: {
              strengths: { where: { isActive: true } },
              rules: true
            }
          }
        }
      });

      return medications.map(med => ({
        medication: med,
        confidence: 0.90,
        reasons: ['acronym-match'],
        matchType: 'acronym',
        matchedField: 'classHuman'
      }));

    } catch (error) {
      this.logger.error('Acronym search failed', { query, error: error.message });
      return [];
    }
  }

  /**
   * Find class-based matches using ATC codes
   */
  async findClassMatches(query) {
    try {
      // Map common class queries to ATC patterns
      const classMap = {
        'glp1': 'A10BJ',
        'glp-1': 'A10BJ',
        'incretin': 'A10BJ',
        'nsaid': 'M01AE',
        'anti-inflammatory': 'M01',
        'statin': 'C10AA',
        'cholesterol': 'C10AA',
        'ppi': 'A02BC',
        'proton pump': 'A02BC',
        'beta2': 'R03AC',
        'bronchodilator': 'R03',
        'inhaler': 'R03',
        'painkiller': 'N02',
        'analgesic': 'N02',
        'antibiotic': 'J01',
        'penicillin': 'J01CA',
        'ace inhibitor': 'C09AA',
        'beta blocker': 'C07'
      };

      const atcPattern = classMap[query.toLowerCase()];
      if (!atcPattern) return [];

      const medications = await this.prisma.medication.findMany({
        where: {
          atcCode: {
            startsWith: atcPattern
          },
          isActive: true
        },
        include: {
          products: {
            where: { isActive: true },
            include: {
              strengths: { where: { isActive: true } },
              rules: true
            }
          }
        },
        take: this.config.maxResults
      });

      return medications.map(med => ({
        medication: med,
        confidence: 0.85,
        reasons: ['class-match', `atc:${atcPattern}`],
        matchType: 'class',
        matchedField: 'atcCode'
      }));

    } catch (error) {
      this.logger.error('Class search failed', { query, error: error.message });
      return [];
    }
  }

  /**
   * Find fuzzy matches using multiple algorithms
   */
  async findFuzzyMatches(query) {
    try {
      // Get search index entries for trigram matching
      const trigramMatches = await this.findTrigramMatches(query);
      const metaphoneMatches = await this.findMetaphoneMatches(query);
      const editDistanceMatches = await this.findEditDistanceMatches(query);
      
      // Combine and score fuzzy matches
      const allMatches = [...trigramMatches, ...metaphoneMatches, ...editDistanceMatches];
      const deduplicatedMatches = this.deduplicateMatches(allMatches);
      
      return deduplicatedMatches
        .filter(match => match.confidence >= this.config.confidenceThresholds.fuzzy)
        .slice(0, this.config.maxResults);

    } catch (error) {
      this.logger.error('Fuzzy search failed', { query, error: error.message });
      return [];
    }
  }

  /**
   * Find trigram-based fuzzy matches
   */
  async findTrigramMatches(query) {
    const trigrams = this.generateTrigrams(query);
    if (trigrams.length === 0) return [];

    try {
      const indexEntries = await this.prisma.searchIndex.findMany({
        where: {
          term: { in: trigrams },
          tokenType: 'trigram'
        }
      });

      const medicationScores = new Map();
      
      // Calculate trigram similarity scores
      for (const entry of indexEntries) {
        for (const medId of entry.medicationIds) {
          const currentScore = medicationScores.get(medId) || 0;
          medicationScores.set(medId, currentScore + (1 / trigrams.length));
        }
      }

      // Get medications with sufficient trigram overlap
      const medicationIds = Array.from(medicationScores.entries())
        .filter(([_, score]) => score >= 0.3)
        .sort(([_, a], [__, b]) => b - a)
        .slice(0, this.config.maxResults)
        .map(([id, _]) => id);

      if (medicationIds.length === 0) return [];

      const medications = await this.prisma.medication.findMany({
        where: {
          id: { in: medicationIds },
          isActive: true
        },
        include: {
          products: {
            where: { isActive: true },
            include: {
              strengths: { where: { isActive: true } },
              rules: true
            }
          }
        }
      });

      return medications.map(med => ({
        medication: med,
        confidence: Math.min(medicationScores.get(med.id) || 0, 0.8),
        reasons: ['trigram-match'],
        matchType: 'fuzzy',
        matchedField: 'trigrams'
      }));

    } catch (error) {
      this.logger.error('Trigram search failed', { query, error: error.message });
      return [];
    }
  }

  /**
   * Find phonetic matches using metaphone
   */
  async findPhoneticMatches(query) {
    try {
      const metaphoneKey = this.generateMetaphoneKey(query);
      
      const medications = await this.prisma.medication.findMany({
        where: {
          metaphoneKey: metaphoneKey,
          isActive: true
        },
        include: {
          products: {
            where: { isActive: true },
            include: {
              strengths: { where: { isActive: true } },
              rules: true
            }
          }
        },
        take: this.config.maxResults
      });

      return medications.map(med => ({
        medication: med,
        confidence: 0.6,
        reasons: ['phonetic-match'],
        matchType: 'phonetic',
        matchedField: 'metaphoneKey'
      }));

    } catch (error) {
      this.logger.error('Phonetic search failed', { query, error: error.message });
      return [];
    }
  }

  /**
   * Rank and filter results by confidence and source priority
   */
  rankAndFilterResults(results, query) {
    // Deduplicate by medication ID
    const deduped = this.deduplicateMatches(results);
    
    // Apply source priority weighting
    const weighted = deduped.map(result => ({
      ...result,
      finalScore: this.calculateFinalScore(result, query)
    }));
    
    // Sort by final score
    const sorted = weighted.sort((a, b) => b.finalScore - a.finalScore);
    
    // Apply confidence thresholds
    const directSelect = sorted.filter(r => r.confidence >= this.config.confidenceThresholds.directSelect);
    const suggestions = sorted.filter(r => 
      r.confidence >= this.config.confidenceThresholds.suggestions && 
      r.confidence < this.config.confidenceThresholds.directSelect
    );
    
    return {
      directSelect: directSelect.slice(0, this.config.maxResults),
      suggestions: suggestions.slice(0, this.config.maxResults),
      all: sorted.slice(0, this.config.maxResults)
    };
  }

  /**
   * Calculate final weighted score
   */
  calculateFinalScore(result, query) {
    let score = result.confidence;
    
    // Source priority weighting
    const sourceRefs = result.medication.sourceRefs || {};
    for (const [source, priority] of Object.entries(this.config.sourcePriority)) {
      if (sourceRefs[source]) {
        score *= (1 + priority / 100);
        break; // Use highest priority source
      }
    }
    
    // Boost exact matches
    if (result.matchType === 'exact') {
      score *= 1.2;
    }
    
    // Boost matches in enabled sources
    if (this.isFromEnabledSource(result.medication)) {
      score *= 1.1;
    }
    
    // String similarity boost
    const similarity = this.calculateStringSimilarity(query, result.medication.genericName);
    score *= (0.8 + 0.2 * similarity);
    
    return Math.min(score, 1.0);
  }

  /**
   * Check if medication is from enabled source
   */
  isFromEnabledSource(medication) {
    const sourceRefs = medication.sourceRefs || {};
    return this.config.enabledSources.some(source => sourceRefs[source]);
  }

  /**
   * Generate trigrams for fuzzy matching
   */
  generateTrigrams(text) {
    const normalized = text.toLowerCase().replace(/[^a-z0-9]/g, '');
    const trigrams = [];
    
    if (normalized.length < 3) return trigrams;
    
    for (let i = 0; i <= normalized.length - 3; i++) {
      trigrams.push(normalized.substring(i, i + 3));
    }
    
    return trigrams;
  }

  /**
   * Generate metaphone key for phonetic matching
   */
  generateMetaphoneKey(text) {
    // Check cache first
    if (this.metaphoneCache.has(text)) {
      return this.metaphoneCache.get(text);
    }
    
    // Simplified metaphone implementation
    // In production, use a proper metaphone library like 'natural'
    const key = text.toLowerCase()
      .replace(/[aeiou]/g, '')
      .replace(/[^a-z]/g, '')
      .replace(/(.)\1+/g, '$1') // Remove consecutive duplicates
      .substring(0, 6);
    
    this.metaphoneCache.set(text, key);
    return key;
  }

  /**
   * Calculate string similarity using Jaro-Winkler distance
   */
  calculateStringSimilarity(str1, str2) {
    // Simplified similarity calculation
    // In production, use a proper string similarity library
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;
    
    if (longer.length === 0) return 1.0;
    
    const editDistance = this.calculateEditDistance(longer.toLowerCase(), shorter.toLowerCase());
    return (longer.length - editDistance) / longer.length;
  }

  /**
   * Calculate edit distance (Levenshtein)
   */
  calculateEditDistance(str1, str2) {
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

  /**
   * Deduplicate matches by medication ID
   */
  deduplicateMatches(matches) {
    const seen = new Set();
    return matches.filter(match => {
      const key = match.medication.id;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  /**
   * Normalize query for consistent processing
   */
  normalizeQuery(query) {
    return query
      .trim()
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, ' ');
  }

  /**
   * Build cache key for result caching
   */
  buildCacheKey(query, options) {
    const optionsHash = JSON.stringify(options);
    return `resolve:${query}:${optionsHash}`;
  }

  /**
   * Build successful response
   */
  buildResponse(query, rankedResults, startTime) {
    const processingTime = Date.now() - startTime;
    
    return {
      query,
      matches: rankedResults.directSelect.map(this.formatMatch.bind(this)),
      suggestions: rankedResults.suggestions.map(this.formatMatch.bind(this)),
      metadata: {
        processingTime,
        totalCandidates: rankedResults.all.length,
        confidenceThresholds: this.config.confidenceThresholds,
        datasetVersion: this.getLatestDatasetVersion(),
        sources: this.config.enabledSources
      }
    };
  }

  /**
   * Format match for API response
   */
  formatMatch(result) {
    const { medication, confidence, reasons, matchType, finalScore } = result;
    
    return {
      medicationId: medication.id,
      genericName: medication.genericName,
      atcCode: medication.atcCode,
      classHuman: medication.classHuman,
      confidence: Math.round(confidence * 100) / 100,
      finalScore: Math.round(finalScore * 100) / 100,
      reasons,
      matchType,
      products: medication.products.map(product => ({
        productId: product.id,
        brandName: product.brandName,
        route: product.route,
        doseForm: product.doseForm,
        intakeType: product.intakeType,
        strengthsCount: product.strengths?.length || 0
      })),
      sourceRefs: medication.sourceRefs
    };
  }

  /**
   * Build empty result for invalid queries
   */
  buildEmptyResult(query, reason) {
    return {
      query,
      matches: [],
      suggestions: [],
      metadata: {
        processingTime: 0,
        totalCandidates: 0,
        reason,
        datasetVersion: this.getLatestDatasetVersion()
      }
    };
  }

  /**
   * Build error result
   */
  buildErrorResult(query, error) {
    return {
      query,
      matches: [],
      suggestions: [],
      error: 'Search temporarily unavailable',
      metadata: {
        processingTime: 0,
        totalCandidates: 0,
        datasetVersion: this.getLatestDatasetVersion()
      }
    };
  }

  /**
   * Get latest dataset version
   */
  getLatestDatasetVersion() {
    const date = new Date();
    return `uk-${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    await this.prisma.$disconnect();
    this.cache.flushAll();
  }
}

module.exports = ProfoundMedicationResolver;

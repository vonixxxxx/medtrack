/**
 * Unit Tests for Profound Medication Resolver
 * Hospital-Grade Test Suite
 */

const ProfoundMedicationResolver = require('../../src/services/ProfoundMedicationResolver');

describe('ProfoundMedicationResolver', () => {
  let resolver;

  beforeEach(() => {
    resolver = new ProfoundMedicationResolver();
  });

  afterEach(async () => {
    await resolver.cleanup();
  });

  describe('Search Resolution', () => {
    test('should resolve exact medication names', async () => {
      const query = 'paracetamol';
      const options = {
        enabledSources: ['nhs_dmd', 'local'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      expect(result).toHaveProperty('query', query);
      expect(result).toHaveProperty('matches');
      expect(result).toHaveProperty('suggestions');
      expect(result).toHaveProperty('metadata');
      expect(result.matches).toBeInstanceOf(Array);
      
      // Check for high confidence exact match
      const exactMatch = result.matches.find(m => m.confidence >= 0.9);
      expect(exactMatch).toBeDefined();
      expect(exactMatch.genericName.toLowerCase()).toContain('paracetamol');
    });

    test('should resolve brand names to generic', async () => {
      const query = 'Ozempic';
      const options = {
        enabledSources: ['nhs_dmd', 'local'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.85,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      expect(result.matches.length).toBeGreaterThan(0);
      const semaglutideMatch = result.matches.find(m => 
        m.genericName.toLowerCase().includes('semaglutide')
      );
      expect(semaglutideMatch).toBeDefined();
      expect(semaglutideMatch.products).toBeInstanceOf(Array);
    });

    test('should handle acronym searches', async () => {
      const query = 'GLP-1';
      const options = {
        enabledSources: ['nhs_dmd', 'local'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.8,
          suggestions: 0.6,
          fuzzy: 0.5
        }
      };

      const result = await resolver.resolve(query, options);
      
      expect(result.matches.length).toBeGreaterThan(0);
      // Should find semaglutide, liraglutide, dulaglutide, etc.
      const glpMedications = result.matches.filter(m => 
        m.classHuman && m.classHuman.toLowerCase().includes('glp')
      );
      expect(glpMedications.length).toBeGreaterThan(0);
    });

    test('should provide fuzzy matching for typos', async () => {
      const query = 'paracitamol'; // Common typo
      const options = {
        enabledSources: ['nhs_dmd', 'local'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      // Should suggest paracetamol
      expect(result.suggestions.length).toBeGreaterThan(0);
      const paracetamolSuggestion = result.suggestions.find(s => 
        s.genericName.toLowerCase().includes('paracetamol')
      );
      expect(paracetamolSuggestion).toBeDefined();
      expect(paracetamolSuggestion.confidence).toBeGreaterThan(0.6);
    });

    test('should return empty results for invalid queries', async () => {
      const query = 'xyz123nonexistent';
      const options = {
        enabledSources: ['nhs_dmd', 'local'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      expect(result.matches).toHaveLength(0);
      expect(result.suggestions).toHaveLength(0);
      expect(result.metadata).toHaveProperty('searchType', 'no_results');
    });
  });

  describe('Confidence Scoring', () => {
    test('should assign higher confidence to exact matches', async () => {
      const query = 'ibuprofen';
      const options = {
        enabledSources: ['nhs_dmd', 'local'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      const exactMatch = result.matches.find(m => 
        m.genericName.toLowerCase() === 'ibuprofen'
      );
      expect(exactMatch).toBeDefined();
      expect(exactMatch.confidence).toBeGreaterThan(0.9);
      expect(exactMatch.matchType).toBe('exact');
    });

    test('should assign lower confidence to fuzzy matches', async () => {
      const query = 'ibuprofin'; // Typo
      const options = {
        enabledSources: ['nhs_dmd', 'local'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      if (result.suggestions.length > 0) {
        const ibuprofenSuggestion = result.suggestions.find(s => 
          s.genericName.toLowerCase().includes('ibuprofen')
        );
        if (ibuprofenSuggestion) {
          expect(ibuprofenSuggestion.confidence).toBeLessThan(0.9);
          expect(ibuprofenSuggestion.confidence).toBeGreaterThan(0.6);
        }
      }
    });
  });

  describe('Source Prioritization', () => {
    test('should prioritize enabled sources', async () => {
      const query = 'aspirin';
      const options = {
        enabledSources: ['nhs_dmd'],  // Only NHS source
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      expect(result.metadata.sources).toContain('nhs_dmd');
      if (result.matches.length > 0) {
        result.matches.forEach(match => {
          expect(match.sourceRefs).toBeDefined();
        });
      }
    });

    test('should handle multiple source conflicts', async () => {
      const query = 'metformin';
      const options = {
        enabledSources: ['nhs_dmd', 'rxnorm', 'local'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      expect(result.metadata.sources.length).toBeGreaterThan(0);
      if (result.matches.length > 0) {
        const metforminMatch = result.matches.find(m => 
          m.genericName.toLowerCase().includes('metformin')
        );
        expect(metforminMatch).toBeDefined();
        expect(metforminMatch.sourceRefs).toBeInstanceOf(Array);
      }
    });
  });

  describe('Performance', () => {
    test('should complete searches within time limits', async () => {
      const start = Date.now();
      
      const query = 'semaglutide';
      const options = {
        enabledSources: ['nhs_dmd', 'local'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      const duration = Date.now() - start;
      expect(duration).toBeLessThan(5000); // 5 second max
      expect(result).toBeDefined();
    });

    test('should handle concurrent searches', async () => {
      const queries = ['paracetamol', 'ibuprofen', 'aspirin', 'metformin', 'insulin'];
      const options = {
        enabledSources: ['nhs_dmd', 'local'],
        maxResults: 5,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const promises = queries.map(query => resolver.resolve(query, options));
      const results = await Promise.all(promises);
      
      expect(results).toHaveLength(5);
      results.forEach(result => {
        expect(result).toHaveProperty('query');
        expect(result).toHaveProperty('matches');
        expect(result).toHaveProperty('suggestions');
      });
    });
  });

  describe('Error Handling', () => {
    test('should handle malformed input gracefully', async () => {
      const queries = ['', '   ', null, undefined, 123, {}];
      const options = {
        enabledSources: ['nhs_dmd'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      for (const query of queries) {
        const result = await resolver.resolve(query, options);
        expect(result).toHaveProperty('matches');
        expect(result).toHaveProperty('suggestions');
        expect(result.matches).toHaveLength(0);
      }
    });

    test('should handle invalid source configurations', async () => {
      const query = 'paracetamol';
      const options = {
        enabledSources: ['invalid_source'],
        maxResults: 10,
        confidenceThresholds: {
          directSelect: 0.9,
          suggestions: 0.7,
          fuzzy: 0.6
        }
      };

      const result = await resolver.resolve(query, options);
      
      // Should still work with fallback behavior
      expect(result).toHaveProperty('matches');
      expect(result).toHaveProperty('suggestions');
    });
  });
});

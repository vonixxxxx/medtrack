const { EnhancedMedicationService, MedicationSummarizer, EnhancedEMAAPI } = require('../src/services/medicationService');
const axios = require('axios');

// Mock axios
jest.mock('axios');

describe('Enhanced Medication Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('EnhancedEMAAPI', () => {
    test('should search medicines successfully', async () => {
      const mockSearchResponse = {
        data: {
          medicines: [
            { id: '1', name: 'Test Medicine', internationalNonProprietaryName: 'Test Generic' }
          ]
        }
      };

      const mockDetailsResponse = {
        data: {
          id: '1',
          name: 'Test Medicine',
          internationalNonProprietaryName: 'Test Generic',
          brandNames: ['Test Brand'],
          pharmaceuticalForm: ['Tablet'],
          strength: ['100mg'],
          atcCode: 'A01BC01',
          therapeuticIndications: ['Test indication'],
          warnings: ['Test warning'],
          sideEffects: ['Test side effect'],
          drugInteractions: ['Test interaction'],
          pregnancyCategory: 'Category B',
          breastfeedingCategory: 'Safe',
          pediatricUse: 'Consult physician',
          geriatricUse: 'Use with caution',
          emaProductNumber: 'EPAR-001',
          marketingAuthorisationNumber: 'MA-001',
          authorizationStatus: 'Authorized',
          authorizationDate: '2023-01-01',
          lastUpdated: '2023-12-01'
        }
      };

      axios.get
        .mockResolvedValueOnce(mockSearchResponse)
        .mockResolvedValueOnce(mockDetailsResponse)
        .mockResolvedValueOnce({ data: {} }) // safety info
        .mockResolvedValueOnce({ data: {} }); // product info

      const results = await EnhancedEMAAPI.searchMedicines('test', 1);

      expect(results).toHaveLength(1);
      expect(results[0].name).toBe('Test Medicine');
      expect(results[0].genericName).toBe('Test Generic');
      expect(results[0].source).toBe('ema');
      expect(axios.get).toHaveBeenCalledTimes(4);
    });

    test('should handle search errors gracefully', async () => {
      axios.get.mockRejectedValue(new Error('API Error'));

      const results = await EnhancedEMAAPI.searchMedicines('test', 1);

      expect(results).toEqual([]);
    });

    test('should get medicine details successfully', async () => {
      const mockResponse = {
        data: {
          id: '1',
          name: 'Test Medicine',
          internationalNonProprietaryName: 'Test Generic'
        }
      };

      axios.get
        .mockResolvedValueOnce(mockResponse)
        .mockResolvedValueOnce({ data: {} })
        .mockResolvedValueOnce({ data: {} });

      const details = await EnhancedEMAAPI.getMedicineDetails('1');

      expect(details).toBeTruthy();
      expect(details.name).toBe('Test Medicine');
      expect(details.source).toBe('ema');
    });

    test('should handle details fetch errors gracefully', async () => {
      axios.get.mockRejectedValue(new Error('API Error'));

      const details = await EnhancedEMAAPI.getMedicineDetails('1');

      expect(details).toBeNull();
    });
  });

  describe('MedicationSummarizer', () => {
    test('should build summarization prompt correctly', () => {
      const medicationData = {
        name: 'Test Medicine',
        genericName: 'Test Generic',
        therapeuticIndications: ['Test indication'],
        warnings: ['Test warning'],
        sideEffects: ['Test side effect'],
        dosageForms: ['Tablet'],
        interactions: ['Test interaction'],
        pregnancyCategory: 'Category B',
        breastfeedingCategory: 'Safe',
        pediatricUse: 'Consult physician',
        geriatricUse: 'Use with caution'
      };

      const prompt = MedicationSummarizer.buildSummarizationPrompt(medicationData);

      expect(prompt).toContain('Test Medicine');
      expect(prompt).toContain('Test Generic');
      expect(prompt).toContain('Test warning');
      expect(prompt).toContain('Test side effect');
      expect(prompt).toContain('Category B');
    });

    test('should provide fallback summarization when LLM is not available', () => {
      const medicationData = {
        name: 'Test Medicine',
        warnings: ['Test warning'],
        sideEffects: ['Side effect 1', 'Side effect 2'],
        dosageForms: ['Tablet', 'Capsule'],
        interactions: ['Interaction 1', 'Interaction 2'],
        pregnancyCategory: 'Category B',
        breastfeedingCategory: 'Safe',
        pediatricUse: 'Consult physician',
        geriatricUse: 'Use with caution'
      };

      const summary = MedicationSummarizer.fallbackSummarization(medicationData);

      expect(summary.overview).toContain('Test Medicine');
      expect(summary.warnings).toContain('Test warning');
      expect(summary.sideEffects).toContain('Side effect 1');
      expect(summary.dosage).toContain('Tablet');
      expect(summary.interactions).toContain('Interaction 1');
      expect(summary.specialPopulations.pregnancy).toBe('Category B');
    });
  });

  describe('EnhancedMedicationService', () => {
    test('should search medications with EMA source', async () => {
      // Mock the EnhancedEMAAPI
      const mockEmaResults = [
        { source: 'ema', id: '1', name: 'Test Medicine' }
      ];

      jest.spyOn(EnhancedEMAAPI, 'searchMedicines').mockResolvedValue(mockEmaResults);

      const results = await EnhancedMedicationService.searchMedications('test', 10, 'ema');

      expect(results).toEqual(mockEmaResults);
      expect(EnhancedEMAAPI.searchMedicines).toHaveBeenCalledWith('test', 10);
    });

    test('should remove duplicates correctly', () => {
      const medications = [
        { name: 'Medicine A', genericName: 'Generic A' },
        { name: 'Medicine A', genericName: 'Generic A' },
        { name: 'Medicine B', genericName: 'Generic B' }
      ];

      const uniqueResults = EnhancedMedicationService.removeDuplicates(medications);

      expect(uniqueResults).toHaveLength(2);
      expect(uniqueResults[0].name).toBe('Medicine A');
      expect(uniqueResults[1].name).toBe('Medicine B');
    });

    test('should sort medications by relevance correctly', () => {
      const medications = [
        { name: 'Generic Medicine', genericName: 'Generic Medicine' },
        { name: 'Medicine Brand', genericName: 'Generic Medicine' },
        { name: 'Other Medicine', genericName: 'Other Generic' }
      ];

      const sortedResults = EnhancedMedicationService.sortByRelevance(medications, 'Generic Medicine');

      expect(sortedResults[0].name).toBe('Generic Medicine');
      expect(sortedResults[1].name).toBe('Medicine Brand');
      expect(sortedResults[2].name).toBe('Other Medicine');
    });

    test('should calculate relevance scores correctly', () => {
      const medication = {
        name: 'Exact Match',
        genericName: 'Generic Match',
        brandNames: ['Brand Match'],
        atcClass: 'ATC Match'
      };

      const exactScore = EnhancedMedicationService.calculateRelevanceScore(medication, 'exact match');
      const partialScore = EnhancedMedicationService.calculateRelevanceScore(medication, 'exact');
      const brandScore = EnhancedMedicationService.calculateRelevanceScore(medication, 'brand');
      const atcScore = EnhancedMedicationService.calculateRelevanceScore(medication, 'atc');

      expect(exactScore).toBeGreaterThan(partialScore);
      expect(partialScore).toBeGreaterThan(brandScore);
      expect(brandScore).toBeGreaterThan(atcScore);
    });

    test('should get medication details with LLM summarization', async () => {
      const mockDetails = {
        source: 'ema',
        id: '1',
        name: 'Test Medicine'
      };

      jest.spyOn(EnhancedEMAAPI, 'getMedicineDetails').mockResolvedValue(mockDetails);
      jest.spyOn(MedicationSummarizer, 'summarizeWithOpenAI').mockResolvedValue('AI Summary');

      const details = await EnhancedMedicationService.getMedicationDetails('1', 'ema');

      expect(details).toEqual(mockDetails);
      expect(details.llmSummary).toBe('AI Summary');
      expect(EnhancedEMAAPI.getMedicineDetails).toHaveBeenCalledWith('1');
      expect(MedicationSummarizer.summarizeWithOpenAI).toHaveBeenCalledWith(mockDetails);
    });
  });

  describe('Error Handling', () => {
    test('should handle network errors gracefully', async () => {
      axios.get.mockRejectedValue(new Error('Network Error'));

      const results = await EnhancedEMAAPI.searchMedicines('test', 1);
      expect(results).toEqual([]);
    });

    test('should handle malformed API responses', async () => {
      axios.get.mockResolvedValue({ data: null });

      const results = await EnhancedEMAAPI.searchMedicines('test', 1);
      expect(results).toEqual([]);
    });

    test('should handle missing medication data gracefully', async () => {
      const mockResponse = {
        data: {
          id: '1',
          name: null,
          internationalNonProprietaryName: null
        }
      };

      axios.get
        .mockResolvedValueOnce(mockResponse)
        .mockResolvedValueOnce({ data: {} })
        .mockResolvedValueOnce({ data: {} });

      const details = await EnhancedEMAAPI.getMedicineDetails('1');

      expect(details.name).toBe('Unknown');
      expect(details.genericName).toBe('');
    });
  });

  describe('Performance and Caching', () => {
    test('should handle concurrent requests efficiently', async () => {
      const mockResponse = {
        data: {
          medicines: Array.from({ length: 10 }, (_, i) => ({
            id: i.toString(),
            name: `Medicine ${i}`,
            internationalNonProprietaryName: `Generic ${i}`
          }))
        }
      };

      axios.get.mockResolvedValue(mockResponse);

      const startTime = Date.now();
      const promises = Array.from({ length: 5 }, () => 
        EnhancedEMAAPI.searchMedicines('test', 10)
      );

      const results = await Promise.all(promises);
      const endTime = Date.now();

      expect(results).toHaveLength(5);
      expect(endTime - startTime).toBeLessThan(5000); // Should complete within 5 seconds
    });

    test('should handle large result sets', async () => {
      const mockResponse = {
        data: {
          medicines: Array.from({ length: 100 }, (_, i) => ({
            id: i.toString(),
            name: `Medicine ${i}`,
            internationalNonProprietaryName: `Generic ${i}`
          }))
        }
      };

      axios.get.mockResolvedValue(mockResponse);

      const results = await EnhancedEMAAPI.searchMedicines('test', 100);

      expect(results).toHaveLength(100);
      expect(results[0].name).toBe('Medicine 0');
      expect(results[99].name).toBe('Medicine 99');
    });
  });
});

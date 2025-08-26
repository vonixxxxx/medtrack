const axios = require('axios');

// API Base URLs and endpoints
const API_ENDPOINTS = {
  rxnorm: {
    base: 'https://rxnav.nlm.nih.gov/REST',
    search: '/drugs.json',
    allNames: '/allconcepts.json',
    rxcui: '/rxcui/{rxcui}/allrelated.json'
  },
  openfda: {
    base: 'https://api.fda.gov',
    search: '/drug/label.json',
    ndc: '/drug/ndc.json'
  },
  nhs: {
    base: 'https://directory.spineservices.nhs.uk/ORD/2-0-0',
    search: '/medicines',
    product: '/medicines/{id}'
  },
  ema: {
    base: 'https://www.ema.europa.eu/en/medicines/api',
    search: '/medicines',
    product: '/medicines/{id}'
  }
};

// Rate limiting and timeout configuration
const API_CONFIG = {
  timeout: 10000,
  maxRetries: 3,
  retryDelay: 1000
};

/**
 * RxNorm API integration for US drug data
 */
class RxNormAPI {
  static async searchDrugs(query, limit = 10) {
    try {
      const response = await axios.get(`${API_ENDPOINTS.rxnorm.base}${API_ENDPOINTS.rxnorm.search}`, {
        params: { name: query },
        timeout: API_CONFIG.timeout
      });

      if (!response.data || !response.data.drugGroup) {
        return [];
      }

      const concepts = response.data.drugGroup.conceptGroup || [];
      const results = [];

      concepts.forEach(group => {
        if (group.concept) {
          group.concept.forEach(concept => {
            results.push({
              source: 'rxnorm',
              id: concept.rxcui,
              name: concept.name,
              genericName: concept.name,
              brandNames: [],
              dosageForms: [],
              strengths: [],
              atcClass: concept.atc || [],
              identifiers: {
                rxcui: concept.rxcui,
                ndc: concept.ndc || []
              },
              sourceUrl: `https://rxnav.nlm.nih.gov/REST/rxcui/${concept.rxcui}/allrelated.json`
            });
          });
        }
      });

      return results.slice(0, limit);
    } catch (error) {
      console.error('RxNorm API error:', error.message);
      return [];
    }
  }

  static async getDrugDetails(rxcui) {
    try {
      const response = await axios.get(
        `${API_ENDPOINTS.rxnorm.base}${API_ENDPOINTS.rxnorm.rxcui.replace('{rxcui}', rxcui)}`,
        { timeout: API_CONFIG.timeout }
      );

      if (!response.data || !response.data.allRelatedGroup) {
        return null;
      }

      const conceptGroup = response.data.allRelatedGroup.conceptGroup;
      let details = {
        source: 'rxnorm',
        id: rxcui,
        name: '',
        genericName: '',
        brandNames: [],
        dosageForms: [],
        strengths: [],
        atcClass: [],
        identifiers: { rxcui }
      };

      // Extract information from concept groups
      conceptGroup.forEach(group => {
        if (group.concept) {
          group.concept.forEach(concept => {
            if (concept.tty === 'BN') { // Brand name
              details.brandNames.push(concept.name);
            } else if (concept.tty === 'IN') { // Ingredient
              details.genericName = concept.name;
            } else if (concept.tty === 'PIN') { // Precise ingredient
              details.name = concept.name;
            }
          });
        }
      });

      return details;
    } catch (error) {
      console.error('RxNorm details API error:', error.message);
      return null;
    }
  }
}

/**
 * openFDA API integration for US drug labeling and safety
 */
class OpenFDAAPI {
  static async searchDrugs(query, limit = 10) {
    try {
      const response = await axios.get(`${API_ENDPOINTS.openfda.base}${API_ENDPOINTS.openfda.search}`, {
        params: {
          search: `openfda.generic_name:${query}`,
          limit: limit
        },
        timeout: API_CONFIG.timeout
      });

      if (!response.data || !response.data.results) {
        return [];
      }

      return response.data.results.map(drug => ({
        source: 'openfda',
        id: drug.id,
        name: drug.openfda.generic_name?.[0] || drug.openfda.brand_name?.[0] || 'Unknown',
        genericName: drug.openfda.generic_name?.[0] || '',
        brandNames: drug.openfda.brand_name || [],
        dosageForms: drug.openfda.dosage_form || [],
        strengths: drug.openfda.substance_name || [],
        atcClass: [],
        identifiers: {
          ndc: drug.openfda.package_ndc || [],
          spl: drug.id
        },
        sourceUrl: `https://www.accessdata.fda.gov/spl/data/${drug.id}/${drug.id}.xml`,
        warnings: drug.warnings || [],
        adverseReactions: drug.adverse_reactions || [],
        drugInteractions: drug.drug_interactions || []
      }));
    } catch (error) {
      console.error('OpenFDA API error:', error.message);
      return [];
    }
  }

  static async getDrugDetails(ndc) {
    try {
      const response = await axios.get(`${API_ENDPOINTS.openfda.base}${API_ENDPOINTS.openfda.ndc}`, {
        params: { search: `package_ndc:${ndc}` },
        timeout: API_CONFIG.timeout
      });

      if (!response.data || !response.data.results || response.data.results.length === 0) {
        return null;
      }

      const drug = response.data.results[0];
      return {
        source: 'openfda',
        id: drug.id,
        name: drug.generic_name || drug.brand_name || 'Unknown',
        genericName: drug.generic_name || '',
        brandNames: drug.brand_name ? [drug.brand_name] : [],
        dosageForms: drug.dosage_form || [],
        strengths: drug.active_ingredients || [],
        atcClass: [],
        identifiers: { ndc: drug.package_ndc },
        sourceUrl: `https://www.accessdata.fda.gov/spl/data/${drug.id}/${drug.id}.xml`
      };
    } catch (error) {
      console.error('OpenFDA NDC API error:', error.message);
      return null;
    }
  }
}

/**
 * NHS dm+d API integration for UK drug data
 */
class NHSdmDAPI {
  static async searchDrugs(query, limit = 10) {
    try {
      const response = await axios.get(`${API_ENDPOINTS.nhs.base}${API_ENDPOINTS.nhs.search}`, {
        params: {
          search: query,
          limit: limit
        },
        timeout: API_CONFIG.timeout
      });

      if (!response.data || !response.data.entry) {
        return [];
      }

      return response.data.entry.map(entry => ({
        source: 'nhs_dmd',
        id: entry.resource.id,
        name: entry.resource.code?.text || 'Unknown',
        genericName: entry.resource.code?.text || '',
        brandNames: [],
        dosageForms: entry.resource.form?.text || [],
        strengths: [],
        atcClass: entry.resource.code?.coding?.[0]?.code || '',
        identifiers: {
          dmd: entry.resource.id,
          vmp: entry.resource.vmp || '',
          amp: entry.resource.amp || ''
        },
        sourceUrl: `https://directory.spineservices.nhs.uk/ORD/2-0-0/medicines/${entry.resource.id}`,
        nhsUrl: entry.resource.url || ''
      }));
    } catch (error) {
      console.error('NHS dm+d API error:', error.message);
      return [];
    }
  }

  static async getDrugDetails(id) {
    try {
      const response = await axios.get(
        `${API_ENDPOINTS.nhs.base}${API_ENDPOINTS.nhs.product.replace('{id}', id)}`,
        { timeout: API_CONFIG.timeout }
      );

      if (!response.data) {
        return null;
      }

      const resource = response.data;
      return {
        source: 'nhs_dmd',
        id: resource.id,
        name: resource.code?.text || 'Unknown',
        genericName: resource.code?.text || '',
        brandNames: [],
        dosageForms: resource.form?.text || [],
        strengths: [],
        atcClass: resource.code?.coding?.[0]?.code || '',
        identifiers: {
          dmd: resource.id,
          vmp: resource.vmp || '',
          amp: resource.amp || ''
        },
        sourceUrl: `https://directory.spineservices.nhs.uk/ORD/2-0-0/medicines/${resource.id}`,
        nhsUrl: resource.url || ''
      };
    } catch (error) {
      console.error('NHS dm+d details API error:', error.message);
      return null;
    }
  }
}

/**
 * EMA API integration for EU drug data
 */
class EMAAPI {
  static async searchDrugs(query, limit = 10) {
    try {
      const response = await axios.get(`${API_ENDPOINTS.ema.base}${API_ENDPOINTS.ema.search}`, {
        params: {
          search: query,
          limit: limit
        },
        timeout: API_CONFIG.timeout
      });

      if (!response.data || !response.data.medicines) {
        return [];
      }

      return response.data.medicines.map(medicine => ({
        source: 'ema',
        id: medicine.id,
        name: medicine.name || 'Unknown',
        genericName: medicine.internationalNonProprietaryName || '',
        brandNames: medicine.brandNames || [],
        dosageForms: medicine.pharmaceuticalForm || [],
        strengths: medicine.strength || [],
        atcClass: medicine.atcCode || '',
        identifiers: {
          ema: medicine.id,
          emaProductNumber: medicine.emaProductNumber || '',
          marketingAuthorisationNumber: medicine.marketingAuthorisationNumber || ''
        },
        sourceUrl: `https://www.ema.europa.eu/en/medicines/human/EPAR/${medicine.id}`,
        authorizationStatus: medicine.authorizationStatus || '',
        authorizationDate: medicine.authorizationDate || ''
      }));
    } catch (error) {
      console.error('EMA API error:', error.message);
      return [];
    }
  }

  static async getDrugDetails(id) {
    try {
      const response = await axios.get(
        `${API_ENDPOINTS.ema.base}${API_ENDPOINTS.ema.product.replace('{id}', id)}`,
        { timeout: API_CONFIG.timeout }
      );

      if (!response.data) {
        return null;
      }

      const medicine = response.data;
      return {
        source: 'ema',
        id: medicine.id,
        name: medicine.name || 'Unknown',
        genericName: medicine.internationalNonProprietaryName || '',
        brandNames: medicine.brandNames || [],
        dosageForms: medicine.pharmaceuticalForm || [],
        strengths: medicine.strength || [],
        atcClass: medicine.atcCode || '',
        identifiers: {
          ema: medicine.id,
          emaProductNumber: medicine.emaProductNumber || '',
          marketingAuthorisationNumber: medicine.marketingAuthorisationNumber || ''
        },
        sourceUrl: `https://www.ema.europa.eu/en/medicines/human/EPAR/${medicine.id}`,
        authorizationStatus: medicine.authorizationStatus || '',
        authorizationDate: medicine.authorizationDate || ''
      };
    } catch (error) {
      console.error('EMA details API error:', error.message);
      return null;
    }
  }
}

/**
 * Main drug search function that queries all sources in parallel
 */
async function searchAllDrugSources(query, limit = 10) {
  try {
    const searchPromises = [
      RxNormAPI.searchDrugs(query, limit),
      OpenFDAAPI.searchDrugs(query, limit),
      NHSdmDAPI.searchDrugs(query, limit),
      EMAAPI.searchDrugs(query, limit)
    ];

    const results = await Promise.allSettled(searchPromises);
    
    const allResults = [];
    results.forEach((result, index) => {
      if (result.status === 'fulfilled' && result.value) {
        allResults.push(...result.value);
      }
    });

    return allResults;
  } catch (error) {
    console.error('Multi-source drug search error:', error.message);
    return [];
  }
}

/**
 * Get detailed drug information from a specific source
 */
async function getDrugDetails(source, id) {
  try {
    switch (source) {
      case 'rxnorm':
        return await RxNormAPI.getDrugDetails(id);
      case 'openfda':
        return await OpenFDAAPI.getDrugDetails(id);
      case 'nhs_dmd':
        return await NHSdmDAPI.getDrugDetails(id);
      case 'ema':
        return await EMAAPI.getDrugDetails(id);
      default:
        return null;
    }
  } catch (error) {
    console.error(`Drug details API error for ${source}:`, error.message);
    return null;
  }
}

module.exports = {
  RxNormAPI,
  OpenFDAAPI,
  NHSdmDAPI,
  EMAAPI,
  searchAllDrugSources,
  getDrugDetails,
  API_ENDPOINTS,
  API_CONFIG
};

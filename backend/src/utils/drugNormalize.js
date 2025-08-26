/**
 * Drug data normalization utility
 * Unifies data from different sources (RxNorm, openFDA, NHS dm+d, EMA) into a consistent format
 */

/**
 * Normalize drug data from multiple sources into a unified schema
 */
function normalizeDrugData(rawData) {
  if (!rawData || !Array.isArray(rawData)) {
    return [];
  }

  return rawData.map(drug => {
    const normalized = {
      id: drug.id || '',
      name: drug.name || 'Unknown',
      genericName: drug.genericName || drug.name || '',
      brandNames: Array.isArray(drug.brandNames) ? drug.brandNames : [],
      dosageForms: Array.isArray(drug.dosageForms) ? drug.dosageForms : [],
      strengths: Array.isArray(drug.strengths) ? drug.strengths : [],
      atcClass: drug.atcClass || '',
      source: drug.source || 'unknown',
      sourceUrl: drug.sourceUrl || '',
      identifiers: normalizeIdentifiers(drug.identifiers, drug.source),
      metadata: extractMetadata(drug),
      lastUpdated: new Date().toISOString()
    };

    // Add source-specific information
    switch (drug.source) {
      case 'rxnorm':
        normalized.usData = extractUSData(drug);
        break;
      case 'openfda':
        normalized.usData = extractUSData(drug);
        normalized.safety = extractSafetyData(drug);
        break;
      case 'nhs_dmd':
        normalized.ukData = extractUKData(drug);
        break;
      case 'ema':
        normalized.euData = extractEUData(drug);
        break;
    }

    return normalized;
  });
}

/**
 * Normalize identifiers based on source
 */
function normalizeIdentifiers(identifiers, source) {
  if (!identifiers) return {};

  const normalized = {};
  
  switch (source) {
    case 'rxnorm':
      normalized.rxcui = identifiers.rxcui || '';
      normalized.ndc = Array.isArray(identifiers.ndc) ? identifiers.ndc : [];
      break;
    case 'openfda':
      normalized.ndc = Array.isArray(identifiers.ndc) ? identifiers.ndc : [];
      normalized.spl = identifiers.spl || '';
      break;
    case 'nhs_dmd':
      normalized.dmd = identifiers.dmd || '';
      normalized.vmp = identifiers.vmp || '';
      normalized.amp = identifiers.amp || '';
      break;
    case 'ema':
      normalized.ema = identifiers.ema || '';
      normalized.emaProductNumber = identifiers.emaProductNumber || '';
      normalized.marketingAuthorisationNumber = identifiers.marketingAuthorisationNumber || '';
      break;
  }

  return normalized;
}

/**
 * Extract US-specific drug data
 */
function extractUSData(drug) {
  return {
    fdaApproved: drug.source === 'openfda',
    ndcCodes: Array.isArray(drug.identifiers?.ndc) ? drug.identifiers.ndc : [],
    rxcui: drug.identifiers?.rxcui || '',
    splId: drug.identifiers?.spl || '',
    sourceUrl: drug.sourceUrl || ''
  };
}

/**
 * Extract UK-specific drug data
 */
function extractUKData(drug) {
  return {
    nhsApproved: true,
    dmdId: drug.identifiers?.dmd || '',
    vmpId: drug.identifiers?.vmp || '',
    ampId: drug.identifiers?.amp || '',
    nhsUrl: drug.nhsUrl || '',
    sourceUrl: drug.sourceUrl || ''
  };
}

/**
 * Extract EU-specific drug data
 */
function extractEUData(drug) {
  return {
    emaApproved: drug.authorizationStatus === 'Authorised',
    emaId: drug.identifiers?.ema || '',
    emaProductNumber: drug.identifiers?.emaProductNumber || '',
    marketingAuthorisationNumber: drug.identifiers?.marketingAuthorisationNumber || '',
    authorizationStatus: drug.authorizationStatus || '',
    authorizationDate: drug.authorizationDate || '',
    sourceUrl: drug.sourceUrl || ''
  };
}

/**
 * Extract safety and clinical data
 */
function extractSafetyData(drug) {
  return {
    warnings: Array.isArray(drug.warnings) ? drug.warnings : [],
    adverseReactions: Array.isArray(drug.adverseReactions) ? drug.adverseReactions : [],
    drugInteractions: Array.isArray(drug.drugInteractions) ? drug.drugInteractions : [],
    contraindications: Array.isArray(drug.contraindications) ? drug.contraindications : []
  };
}

/**
 * Extract additional metadata
 */
function extractMetadata(drug) {
  const metadata = {
    source: drug.source,
    searchScore: drug.searchScore || 0,
    popularity: drug.popularity || 0
  };

  // Add source-specific metadata
  if (drug.source === 'openfda' && drug.openfda) {
    metadata.manufacturer = drug.openfda.manufacturer_name?.[0] || '';
    metadata.substance = drug.openfda.substance_name?.[0] || '';
    metadata.route = drug.openfda.route?.[0] || '';
  }

  if (drug.source === 'ema') {
    metadata.therapeuticArea = drug.therapeuticArea || '';
    metadata.orphanDesignation = drug.orphanDesignation || false;
  }

  return metadata;
}

/**
 * Deduplicate drugs based on name similarity and source
 */
function deduplicateDrugs(normalizedData) {
  const seen = new Map();
  const deduplicated = [];

  normalizedData.forEach(drug => {
    const key = drug.name.toLowerCase().trim();
    
    if (!seen.has(key)) {
      seen.set(key, drug);
      deduplicated.push(drug);
    } else {
      // Merge with existing drug if it has additional information
      const existing = seen.get(key);
      const merged = mergeDrugData(existing, drug);
      seen.set(key, merged);
      
      // Update the existing entry in deduplicated array
      const index = deduplicated.findIndex(d => d.id === existing.id);
      if (index !== -1) {
        deduplicated[index] = merged;
      }
    }
  });

  return deduplicated;
}

/**
 * Merge two drug entries, combining their information
 */
function mergeDrugData(drug1, drug2) {
  const merged = { ...drug1 };

  // Merge arrays, removing duplicates
  merged.brandNames = [...new Set([...drug1.brandNames, ...drug2.brandNames])];
  merged.dosageForms = [...new Set([...drug1.dosageForms, ...drug2.dosageForms])];
  merged.strengths = [...new Set([...drug1.strengths, ...drug2.strengths])];

  // Merge identifiers
  merged.identifiers = { ...drug1.identifiers, ...drug2.identifiers };

  // Merge source-specific data
  if (drug1.usData || drug2.usData) {
    merged.usData = { ...drug1.usData, ...drug2.usData };
  }
  if (drug1.ukData || drug2.ukData) {
    merged.ukData = { ...drug1.ukData, ...drug2.ukData };
  }
  if (drug1.euData || drug2.euData) {
    merged.euData = { ...drug1.euData, ...drug2.euData };
  }

  // Merge safety data
  if (drug1.safety || drug2.safety) {
    merged.safety = { ...drug1.safety, ...drug2.safety };
  }

  // Update metadata to reflect multiple sources
  merged.metadata.sources = [drug1.source, drug2.source];
  merged.metadata.lastUpdated = new Date().toISOString();

  return merged;
}

/**
 * Sort drugs by relevance and source priority
 */
function sortDrugsByRelevance(drugs, query) {
  return drugs.sort((a, b) => {
    // Exact name match gets highest priority
    const aExactMatch = a.name.toLowerCase() === query.toLowerCase();
    const bExactMatch = b.name.toLowerCase() === query.toLowerCase();
    
    if (aExactMatch && !bExactMatch) return -1;
    if (!aExactMatch && bExactMatch) return 1;

    // Starts with query gets second priority
    const aStartsWith = a.name.toLowerCase().startsWith(query.toLowerCase());
    const bStartsWith = b.name.toLowerCase().startsWith(query.toLowerCase());
    
    if (aStartsWith && !bStartsWith) return -1;
    if (!aStartsWith && bStartsWith) return 1;

    // Source priority: RxNorm > openFDA > NHS > EMA
    const sourcePriority = { rxnorm: 4, openfda: 3, nhs_dmd: 2, ema: 1 };
    const aPriority = sourcePriority[a.source] || 0;
    const bPriority = sourcePriority[b.source] || 0;
    
    if (aPriority !== bPriority) {
      return bPriority - aPriority;
    }

    // Finally, sort by name length (shorter names first)
    return a.name.length - b.name.length;
  });
}

/**
 * Main normalization function
 */
function normalizeAndSortDrugData(rawData, query = '') {
  try {
    // Normalize the data
    const normalized = normalizeDrugData(rawData);
    
    // Deduplicate based on name similarity
    const deduplicated = deduplicateDrugs(normalized);
    
    // Sort by relevance
    const sorted = sortDrugsByRelevance(deduplicated, query);
    
    return sorted;
  } catch (error) {
    console.error('Drug data normalization error:', error);
    return [];
  }
}

module.exports = {
  normalizeDrugData,
  normalizeIdentifiers,
  extractUSData,
  extractUKData,
  extractEUData,
  extractSafetyData,
  extractMetadata,
  deduplicateDrugs,
  mergeDrugData,
  sortDrugsByRelevance,
  normalizeAndSortDrugData
};

/**
 * NHS Dictionary of Medicines and Devices (dm+d) Adapter
 * Authoritative UK medication data source
 * Provides VMP/AMP mapping with full pharmaceutical details
 */

const BaseAdapter = require('../BaseAdapter');
const axios = require('axios');
const xml2js = require('xml2js');

class DMDAdapter extends BaseAdapter {
  constructor(config = {}) {
    super('dmd', {
      baseUrl: 'https://services.nhsbsa.nhs.uk/nww-dm-api',
      version: '2025-08',
      timeout: 30000,
      retryAttempts: 3,
      ...config
    });
    
    this.vmps = new Map(); // Virtual Medicinal Products
    this.amps = new Map(); // Actual Medicinal Products
    this.vtms = new Map(); // Virtual Therapeutic Moieties
    this.routeMap = this.initializeRouteMap();
    this.formMap = this.initializeDoseFormMap();
  }

  async initialize() {
    this.logger.info('Initializing NHS dm+d adapter');
    
    // Verify API connectivity
    try {
      const response = await axios.get(`${this.config.baseUrl}/health`, {
        timeout: this.config.timeout
      });
      
      if (response.status !== 200) {
        throw new Error(`dm+d API health check failed: ${response.status}`);
      }
      
      this.logger.info('dm+d API connectivity verified');
    } catch (error) {
      this.logger.error('dm+d API connectivity failed', { error: error.message });
      throw error;
    }
  }

  async fetchData() {
    this.logger.info('Fetching data from NHS dm+d');
    
    const data = {
      vtms: await this.fetchVTMs(),
      vmps: await this.fetchVMPs(),
      amps: await this.fetchAMPs(),
      routes: await this.fetchRoutes(),
      forms: await this.fetchDoseForms(),
      units: await this.fetchUnits()
    };
    
    this.logger.info(`Fetched ${data.vtms.length} VTMs, ${data.vmps.length} VMPs, ${data.amps.length} AMPs`);
    
    return data;
  }

  /**
   * Fetch Virtual Therapeutic Moieties (ingredient level)
   */
  async fetchVTMs() {
    try {
      const response = await this.makeAPIRequest('/vtm');
      return response.data.vtms || [];
    } catch (error) {
      this.logger.error('Failed to fetch VTMs', { error: error.message });
      return [];
    }
  }

  /**
   * Fetch Virtual Medicinal Products (generic products)
   */
  async fetchVMPs() {
    try {
      const response = await this.makeAPIRequest('/vmp');
      return response.data.vmps || [];
    } catch (error) {
      this.logger.error('Failed to fetch VMPs', { error: error.message });
      return [];
    }
  }

  /**
   * Fetch Actual Medicinal Products (branded products)
   */
  async fetchAMPs() {
    try {
      const response = await this.makeAPIRequest('/amp');
      return response.data.amps || [];
    } catch (error) {
      this.logger.error('Failed to fetch AMPs', { error: error.message });
      return [];
    }
  }

  /**
   * Fetch route of administration mappings
   */
  async fetchRoutes() {
    try {
      const response = await this.makeAPIRequest('/routes');
      return response.data.routes || [];
    } catch (error) {
      this.logger.error('Failed to fetch routes', { error: error.message });
      return [];
    }
  }

  /**
   * Fetch dose form mappings
   */
  async fetchDoseForms() {
    try {
      const response = await this.makeAPIRequest('/forms');
      return response.data.forms || [];
    } catch (error) {
      this.logger.error('Failed to fetch dose forms', { error: error.message });
      return [];
    }
  }

  /**
   * Fetch unit mappings
   */
  async fetchUnits() {
    try {
      const response = await this.makeAPIRequest('/units');
      return response.data.units || [];
    } catch (error) {
      this.logger.error('Failed to fetch units', { error: error.message });
      return [];
    }
  }

  async transformData(rawData) {
    this.logger.info('Transforming dm+d data to canonical format');
    
    const medications = [];
    const products = [];
    const strengths = [];
    const rules = [];
    
    // Process VTMs into canonical medications
    for (const vtm of rawData.vtms) {
      const medication = await this.transformVTM(vtm);
      if (medication) {
        medications.push(medication);
      }
    }
    
    // Process VMPs into generic products
    for (const vmp of rawData.vmps) {
      const product = await this.transformVMP(vmp);
      if (product) {
        products.push(product);
        
        // Extract strengths from VMP
        const vmpStrengths = await this.extractVMPStrengths(vmp, product.id);
        strengths.push(...vmpStrengths);
        
        // Extract rules from VMP
        const vmpRules = await this.extractVMPRules(vmp, product.id);
        if (vmpRules) {
          rules.push(vmpRules);
        }
      }
    }
    
    // Process AMPs into branded products
    for (const amp of rawData.amps) {
      const product = await this.transformAMP(amp);
      if (product) {
        products.push(product);
        
        // Extract strengths from AMP
        const ampStrengths = await this.extractAMPStrengths(amp, product.id);
        strengths.push(...ampStrengths);
        
        // Extract rules from AMP
        const ampRules = await this.extractAMPRules(amp, product.id);
        if (ampRules) {
          rules.push(ampRules);
        }
      }
    }
    
    return {
      medications,
      products,
      strengths,
      rules
    };
  }

  /**
   * Transform VTM to canonical medication
   */
  async transformVTM(vtm) {
    try {
      return {
        id: `dmd-vtm-${vtm.vtmid}`,
        genericName: this.normalizeGenericName(vtm.nm),
        atcCode: vtm.atc_code,
        classHuman: this.mapATCToHumanClass(vtm.atc_code),
        synonyms: this.extractSynonyms(vtm),
        sourceRefs: {
          dmd: [`VTM:${vtm.vtmid}`],
          version: this.config.version
        }
      };
    } catch (error) {
      this.logger.error(`Failed to transform VTM ${vtm.vtmid}`, { error: error.message });
      return null;
    }
  }

  /**
   * Transform VMP to canonical product
   */
  async transformVMP(vmp) {
    try {
      const medicationId = `dmd-vtm-${vmp.vtmid}`;
      
      return {
        id: `dmd-vmp-${vmp.vpid}`,
        medicationId,
        brandName: null, // VMPs are generic
        route: this.mapDMDRoute(vmp.route_cd),
        doseForm: this.mapDMDForm(vmp.form_cd),
        intakeType: this.mapToIntakeType(vmp.route_cd, vmp.form_cd),
        defaultPlaces: this.getDefaultPlaces(vmp.route_cd),
        allowedFrequencies: this.getAllowedFrequencies(vmp),
        provenance: {
          dmd: {
            vpid: vmp.vpid,
            vtmid: vmp.vtmid,
            version: this.config.version
          }
        }
      };
    } catch (error) {
      this.logger.error(`Failed to transform VMP ${vmp.vpid}`, { error: error.message });
      return null;
    }
  }

  /**
   * Transform AMP to canonical branded product
   */
  async transformAMP(amp) {
    try {
      const medicationId = `dmd-vtm-${amp.vtmid}`;
      
      return {
        id: `dmd-amp-${amp.apid}`,
        medicationId,
        brandName: this.normalizeBrandName(amp.nm),
        route: this.mapDMDRoute(amp.route_cd),
        doseForm: this.mapDMDForm(amp.form_cd),
        intakeType: this.mapToIntakeType(amp.route_cd, amp.form_cd),
        defaultPlaces: this.getDefaultPlaces(amp.route_cd),
        allowedFrequencies: this.getAllowedFrequencies(amp),
        provenance: {
          dmd: {
            apid: amp.apid,
            vpid: amp.vpid,
            vtmid: amp.vtmid,
            version: this.config.version
          }
        }
      };
    } catch (error) {
      this.logger.error(`Failed to transform AMP ${amp.apid}`, { error: error.message });
      return null;
    }
  }

  /**
   * Extract VMP strengths
   */
  async extractVMPStrengths(vmp, productId) {
    const strengths = [];
    
    if (vmp.virtual_product_ingredient) {
      for (const ingredient of vmp.virtual_product_ingredient) {
        if (ingredient.strength_val_nmrtr_val && ingredient.strength_val_nmrtr_uom_cd) {
          strengths.push({
            id: `dmd-vmp-str-${vmp.vpid}-${ingredient.ingredient_substance_id}`,
            productId,
            value: parseFloat(ingredient.strength_val_nmrtr_val),
            unit: this.mapDMDUnit(ingredient.strength_val_nmrtr_uom_cd),
            per: ingredient.strength_val_dnmtr_val ? 
              `${ingredient.strength_val_dnmtr_val} ${this.mapDMDUnit(ingredient.strength_val_dnmtr_uom_cd)}` : 
              null,
            frequency: this.getDefaultFrequency(vmp),
            label: this.buildStrengthLabel(ingredient),
            provenance: {
              dmd: {
                vpid: vmp.vpid,
                ingredient_id: ingredient.ingredient_substance_id
              }
            }
          });
        }
      }
    }
    
    return strengths;
  }

  /**
   * Initialize route mapping
   */
  initializeRouteMap() {
    return {
      '1': 'oral',
      '2': 'sublingual',
      '3': 'rectal',
      '4': 'vaginal',
      '5': 'topical',
      '6': 'subcutaneous',
      '7': 'intramuscular',
      '8': 'intravenous',
      '9': 'inhalation',
      '10': 'nasal',
      '11': 'ophthalmic',
      '12': 'otic',
      '13': 'transdermal',
      '14': 'epidural',
      '15': 'intrathecal',
      '16': 'intraperitoneal',
      '17': 'intradermal',
      '18': 'intraosseous',
      '19': 'intrapleural',
      '20': 'intravesical'
    };
  }

  /**
   * Initialize dose form mapping
   */
  initializeDoseFormMap() {
    return {
      '1': 'tablet',
      '2': 'capsule',
      '3': 'solution',
      '4': 'suspension',
      '5': 'injection',
      '6': 'cream',
      '7': 'ointment',
      '8': 'gel',
      '9': 'patch',
      '10': 'inhaler',
      '11': 'spray',
      '12': 'drops',
      '13': 'pessary',
      '14': 'suppository',
      '15': 'foam',
      '16': 'lotion',
      '17': 'shampoo',
      '18': 'powder',
      '19': 'granules',
      '20': 'syrup'
    };
  }

  /**
   * Map dm+d route code to canonical route
   */
  mapDMDRoute(routeCode) {
    return this.routeMap[routeCode] || 'oral';
  }

  /**
   * Map dm+d form code to canonical dose form
   */
  mapDMDForm(formCode) {
    return this.formMap[formCode] || 'tablet';
  }

  /**
   * Map route and form to intake type
   */
  mapToIntakeType(routeCode, formCode) {
    const route = this.mapDMDRoute(routeCode);
    const form = this.mapDMDForm(formCode);
    
    if (route === 'inhalation') return 'Inhaler';
    if (route === 'injection' || route === 'subcutaneous' || route === 'intramuscular' || route === 'intravenous') return 'Injection';
    if (route === 'topical' || form === 'cream' || form === 'ointment' || form === 'gel') return 'Topical';
    if (route === 'ophthalmic' || route === 'otic' || form === 'drops') return 'Drops';
    if (form === 'patch') return 'Patch';
    if (form === 'tablet' || form === 'capsule') return 'Pill/Tablet';
    if (form === 'solution' || form === 'suspension' || form === 'syrup') return 'Liquid';
    
    return 'Other';
  }

  /**
   * Get default administration places based on route
   */
  getDefaultPlaces(routeCode) {
    const route = this.mapDMDRoute(routeCode);
    
    switch (route) {
      case 'intravenous':
      case 'intramuscular':
      case 'epidural':
      case 'intrathecal':
        return ['at clinic', 'hospital administration'];
      
      case 'subcutaneous':
        return ['at home', 'self administered', 'at clinic', 'caregiver administered'];
      
      case 'oral':
      case 'topical':
      case 'inhalation':
        return ['at home', 'self administered'];
      
      default:
        return ['at home', 'self administered', 'at clinic'];
    }
  }

  /**
   * Get allowed frequencies for a product
   */
  getAllowedFrequencies(product) {
    // Base frequencies that apply to most medications
    const baseFrequencies = ['daily', 'twice daily', 'three times daily'];
    
    // Add route-specific frequencies
    const route = this.mapDMDRoute(product.route_cd);
    
    switch (route) {
      case 'subcutaneous':
        return ['weekly', 'twice weekly', 'daily'];
      
      case 'inhalation':
        return ['twice daily', 'once daily', 'when needed'];
      
      case 'topical':
        return ['twice daily', 'three times daily', 'four times daily'];
      
      case 'ophthalmic':
        return ['daily', 'twice daily', 'three times daily', 'four times daily'];
      
      default:
        return baseFrequencies;
    }
  }

  /**
   * Extract synonyms from dm+d data
   */
  extractSynonyms(vtm) {
    const synonyms = [];
    
    // Add common abbreviations and alternative names
    if (vtm.abbreviation) {
      synonyms.push(vtm.abbreviation.toLowerCase());
    }
    
    // Add generic name variations
    const name = vtm.nm.toLowerCase();
    synonyms.push(name);
    
    // Add common medication class terms
    if (vtm.atc_code) {
      const classTerms = this.getATCClassTerms(vtm.atc_code);
      synonyms.push(...classTerms);
    }
    
    return [...new Set(synonyms)];
  }

  /**
   * Get ATC class terms for synonyms
   */
  getATCClassTerms(atcCode) {
    const terms = [];
    
    // Map common ATC codes to searchable terms
    const atcMap = {
      'A10BJ': ['glp-1', 'glp1', 'incretin'],
      'M01AE': ['nsaid', 'anti-inflammatory'],
      'C10AA': ['statin', 'cholesterol'],
      'A02BC': ['ppi', 'proton pump inhibitor'],
      'R03AC': ['beta2-agonist', 'bronchodilator'],
      'N02BE': ['analgesic', 'painkiller'],
      'J01': ['antibiotic', 'antimicrobial']
    };
    
    // Check for partial matches
    for (const [code, classTerms] of Object.entries(atcMap)) {
      if (atcCode.startsWith(code)) {
        terms.push(...classTerms);
        break;
      }
    }
    
    return terms;
  }

  /**
   * Make API request with retry logic
   */
  async makeAPIRequest(endpoint, params = {}) {
    let lastError;
    
    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      try {
        const response = await axios.get(`${this.config.baseUrl}${endpoint}`, {
          params,
          timeout: this.config.timeout,
          headers: {
            'Accept': 'application/json',
            'User-Agent': 'MedTrack-DMD-Adapter/1.0'
          }
        });
        
        return response;
        
      } catch (error) {
        lastError = error;
        
        if (attempt < this.config.retryAttempts) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          this.logger.warn(`API request failed, retrying in ${delay}ms`, { 
            attempt, 
            endpoint, 
            error: error.message 
          });
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError;
  }

  async validateData(transformedData) {
    this.logger.info('Validating dm+d transformed data');
    
    const { medications, products, strengths, rules } = transformedData;
    
    // Validate medications
    const validMedications = medications.filter(med => {
      if (!med.genericName || med.genericName.length < 2) {
        this.logger.warn(`Invalid medication: missing or short generic name`, { med });
        return false;
      }
      return true;
    });
    
    // Validate products
    const validProducts = products.filter(product => {
      if (!product.medicationId || !product.route || !product.doseForm) {
        this.logger.warn(`Invalid product: missing required fields`, { product });
        return false;
      }
      return true;
    });
    
    // Validate strengths
    const validStrengths = strengths.filter(strength => {
      if (!strength.productId || !strength.value || !strength.unit) {
        this.logger.warn(`Invalid strength: missing required fields`, { strength });
        return false;
      }
      if (strength.value <= 0 || strength.value > 100000) {
        this.logger.warn(`Invalid strength: value out of range`, { strength });
        return false;
      }
      return true;
    });
    
    this.logger.info(`Validation complete: ${validMedications.length} medications, ${validProducts.length} products, ${validStrengths.length} strengths`);
    
    return {
      medications: validMedications,
      products: validProducts,
      strengths: validStrengths,
      rules
    };
  }

  /**
   * Normalize generic name for consistency
   */
  normalizeGenericName(name) {
    return name
      .toLowerCase()
      .trim()
      .replace(/\s+/g, ' ')
      .replace(/[^\w\s-]/g, '');
  }

  /**
   * Normalize brand name for consistency
   */
  normalizeBrandName(name) {
    return name
      .trim()
      .replace(/\s+/g, ' ');
  }

  /**
   * Map ATC code to human-readable class
   */
  mapATCToHumanClass(atcCode) {
    if (!atcCode) return null;
    
    const classMap = {
      'A10BJ': 'GLP-1 receptor agonist',
      'M01AE': 'NSAID',
      'C10AA': 'HMG CoA reductase inhibitor',
      'A02BC': 'Proton pump inhibitor',
      'R03AC': 'Beta2-adrenergic agonist',
      'N02BE': 'Analgesic',
      'J01CA': 'Penicillin'
    };
    
    // Try exact match first
    if (classMap[atcCode]) {
      return classMap[atcCode];
    }
    
    // Try partial matches
    for (const [code, className] of Object.entries(classMap)) {
      if (atcCode.startsWith(code)) {
        return className;
      }
    }
    
    return null;
  }
}

module.exports = DMDAdapter;

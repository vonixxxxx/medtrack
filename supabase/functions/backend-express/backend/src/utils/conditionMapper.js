/**
 * Maps condition names to Patient model boolean fields
 * Used to convert parsedData.conditions array into Patient boolean columns
 */

/**
 * Universal condition to database column mapping
 * Maps normalized condition names to Patient model boolean fields
 */
const CONDITION_MAP = {
  // Diabetes
  'type 2 diabetes': 't2dm',
  'type 2 diabetes mellitus': 't2dm',
  't2dm': 't2dm',
  'diabetes type 2': 't2dm',
  'prediabetes': 'prediabetes',
  'impaired glucose tolerance': 'prediabetes',
  'igt': 'prediabetes',
  
  // Cardiovascular
  'hypertension': 'hypertension',
  'htn': 'htn',
  'high blood pressure': 'hypertension',
  'dyslipidaemia': 'dyslipidaemia',
  'dyslipidemia': 'dyslipidaemia',
  'high cholesterol': 'dyslipidaemia',
  'hyperlipidemia': 'dyslipidaemia',
  'ascvd': 'ascvd',
  'atherosclerotic cardiovascular disease': 'ascvd',
  'ischaemic heart disease': 'ischaemic_heart_disease',
  'ischemic heart disease': 'ischaemic_heart_disease',
  'ihd': 'ischaemic_heart_disease',
  'coronary artery disease': 'ischaemic_heart_disease',
  'cad': 'ischaemic_heart_disease',
  'heart failure': 'heart_failure',
  'hf': 'heart_failure',
  'cerebrovascular disease': 'cerebrovascular_disease',
  'stroke': 'cerebrovascular_disease',
  'cva': 'cerebrovascular_disease',
  'pulmonary hypertension': 'pulmonary_hypertension',
  'dvt': 'dvt',
  'deep vein thrombosis': 'dvt',
  'pe': 'pe',
  'pulmonary embolism': 'pe',
  
  // Sleep and respiratory
  'osa': 'osa',
  'obstructive sleep apnea': 'osa',
  'obstructive sleep apnoea': 'osa',
  'sleep apnoea': 'osa',
  'sleep apnea': 'osa',
  'sleep studies': 'sleep_studies',
  'cpap': 'cpap',
  'asthma': 'asthma',
  
  // Renal
  'ckd': 'ckd',
  'chronic kidney disease': 'ckd',
  'kidney stones': 'kidney_stones',
  'nephrolithiasis': 'kidney_stones',
  
  // Gastrointestinal
  'gord': 'gord',
  'gerd': 'gord',
  'gastroesophageal reflux disease': 'gord',
  
  // Metabolic
  'masld': 'masld',
  'metabolic dysfunction associated steatotic liver disease': 'masld',
  'nafld': 'masld',
  'non-alcoholic fatty liver disease': 'masld',
  'obesity': 'obesity',
  
  // Reproductive
  'infertility': 'infertility',
  'pcos': 'pcos',
  'polycystic ovary syndrome': 'pcos',
  
  // Mental health
  'anxiety': 'anxiety',
  'depression': 'depression',
  'bipolar disorder': 'bipolar_disorder',
  'emotional eating': 'emotional_eating',
  'schizoaffective disorder': 'schizoaffective_disorder',
  
  // Musculoskeletal
  'oa knee': 'oa_knee',
  'osteoarthritis knee': 'oa_knee',
  'knee osteoarthritis': 'oa_knee',
  'oa hip': 'oa_hip',
  'osteoarthritis hip': 'oa_hip',
  'hip osteoarthritis': 'oa_hip',
  'limited mobility': 'limited_mobility',
  'lymphoedema': 'lymphoedema',
  'lymphedema': 'lymphoedema',
  
  // Endocrine
  'thyroid disorder': 'thyroid_disorder',
  'hypothyroidism': 'thyroid_disorder',
  'hyperthyroidism': 'thyroid_disorder',
  
  // Neurological
  'iih': 'iih',
  'idiopathic intracranial hypertension': 'iih',
  'epilepsy': 'epilepsy',
  'functional neurological disorder': 'functional_neurological_disorder',
  'fnd': 'functional_neurological_disorder',
  
  // Oncology
  'cancer': 'cancer',
  'malignancy': 'cancer',
  
  // Bariatric
  'gastric band': 'bariatric_gastric_band',
  'gastric sleeve': 'bariatric_sleeve',
  'sleeve gastrectomy': 'bariatric_sleeve',
  'gastric bypass': 'bariatric_bypass',
  'roux-en-y': 'bariatric_bypass',
  'gastric balloon': 'bariatric_balloon',
};

/**
 * All condition fields in the Patient model
 * These must be initialized to 0 before mapping
 */
const CONDITION_FIELDS = [
  't2dm',
  'prediabetes',
  'htn',
  'hypertension',
  'dyslipidaemia',
  'ascvd',
  'ckd',
  'osa',
  'sleep_studies',
  'cpap',
  'asthma',
  'ischaemic_heart_disease',
  'heart_failure',
  'cerebrovascular_disease',
  'pulmonary_hypertension',
  'dvt',
  'pe',
  'gord',
  'kidney_stones',
  'masld',
  'infertility',
  'pcos',
  'anxiety',
  'depression',
  'bipolar_disorder',
  'emotional_eating',
  'schizoaffective_disorder',
  'oa_knee',
  'oa_hip',
  'limited_mobility',
  'lymphoedema',
  'thyroid_disorder',
  'iih',
  'epilepsy',
  'functional_neurological_disorder',
  'cancer',
  'bariatric_gastric_band',
  'bariatric_sleeve',
  'bariatric_bypass',
  'bariatric_balloon'
];

const CONDITION_TO_FIELD_MAP = CONDITION_MAP;

/**
 * Normalize condition name to standard format
 * @param {string} condition - Raw condition name
 * @returns {string} - Normalized condition name
 */
function normalizeConditionName(condition) {
  if (!condition || typeof condition !== 'string') return '';
  
  const normalized = condition.trim().toLowerCase();
  
  // Direct mapping
  if (CONDITION_TO_FIELD_MAP[normalized]) {
    return normalized;
  }
  
  // Try partial matches
  for (const [key, field] of Object.entries(CONDITION_TO_FIELD_MAP)) {
    if (normalized.includes(key) || key.includes(normalized)) {
      return normalized;
    }
  }
  
  return normalized;
}

/**
 * Map condition name to Patient boolean field
 * @param {string} condition - Condition name
 * @returns {string|null} - Patient field name or null if not mappable
 */
function mapConditionToField(condition) {
  if (!condition || typeof condition !== 'string') return null;
  
  const normalized = condition.trim().toLowerCase();
  
  // Direct lookup
  if (CONDITION_TO_FIELD_MAP[normalized]) {
    return CONDITION_TO_FIELD_MAP[normalized];
  }
  
  // Partial match lookup
  for (const [key, field] of Object.entries(CONDITION_TO_FIELD_MAP)) {
    if (normalized.includes(key) || key.includes(normalized)) {
      return field;
    }
  }
  
  return null;
}

/**
 * Map conditions array to Patient boolean fields (0/1)
 * @param {string[]} conditions - Array of condition names
 * @returns {Object} - Object with Patient field names as keys and 1 as values
 */
function mapConditionsToPatientFields(conditions) {
  const fieldMap = {};
  
  if (!Array.isArray(conditions)) return fieldMap;
  
  for (const condition of conditions) {
    const field = mapConditionToField(condition);
    if (field) {
      fieldMap[field] = 1; // Set to 1 (present)
    }
  }
  
  return fieldMap;
}

/**
 * Initialize all condition fields to 0
 * @param {Object} data - Patient data object
 * @returns {Object} - Data with all condition fields set to 0
 */
function initializeConditionFields(data = {}) {
  const initialized = { ...data };
  for (const field of CONDITION_FIELDS) {
    if (initialized[field] === null || initialized[field] === undefined) {
      initialized[field] = 0;
    }
  }
  return initialized;
}

/**
 * Map conditions array to set specific columns to 1
 * @param {Object} data - Patient data object (must have condition fields initialized to 0)
 * @param {string[]} conditions - Array of condition names
 * @returns {Object} - Data with mapped conditions set to 1
 */
function mapConditionsToColumns(data, conditions) {
  if (!Array.isArray(conditions)) return data;
  
  const mapped = { ...data };
  
  for (const condition of conditions) {
    if (!condition || typeof condition !== 'string') continue;
    
    // Normalize the condition name
    const normalized = condition.trim().toLowerCase();
    
    // Try direct lookup first
    let field = CONDITION_MAP[normalized];
    
    // If not found, try partial matching
    if (!field) {
      for (const [key, mappedField] of Object.entries(CONDITION_MAP)) {
        if (normalized.includes(key) || key.includes(normalized)) {
          field = mappedField;
          break;
        }
      }
    }
    
    // Set the field to 1 if found
    if (field && CONDITION_FIELDS.includes(field)) {
      mapped[field] = 1;
    }
  }
  
  return mapped;
}

module.exports = {
  normalizeConditionName,
  mapConditionToField,
  mapConditionsToPatientFields,
  initializeConditionFields,
  mapConditionsToColumns,
  CONDITION_MAP,
  CONDITION_FIELDS,
  CONDITION_TO_FIELD_MAP
};


/**
 * HbA1c Adjustment Calculator Service
 * Implements the MES (Medication Effect Score) calculation for HbA1c adjustment
 */

/**
 * Calculate adjusted HbA1c based on medication effect score
 * @param {number} measuredHbA1cPercent - Measured HbA1c in percentage
 * @param {number} weightKg - Patient weight in kilograms
 * @param {Object} medications - Object with medication names as keys and doses as values
 * @returns {Object} - Calculation results including MES, adjusted HbA1c values
 */
function calculateAdjustedHbA1c(measuredHbA1cPercent, weightKg, medications) {
  // Drug data: [maxDose, factor]
  const drugData = {
    insulin: [1.0, 2.5],
    metformin: [2550, 1.5],
    glimepiride: [8, 1.5],
    glipizide: [40, 1.5],
    glyburide: [20, 1.5],
    pioglitazone: [45, 0.95],
    sitagliptin: [100, 0.7],
    saxagliptin: [5, 0.7],
    linagliptin: [5, 0.7],
    liraglutide: [1.8, 1.15],
    exenatide_bid: [0.02, 0.7],
    exenatide_qw: [2, 1.1],
    dulaglutide: [1.5, 1.2],
    semaglutide: [1, 1.4],
    dapagliflozin: [10, 0.7],
    canagliflozin: [300, 0.9],
    empagliflozin: [25, 0.7],
  };

  let mes = 0;
  
  // Calculate MES for each medication
  for (const [drug, actualDose] of Object.entries(medications)) {
    if (actualDose <= 0) continue;
    if (!drugData[drug]) continue;
    
    const [maxDose, factor] = drugData[drug];
    mes += (actualDose / maxDose) * factor;
  }

  // Calculate adjusted HbA1c
  const adjustedHbA1cPercent = measuredHbA1cPercent + mes;
  const adjustedHbA1cMmolMol = (adjustedHbA1cPercent - 2.15) * 10.929;

  return {
    MES: Number(mes.toFixed(2)),
    adjustedHbA1cPercent: Number(adjustedHbA1cPercent.toFixed(2)),
    adjustedHbA1cMmolMol: Number(adjustedHbA1cMmolMol.toFixed(2)),
    measuredHbA1cPercent: Number(measuredHbA1cPercent.toFixed(2)),
    weightKg: Number(weightKg.toFixed(1)),
    medicationCount: Object.keys(medications).filter(drug => medications[drug] > 0).length
  };
}

/**
 * Get medication information for a specific drug
 * @param {string} drugName - Name of the medication
 * @returns {Object|null} - Drug information or null if not found
 */
function getMedicationInfo(drugName) {
  const drugData = {
    insulin: { maxDose: 1.0, factor: 2.5, unit: 'units/kg/day' },
    metformin: { maxDose: 2550, factor: 1.5, unit: 'mg/day' },
    glimepiride: { maxDose: 8, factor: 1.5, unit: 'mg/day' },
    glipizide: { maxDose: 40, factor: 1.5, unit: 'mg/day' },
    glyburide: { maxDose: 20, factor: 1.5, unit: 'mg/day' },
    pioglitazone: { maxDose: 45, factor: 0.95, unit: 'mg/day' },
    sitagliptin: { maxDose: 100, factor: 0.7, unit: 'mg/day' },
    saxagliptin: { maxDose: 5, factor: 0.7, unit: 'mg/day' },
    linagliptin: { maxDose: 5, factor: 0.7, unit: 'mg/day' },
    liraglutide: { maxDose: 1.8, factor: 1.15, unit: 'mg/day' },
    exenatide_bid: { maxDose: 0.02, factor: 0.7, unit: 'mg/day' },
    exenatide_qw: { maxDose: 2, factor: 1.1, unit: 'mg/week' },
    dulaglutide: { maxDose: 1.5, factor: 1.2, unit: 'mg/week' },
    semaglutide: { maxDose: 1, factor: 1.4, unit: 'mg/week' },
    dapagliflozin: { maxDose: 10, factor: 0.7, unit: 'mg/day' },
    canagliflozin: { maxDose: 300, factor: 0.9, unit: 'mg/day' },
    empagliflozin: { maxDose: 25, factor: 0.7, unit: 'mg/day' },
  };

  return drugData[drugName] || null;
}

/**
 * Get all available medications
 * @returns {Array} - Array of medication objects with display names
 */
function getAllMedications() {
  return [
    { key: 'insulin', name: 'Insulin', maxDose: 1.0, unit: 'units/kg/day' },
    { key: 'metformin', name: 'Metformin', maxDose: 2550, unit: 'mg/day' },
    { key: 'glimepiride', name: 'Glimepiride', maxDose: 8, unit: 'mg/day' },
    { key: 'glipizide', name: 'Glipizide', maxDose: 40, unit: 'mg/day' },
    { key: 'glyburide', name: 'Glyburide', maxDose: 20, unit: 'mg/day' },
    { key: 'pioglitazone', name: 'Pioglitazone', maxDose: 45, unit: 'mg/day' },
    { key: 'sitagliptin', name: 'Sitagliptin', maxDose: 100, unit: 'mg/day' },
    { key: 'saxagliptin', name: 'Saxagliptin', maxDose: 5, unit: 'mg/day' },
    { key: 'linagliptin', name: 'Linagliptin', maxDose: 5, unit: 'mg/day' },
    { key: 'liraglutide', name: 'Liraglutide', maxDose: 1.8, unit: 'mg/day' },
    { key: 'exenatide_bid', name: 'Exenatide (BID)', maxDose: 0.02, unit: 'mg/day' },
    { key: 'exenatide_qw', name: 'Exenatide (QW)', maxDose: 2, unit: 'mg/week' },
    { key: 'dulaglutide', name: 'Dulaglutide', maxDose: 1.5, unit: 'mg/week' },
    { key: 'semaglutide', name: 'Semaglutide', maxDose: 1, unit: 'mg/week' },
    { key: 'dapagliflozin', name: 'Dapagliflozin', maxDose: 10, unit: 'mg/day' },
    { key: 'canagliflozin', name: 'Canagliflozin', maxDose: 300, unit: 'mg/day' },
    { key: 'empagliflozin', name: 'Empagliflozin', maxDose: 25, unit: 'mg/day' }
  ];
}

module.exports = {
  calculateAdjustedHbA1c,
  getMedicationInfo,
  getAllMedications
};



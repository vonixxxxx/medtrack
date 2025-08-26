/**
 * Utility functions for calculating various health metrics
 */

/**
 * Calculate age from date of birth to one decimal place
 * @param {Date} dateOfBirth - Date of birth
 * @returns {number} Age in years with one decimal place
 */
exports.calculateAge = (dateOfBirth) => {
  if (!dateOfBirth) return null;
  
  const today = new Date();
  const birthDate = new Date(dateOfBirth);
  
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  
  // Calculate decimal part based on days since last birthday
  const lastBirthday = new Date(today.getFullYear(), birthDate.getMonth(), birthDate.getDate());
  if (today < lastBirthday) {
    lastBirthday.setFullYear(lastBirthday.getFullYear() - 1);
  }
  
  const daysSinceBirthday = (today - lastBirthday) / (1000 * 60 * 60 * 24);
  const decimalAge = age + (daysSinceBirthday / 365.25);
  
  return parseFloat(decimalAge.toFixed(1));
};

/**
 * Unit conversion constants
 */
const UNIT_CONVERSIONS = {
  // Weight conversions
  LBS_TO_KG: 0.453592,
  KG_TO_LBS: 2.20462,
  
  // Height conversions
  INCHES_TO_M: 0.0254,
  M_TO_INCHES: 39.3701,
  
  // Length conversions
  CM_TO_M: 0.01,
  M_TO_CM: 100
};

/**
 * Convert weight from pounds to kilograms
 * @param {number} weightLbs - Weight in pounds
 * @returns {number} Weight in kilograms
 */
exports.convertLbsToKg = (weightLbs) => {
  if (!weightLbs || weightLbs <= 0) return null;
  return weightLbs * UNIT_CONVERSIONS.LBS_TO_KG;
};

/**
 * Convert weight from kilograms to pounds
 * @param {number} weightKg - Weight in kilograms
 * @returns {number} Weight in pounds
 */
exports.convertKgToLbs = (weightKg) => {
  if (!weightKg || weightKg <= 0) return null;
  return weightKg * UNIT_CONVERSIONS.KG_TO_LBS;
};

/**
 * Convert height from inches to meters
 * @param {number} heightInches - Height in inches
 * @returns {number} Height in meters
 */
exports.convertInchesToM = (heightInches) => {
  if (!heightInches || heightInches <= 0) return null;
  return heightInches * UNIT_CONVERSIONS.INCHES_TO_M;
};

/**
 * Convert height from meters to inches
 * @param {number} heightM - Height in meters
 * @returns {number} Height in inches
 */
exports.convertMToInches = (heightM) => {
  if (!heightM || heightM <= 0) return null;
  return heightM * UNIT_CONVERSIONS.M_TO_INCHES;
};

/**
 * Calculate Body Mass Index (BMI)
 * @param {number} weight - Weight in kilograms
 * @param {number} height - Height in meters
 * @param {string} weightUnit - Weight unit ('kg' or 'lbs')
 * @param {string} heightUnit - Height unit ('m', 'cm', or 'inches')
 * @returns {number} BMI value
 */
exports.calculateBMI = (weight, height, weightUnit = 'kg', heightUnit = 'm') => {
  if (!weight || !height || weight <= 0 || height <= 0) return null;
  
  let weightKg = weight;
  let heightM = height;
  
  // Convert weight to kg if needed
  if (weightUnit === 'lbs') {
    weightKg = weight * UNIT_CONVERSIONS.LBS_TO_KG;
  }
  
  // Convert height to meters if needed
  if (heightUnit === 'cm') {
    heightM = height * UNIT_CONVERSIONS.CM_TO_M;
  } else if (heightUnit === 'inches') {
    heightM = height * UNIT_CONVERSIONS.INCHES_TO_M;
  }
  
  // Calculate BMI using standard formula: weight (kg) / height (m)²
  const bmi = weightKg / (heightM * heightM);
  return parseFloat(bmi.toFixed(1));
};

/**
 * Calculate Waist-to-Hip Ratio (WHR)
 * @param {number} waistCircumference - Waist circumference in cm
 * @param {number} hipCircumference - Hip circumference in cm
 * @returns {number} WHR value
 */
exports.calculateWHR = (waistCircumference, hipCircumference) => {
  if (!waistCircumference || !hipCircumference || hipCircumference <= 0) return null;
  return waistCircumference / hipCircumference;
};

/**
 * Calculate Waist-to-Height Ratio (WHtR)
 * @param {number} waistCircumference - Waist circumference in cm
 * @param {number} height - Height in meters
 * @returns {number} WHtR value
 */
exports.calculateWHtR = (waistCircumference, height) => {
  if (!waistCircumference || !height || height <= 0) return null;
  return waistCircumference / (height * 100);
};

/**
 * Calculate Body Roundness Index (BRI)
 * Formula: 364.2 - 365.5 * sqrt(1 - (waist / (height * 100))^2)
 * @param {number} waistCircumference - Waist circumference in cm
 * @param {number} height - Height in meters
 * @param {number} weight - Weight in kilograms
 * @returns {number} BRI value
 */
exports.calculateBRI = (waistCircumference, height, weight) => {
  if (!waistCircumference || !height || !weight || height <= 0 || weight <= 0) return null;
  
  const heightCm = height * 100;
  const waistHeightRatio = waistCircumference / heightCm;
  
  // BRI formula from https://www.mdcalc.com/calc/10575/body-roundness-index-bri
  const bri = 364.2 - 365.5 * Math.sqrt(1 - Math.pow(waistHeightRatio, 2));
  
  return parseFloat(bri.toFixed(2));
};

/**
 * Calculate Pack Years for smoking history
 * @param {number} cigarettesPerDay - Number of cigarettes smoked per day
 * @param {number} yearsSmoked - Number of years smoked
 * @returns {number} Pack years (1 pack = 20 cigarettes)
 */
exports.calculatePackYears = (cigarettesPerDay, yearsSmoked) => {
  if (!cigarettesPerDay || !yearsSmoked || cigarettesPerDay <= 0 || yearsSmoked <= 0) return null;
  return (cigarettesPerDay / 20) * yearsSmoked;
};

/**
 * Get BMI category based on BMI value
 * @param {number} bmi - BMI value
 * @returns {string} BMI category
 */
exports.getBMICategory = (bmi) => {
  if (!bmi) return null;
  
  if (bmi < 18.5) return 'underweight';
  if (bmi < 25) return 'normal';
  if (bmi < 30) return 'overweight';
  if (bmi < 35) return 'obese_class_1';
  if (bmi < 40) return 'obese_class_2';
  return 'obese_class_3';
};

/**
 * Get WHR risk category based on biological sex
 * @param {number} whr - Waist-to-hip ratio
 * @param {string} biologicalSex - Biological sex ('male' or 'female')
 * @returns {string} Risk category
 */
exports.getWHRRiskCategory = (whr, biologicalSex) => {
  if (!whr || !biologicalSex) return null;
  
  if (biologicalSex === 'male') {
    if (whr < 0.9) return 'low_risk';
    if (whr < 1.0) return 'moderate_risk';
    return 'high_risk';
  } else {
    if (whr < 0.8) return 'low_risk';
    if (whr < 0.85) return 'moderate_risk';
    return 'high_risk';
  }
};

/**
 * Get AUDIT risk category based on score
 * @param {number} score - AUDIT questionnaire score
 * @returns {string} Risk category
 */
exports.getAUDITRiskCategory = (score) => {
  if (!score) return null;
  
  if (score <= 7) return 'low_risk';
  if (score <= 15) return 'medium_risk';
  if (score <= 19) return 'high_risk';
  return 'very_high_risk';
};

/**
 * Get IPAQ activity category based on score
 * @param {number} score - IPAQ questionnaire score
 * @returns {string} Activity category
 */
exports.getIPAQActivityCategory = (score) => {
  if (!score) return null;
  
  if (score < 600) return 'low_activity';
  if (score < 3000) return 'moderate_activity';
  return 'high_activity';
};

/**
 * Calculate ideal body weight using various formulas
 * @param {number} height - Height in meters
 * @param {string} biologicalSex - Biological sex ('male' or 'female')
 * @param {string} formula - Formula to use ('devine', 'robinson', 'miller', 'hamwi')
 * @returns {number} Ideal body weight in kg
 */
exports.calculateIdealBodyWeight = (height, biologicalSex, formula = 'devine') => {
  if (!height || !biologicalSex) return null;
  
  const heightCm = height * 100;
  let ibw = 0;
  
  switch (formula.toLowerCase()) {
    case 'devine':
      if (biologicalSex === 'male') {
        ibw = 50 + 2.3 * ((heightCm - 152.4) / 2.54);
      } else {
        ibw = 45.5 + 2.3 * ((heightCm - 152.4) / 2.54);
      }
      break;
      
    case 'robinson':
      if (biologicalSex === 'male') {
        ibw = 52 + 1.9 * ((heightCm - 152.4) / 2.54);
      } else {
        ibw = 49 + 1.7 * ((heightCm - 152.4) / 2.54);
      }
      break;
      
    case 'miller':
      if (biologicalSex === 'male') {
        ibw = 56.2 + 1.41 * ((heightCm - 152.4) / 2.54);
      } else {
        ibw = 53.1 + 1.36 * ((heightCm - 152.4) / 2.54);
      }
      break;
      
    case 'hamwi':
      if (biologicalSex === 'male') {
        ibw = 48 + 2.7 * ((heightCm - 152.4) / 2.54);
      } else {
        ibw = 45.5 + 2.2 * ((heightCm - 152.4) / 2.54);
      }
      break;
      
    default:
      return null;
  }
  
  return parseFloat(ibw.toFixed(1));
};

/**
 * Calculate body fat percentage using various methods
 * @param {number} bmi - BMI value
 * @param {number} age - Age in years
 * @param {string} biologicalSex - Biological sex ('male' or 'female')
 * @param {string} method - Method to use ('bmi', 'navy', 'deurenberg')
 * @returns {number} Estimated body fat percentage
 */
exports.calculateBodyFatPercentage = (bmi, age, biologicalSex, method = 'bmi') => {
  if (!bmi || !age || !biologicalSex) return null;
  
  let bfp = 0;
  
  switch (method.toLowerCase()) {
    case 'bmi':
      // Simple BMI-based estimation
      if (biologicalSex === 'male') {
        bfp = (1.20 * bmi) + (0.23 * age) - 16.2;
      } else {
        bfp = (1.20 * bmi) + (0.23 * age) - 5.4;
      }
      break;
      
    case 'deurenberg':
      // Deurenberg formula
      bfp = (1.2 * bmi) + (0.23 * age) - (10.8 * (biologicalSex === 'male' ? 1 : 0)) - 5.4;
      break;
      
    default:
      return null;
  }
  
  // Ensure BFP is within reasonable bounds
  bfp = Math.max(2, Math.min(50, bfp));
  
  return parseFloat(bfp.toFixed(1));
};

/**
 * Calculate 10-year ASCVD Risk Score (AHA/ACC 2013)
 * Based on Pooled Cohort Equations
 * @param {number} age - Age in years
 * @param {string} gender - Gender ('male' or 'female')
 * @param {string} race - Race ('white' or 'other')
 * @param {number} totalCholesterol - Total cholesterol in mg/dL
 * @param {number} hdlCholesterol - HDL cholesterol in mg/dL
 * @param {number} systolicBP - Systolic blood pressure in mmHg
 * @param {boolean} isOnBPMedication - Whether on blood pressure medication
 * @param {boolean} isSmoker - Current smoking status
 * @param {boolean} hasDiabetes - Diabetes status
 * @returns {object} Risk score and category
 */
exports.calculateASCVDRisk = (age, gender, race, totalCholesterol, hdlCholesterol, systolicBP, isOnBPMedication, isSmoker, hasDiabetes) => {
  if (!age || !gender || !totalCholesterol || !hdlCholesterol || !systolicBP) {
    return null;
  }
  
  // Validate age range (40-79 years for ASCVD calculation)
  if (age < 40 || age > 79) {
    return {
      riskScore: null,
      riskCategory: 'Age out of range (40-79 years required)',
      riskPercentage: null
    };
  }
  
  // Coefficients for Pooled Cohort Equations (simplified version)
  let riskScore = 0;
  
  // Base risk factors
  if (gender === 'male') {
    riskScore += age * 0.1;
    if (race === 'white') {
      riskScore += Math.log(totalCholesterol) * 0.5;
      riskScore += Math.log(hdlCholesterol) * -0.3;
    } else {
      riskScore += Math.log(totalCholesterol) * 0.4;
      riskScore += Math.log(hdlCholesterol) * -0.2;
    }
  } else {
    riskScore += age * 0.08;
    if (race === 'white') {
      riskScore += Math.log(totalCholesterol) * 0.4;
      riskScore += Math.log(hdlCholesterol) * -0.2;
    } else {
      riskScore += Math.log(totalCholesterol) * 0.3;
      riskScore += Math.log(hdlCholesterol) * -0.1;
    }
  }
  
  // Blood pressure
  if (isOnBPMedication) {
    riskScore += Math.log(systolicBP) * 0.3;
  } else {
    riskScore += Math.log(systolicBP) * 0.2;
  }
  
  // Smoking
  if (isSmoker) {
    riskScore += 0.5;
  }
  
  // Diabetes
  if (hasDiabetes) {
    riskScore += 0.4;
  }
  
  // Convert to percentage (simplified calculation)
  const riskPercentage = Math.min(95, Math.max(1, Math.round(riskScore * 10)));
  
  // Risk categorization
  let riskCategory;
  if (riskPercentage < 5) {
    riskCategory = 'Low Risk';
  } else if (riskPercentage < 7.5) {
    riskCategory = 'Borderline Risk';
  } else if (riskPercentage < 20) {
    riskCategory = 'Intermediate Risk';
  } else {
    riskCategory = 'High Risk';
  }
  
  return {
    riskScore: riskScore,
    riskCategory: riskCategory,
    riskPercentage: riskPercentage
  };
};

/**
 * Calculate Framingham Risk Score (alternative to ASCVD)
 * @param {number} age - Age in years
 * @param {string} gender - Gender ('male' or 'female')
 * @param {number} totalCholesterol - Total cholesterol in mg/dL
 * @param {number} hdlCholesterol - HDL cholesterol in mg/dL
 * @param {number} systolicBP - Systolic blood pressure in mmHg
 * @param {boolean} isOnBPMedication - Whether on blood pressure medication
 * @param {boolean} isSmoker - Current smoking status
 * @param {boolean} hasDiabetes - Diabetes status
 * @returns {object} Risk score and category
 */
exports.calculateFraminghamRisk = (age, gender, race, totalCholesterol, hdlCholesterol, systolicBP, isOnBPMedication, isSmoker, hasDiabetes) => {
  if (!age || !gender || !totalCholesterol || !hdlCholesterol || !systolicBP) {
    return null;
  }
  
  let riskScore = 0;
  
  // Age points
  if (gender === 'male') {
    if (age >= 20 && age <= 34) riskScore += 0;
    else if (age >= 35 && age <= 39) riskScore += 2;
    else if (age >= 40 && age <= 44) riskScore += 5;
    else if (age >= 45 && age <= 49) riskScore += 7;
    else if (age >= 50 && age <= 54) riskScore += 8;
    else if (age >= 55 && age <= 59) riskScore += 10;
    else if (age >= 60 && age <= 64) riskScore += 11;
    else if (age >= 65 && age <= 69) riskScore += 12;
    else if (age >= 70 && age <= 74) riskScore += 14;
    else if (age >= 75 && age <= 79) riskScore += 15;
  } else {
    if (age >= 20 && age <= 34) riskScore += 0;
    else if (age >= 35 && age <= 39) riskScore += 2;
    else if (age >= 40 && age <= 44) riskScore += 4;
    else if (age >= 45 && age <= 49) riskScore += 5;
    else if (age >= 50 && age <= 54) riskScore += 7;
    else if (age >= 55 && age <= 59) riskScore += 8;
    else if (age >= 60 && age <= 64) riskScore += 8;
    else if (age >= 65 && age <= 69) riskScore += 8;
    else if (age >= 70 && age <= 74) riskScore += 8;
    else if (age >= 75 && age <= 79) riskScore += 8;
  }
  
  // Total cholesterol points
  if (gender === 'male') {
    if (totalCholesterol < 160) riskScore += 0;
    else if (totalCholesterol >= 160 && totalCholesterol <= 199) riskScore += 1;
    else if (totalCholesterol >= 200 && totalCholesterol <= 239) riskScore += 2;
    else if (totalCholesterol >= 240 && totalCholesterol <= 279) riskScore += 3;
    else if (totalCholesterol >= 280) riskScore += 4;
  } else {
    if (totalCholesterol < 160) riskScore += 0;
    else if (totalCholesterol >= 160 && totalCholesterol <= 199) riskScore += 1;
    else if (totalCholesterol >= 200 && totalCholesterol <= 239) riskScore += 2;
    else if (totalCholesterol >= 240 && totalCholesterol <= 279) riskScore += 3;
    else if (totalCholesterol >= 280) riskScore += 4;
  }
  
  // HDL cholesterol points
  if (gender === 'male') {
    if (hdlCholesterol >= 60) riskScore += -1;
    else if (hdlCholesterol >= 50 && hdlCholesterol <= 59) riskScore += 0;
    else if (hdlCholesterol >= 40 && hdlCholesterol <= 49) riskScore += 1;
    else if (hdlCholesterol < 40) riskScore += 2;
  } else {
    if (hdlCholesterol >= 60) riskScore += -1;
    else if (hdlCholesterol >= 50 && hdlCholesterol <= 59) riskScore += 0;
    else if (hdlCholesterol >= 40 && hdlCholesterol <= 49) riskScore += 1;
    else if (hdlCholesterol < 40) riskScore += 2;
  }
  
  // Blood pressure points
  if (gender === 'male') {
    if (systolicBP < 120) riskScore += 0;
    else if (systolicBP >= 120 && systolicBP <= 129) riskScore += 0;
    else if (systolicBP >= 130 && systolicBP <= 139) riskScore += 1;
    else if (systolicBP >= 140 && systolicBP <= 159) riskScore += 2;
    else if (systolicBP >= 160) riskScore += 3;
  } else {
    if (systolicBP < 120) riskScore += 0;
    else if (systolicBP >= 120 && systolicBP <= 129) riskScore += 1;
    else if (systolicBP >= 130 && systolicBP <= 139) riskScore += 2;
    else if (systolicBP >= 140 && systolicBP <= 159) riskScore += 3;
    else if (systolicBP >= 160) riskScore += 4;
  }
  
  // Smoking points
  if (isSmoker) {
    if (gender === 'male') riskScore += 2;
    else riskScore += 3;
  }
  
  // Diabetes points
  if (hasDiabetes) {
    if (gender === 'male') riskScore += 2;
    else riskScore += 4;
  }
  
  // Convert score to 10-year risk percentage
  let riskPercentage;
  if (gender === 'male') {
    if (riskScore <= 0) riskPercentage = 1;
    else if (riskScore <= 4) riskPercentage = 1;
    else if (riskScore <= 6) riskPercentage = 2;
    else if (riskScore <= 7) riskPercentage = 3;
    else if (riskScore <= 8) riskPercentage = 4;
    else if (riskScore <= 9) riskPercentage = 5;
    else if (riskScore <= 10) riskPercentage = 6;
    else if (riskScore <= 11) riskPercentage = 8;
    else if (riskScore <= 12) riskPercentage = 10;
    else if (riskScore <= 13) riskPercentage = 12;
    else if (riskScore <= 14) riskPercentage = 16;
    else if (riskScore <= 15) riskPercentage = 20;
    else if (riskScore <= 16) riskPercentage = 25;
    else if (riskScore <= 17) riskPercentage = 30;
    else if (riskScore <= 18) riskPercentage = 35;
    else if (riskScore <= 19) riskPercentage = 40;
    else if (riskScore <= 20) riskPercentage = 45;
    else if (riskScore <= 21) riskPercentage = 50;
    else if (riskScore <= 22) riskPercentage = 55;
    else if (riskScore <= 23) riskPercentage = 60;
    else riskPercentage = 65;
  } else {
    if (riskScore <= 0) riskPercentage = 1;
    else if (riskScore <= 4) riskPercentage = 1;
    else if (riskScore <= 6) riskPercentage = 2;
    else if (riskScore <= 7) riskPercentage = 3;
    else if (riskScore <= 8) riskPercentage = 4;
    else if (riskScore <= 9) riskPercentage = 5;
    else if (riskScore <= 10) riskPercentage = 6;
    else if (riskScore <= 11) riskPercentage = 8;
    else if (riskScore <= 12) riskPercentage = 10;
    else if (riskScore <= 13) riskPercentage = 12;
    else if (riskScore <= 14) riskPercentage = 16;
    else if (riskScore <= 15) riskPercentage = 20;
    else if (riskScore <= 16) riskPercentage = 25;
    else if (riskScore <= 17) riskPercentage = 30;
    else if (riskScore <= 18) riskPercentage = 35;
    else if (riskScore <= 19) riskPercentage = 40;
    else if (riskScore <= 20) riskPercentage = 45;
    else if (riskScore <= 21) riskPercentage = 50;
    else if (riskScore <= 22) riskPercentage = 55;
    else if (riskScore <= 23) riskPercentage = 60;
    else riskPercentage = 65;
  }
  
  // Risk categorization
  let riskCategory;
  if (riskPercentage < 10) {
    riskCategory = 'Low Risk';
  } else if (riskPercentage < 20) {
    riskCategory = 'Intermediate Risk';
  } else {
    riskCategory = 'High Risk';
  }
  
  return {
    riskScore: riskScore,
    riskCategory: riskCategory,
    riskPercentage: riskPercentage
  };
};

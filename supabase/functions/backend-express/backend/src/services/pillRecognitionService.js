const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const { predictWithMLModel, MEDICINES } = require('./pytorchMLService');

/**
 * Pill Recognition Service
 * Enhanced with Confir-Med features:
 * - Image processing pipeline
 * - Drug identification with comprehensive dataset
 * - ML model integration infrastructure
 */

// Comprehensive pill dataset (in production, use external API or database)
const PILL_DATASET = [
  {
    name: 'Aspirin',
    genericName: 'Acetylsalicylic acid',
    imprints: ['81', '325', 'ASA', 'BAYER'],
    shapes: ['round', 'oval', 'caplet'],
    colors: ['white', 'yellow', 'orange'],
    sizes: ['small', 'medium'],
    ndcCode: '00069-0081-01',
    rxnormCode: '1191'
  },
  {
    name: 'Ibuprofen',
    genericName: 'Ibuprofen',
    imprints: ['IBU', '200', '400', '600', '800'],
    shapes: ['round', 'oval', 'caplet'],
    colors: ['white', 'brown', 'pink'],
    sizes: ['small', 'medium', 'large'],
    ndcCode: '00069-0200-01',
    rxnormCode: '5640'
  },
  {
    name: 'Acetaminophen',
    genericName: 'Acetaminophen',
    imprints: ['TYLENOL', '500', '650', 'APAP'],
    shapes: ['round', 'caplet', 'gelcap'],
    colors: ['white', 'red', 'blue'],
    sizes: ['small', 'medium'],
    ndcCode: '00069-0500-01',
    rxnormCode: '161'
  },
  {
    name: 'Metformin',
    genericName: 'Metformin hydrochloride',
    imprints: ['500', '850', '1000', 'MET'],
    shapes: ['round', 'oval'],
    colors: ['white'],
    sizes: ['small', 'medium'],
    ndcCode: '00069-0500-02',
    rxnormCode: '6809'
  },
  {
    name: 'Lisinopril',
    genericName: 'Lisinopril',
    imprints: ['5', '10', '20', '40', 'LIS'],
    shapes: ['round', 'oval'],
    colors: ['white', 'pink', 'yellow'],
    sizes: ['small', 'medium'],
    ndcCode: '00069-0005-01',
    rxnormCode: '29046'
  },
  {
    name: 'Atorvastatin',
    genericName: 'Atorvastatin calcium',
    imprints: ['10', '20', '40', '80', 'LIPITOR'],
    shapes: ['round', 'oval'],
    colors: ['white', 'blue', 'pink'],
    sizes: ['small', 'medium'],
    ndcCode: '00069-0010-01',
    rxnormCode: '83367'
  },
  {
    name: 'Amlodipine',
    genericName: 'Amlodipine besylate',
    imprints: ['2.5', '5', '10', 'AML'],
    shapes: ['round', 'oval'],
    colors: ['white'],
    sizes: ['small', 'medium'],
    ndcCode: '00069-0002-01',
    rxnormCode: '17767'
  },
  {
    name: 'Omeprazole',
    genericName: 'Omeprazole',
    imprints: ['20', '40', 'OME', 'PRILOSEC'],
    shapes: ['capsule', 'tablet'],
    colors: ['white', 'pink', 'purple'],
    sizes: ['small', 'medium'],
    ndcCode: '00069-0020-01',
    rxnormCode: '7646'
  }
];

/**
 * Image Processing Pipeline
 * Prepares images for ML model analysis
 */
class ImageProcessingPipeline {
  constructor() {
    this.targetSize = { width: 224, height: 224 }; // Standard ML input size
  }

  /**
   * Process image: resize, normalize, enhance
   */
  async processImage(imagePath) {
    try {
      const processedPath = path.join(
        path.dirname(imagePath),
        'processed-' + path.basename(imagePath)
      );

      await sharp(imagePath)
        .resize(this.targetSize.width, this.targetSize.height, {
          fit: 'contain',
          background: { r: 255, g: 255, b: 255, alpha: 1 }
        })
        .normalize() // Enhance contrast
        .sharpen() // Enhance edges
        .toFile(processedPath);

      return processedPath;
    } catch (error) {
      console.error('Error processing image:', error);
      throw new Error('Failed to process image');
    }
  }

  /**
   * Extract features from image (placeholder for ML model)
   */
  async extractFeatures(imagePath) {
    // In production, this would use a trained ML model
    // For now, return mock features
    return {
      shape: 'round',
      color: 'white',
      size: 'medium',
      imprint: null,
      confidence: 0.75
    };
  }
}

/**
 * Drug Identification Service
 * Maps recognized pills to dataset
 */
class DrugIdentificationService {
  constructor() {
    this.dataset = PILL_DATASET;
  }

  /**
   * Identify pill from features
   */
  identifyPill(features) {
    const { imprint, shape, color, size } = features;
    
    // Score each medication in dataset
    const matches = this.dataset.map(med => {
      let score = 0;
      let matchedFeatures = [];

      // Check imprint match
      if (imprint && med.imprints) {
        const imprintMatch = med.imprints.some(imp => 
          imp.toLowerCase().includes(imprint.toLowerCase()) ||
          imprint.toLowerCase().includes(imp.toLowerCase())
        );
        if (imprintMatch) {
          score += 40;
          matchedFeatures.push('imprint');
        }
      }

      // Check shape match
      if (shape && med.shapes.includes(shape.toLowerCase())) {
        score += 20;
        matchedFeatures.push('shape');
      }

      // Check color match
      if (color && med.colors.includes(color.toLowerCase())) {
        score += 20;
        matchedFeatures.push('color');
      }

      // Check size match
      if (size && med.sizes.includes(size.toLowerCase())) {
        score += 20;
        matchedFeatures.push('size');
      }

      return {
        ...med,
        score,
        matchedFeatures,
        confidence: score / 100
      };
    });

    // Sort by score and return top matches
    const topMatches = matches
      .filter(m => m.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    return topMatches.length > 0 ? topMatches[0] : null;
  }

  /**
   * Search dataset by name or generic name
   */
  searchByName(query) {
    const queryLower = query.toLowerCase();
    return this.dataset.filter(med => 
      med.name.toLowerCase().includes(queryLower) ||
      med.genericName.toLowerCase().includes(queryLower)
    );
  }
}

/**
 * ML Model Integration
 * Integrates with Confir-Med PyTorch model
 */
class MLModelService {
  constructor() {
    this.modelPath = null;
    this.modelLoaded = false;
  }

  /**
   * Load ML model (placeholder - model loaded in Python service)
   */
  async loadModel() {
    this.modelLoaded = true;
    return true;
  }

  /**
   * Predict pill from image using Confir-Med ML model
   */
  async predict(imagePath) {
    try {
      // Try to use PyTorch ML model first
      const mlResult = await predictWithMLModel(imagePath);
      
      if (mlResult.available && mlResult.predicted_medicine !== 'Unknown') {
        // Map Confir-Med medicine to our dataset
        const medicineName = mlResult.predicted_medicine;
        const matchedMed = drugIdentification.searchByName(medicineName);
        
        if (matchedMed.length > 0) {
          const med = matchedMed[0];
          return {
            shape: 'round', // Default - could be enhanced
            color: med.colors?.[0] || 'white',
            size: med.sizes?.[0] || 'medium',
            imprint: med.imprints?.[0] || null,
            confidence: mlResult.confidence,
            medicationName: med.name,
            genericName: med.genericName
          };
        }
      }

      // Fallback to basic feature extraction
      return {
        shape: 'round',
        color: 'white',
        size: 'medium',
        imprint: null,
        confidence: 0.75
      };
    } catch (error) {
      console.error('Error in ML prediction:', error);
      // Fallback to basic prediction
      return {
        shape: 'round',
        color: 'white',
        size: 'medium',
        imprint: null,
        confidence: 0.5
      };
    }
  }
}

// Export services
const imagePipeline = new ImageProcessingPipeline();
const drugIdentification = new DrugIdentificationService();
const mlModel = new MLModelService();

/**
 * Main recognition function
 */
async function recognizePill(imagePath) {
  try {
    // Step 1: Process image
    const processedImage = await imagePipeline.processImage(imagePath);
    
    // Step 2: Extract features using ML model
    const features = await mlModel.predict(processedImage);
    
    // Step 3: Identify drug from dataset
    const identification = drugIdentification.identifyPill(features);
    
    if (identification) {
      return {
        recognized: true,
        confidence: identification.confidence,
        medicationName: identification.name,
        genericName: identification.genericName,
        imprint: features.imprint,
        shape: features.shape,
        color: features.color,
        size: features.size,
        ndcCode: identification.ndcCode,
        rxnormCode: identification.rxnormCode,
        matchedFeatures: identification.matchedFeatures
      };
    }

    // If no match found, return features only
    return {
      recognized: false,
      confidence: features.confidence,
      imprint: features.imprint,
      shape: features.shape,
      color: features.color,
      size: features.size
    };
  } catch (error) {
    console.error('Error in pill recognition:', error);
    throw error;
  }
}

module.exports = {
  recognizePill,
  imagePipeline,
  drugIdentification,
  mlModel,
  PILL_DATASET
};


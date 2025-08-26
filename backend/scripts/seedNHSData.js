const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

// Comprehensive NHS Medicines A-Z dataset
const nhsMedications = [
  // Pain Management & Anti-inflammatory
  {
    genericName: 'Paracetamol',
    atcClass: 'N02BE01',
    synonyms: ['acetaminophen', 'APAP', 'Calpol', 'Tylenol'],
    products: [
      {
        brandName: 'Generic Paracetamol',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily', 'three times daily', 'four times daily', 'every 4-6 hours', 'when needed'],
        strengths: [
          { value: 500, unit: 'mg', frequency: 'every 4-6 hours' },
          { value: 1000, unit: 'mg', frequency: 'every 4-6 hours' }
        ]
      },
      {
        brandName: 'Calpol',
        route: 'oral',
        form: 'suspension',
        allowedIntakeType: 'Liquid',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily', 'three times daily', 'four times daily', 'every 4-6 hours', 'when needed'],
        strengths: [
          { value: 120, unit: 'mg/5ml', frequency: 'every 4-6 hours' },
          { value: 250, unit: 'mg/5ml', frequency: 'every 4-6 hours' }
        ]
      }
    ]
  },
  {
    genericName: 'Ibuprofen',
    atcClass: 'M01AE01',
    synonyms: ['IBU', 'NSAID', 'Brufen', 'Nurofen'],
    products: [
      {
        brandName: 'Generic Ibuprofen',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily', 'three times daily', 'four times daily', 'every 6-8 hours', 'when needed'],
        strengths: [
          { value: 200, unit: 'mg', frequency: 'every 6-8 hours' },
          { value: 400, unit: 'mg', frequency: 'every 6-8 hours' },
          { value: 600, unit: 'mg', frequency: 'every 6-8 hours' }
        ]
      },
      {
        brandName: 'Nurofen',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily', 'three times daily', 'four times daily', 'every 6-8 hours', 'when needed'],
        strengths: [
          { value: 200, unit: 'mg', frequency: 'every 6-8 hours' },
          { value: 400, unit: 'mg', frequency: 'every 6-8 hours' }
        ]
      }
    ]
  },
  {
    genericName: 'Aspirin',
    atcClass: 'B01AC06',
    synonyms: ['ASA', 'NSAID', 'Disprin', 'Bayer'],
    products: [
      {
        brandName: 'Low-dose Aspirin',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'once daily'],
        strengths: [
          { value: 75, unit: 'mg', frequency: 'daily' },
          { value: 81, unit: 'mg', frequency: 'daily' },
          { value: 100, unit: 'mg', frequency: 'daily' }
        ]
      },
      {
        brandName: 'Pain Relief Aspirin',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['when needed', 'every 4-6 hours'],
        strengths: [
          { value: 300, unit: 'mg', frequency: 'when needed' },
          { value: 500, unit: 'mg', frequency: 'when needed' }
        ]
      }
    ]
  },

  // Diabetes Medications
  {
    genericName: 'Metformin',
    atcClass: 'A10BA02',
    synonyms: ['MET', 'Glucophage', 'Fortamet'],
    products: [
      {
        brandName: 'Generic Metformin',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily', 'three times daily'],
        strengths: [
          { value: 500, unit: 'mg', frequency: 'twice daily' },
          { value: 850, unit: 'mg', frequency: 'twice daily' },
          { value: 1000, unit: 'mg', frequency: 'twice daily' }
        ]
      },
      {
        brandName: 'Metformin SR',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily'],
        strengths: [
          { value: 500, unit: 'mg', frequency: 'daily' },
          { value: 750, unit: 'mg', frequency: 'daily' },
          { value: 1000, unit: 'mg', frequency: 'daily' }
        ]
      }
    ]
  },
  {
    genericName: 'Semaglutide',
    atcClass: 'A10BJ06',
    synonyms: ['GLP-1', 'GLP1', 'Ozempic', 'Rybelsus'],
    products: [
      {
        brandName: 'Ozempic',
        route: 'subcutaneous',
        form: 'injection-pen',
        allowedIntakeType: 'Injection',
        defaultPlaces: ['at home', 'self administered', 'at clinic'],
        allowedFrequencies: ['weekly'],
        strengths: [
          { value: 0.25, unit: 'mg', frequency: 'weekly' },
          { value: 0.5, unit: 'mg', frequency: 'weekly' },
          { value: 1, unit: 'mg', frequency: 'weekly' },
          { value: 2, unit: 'mg', frequency: 'weekly' }
        ]
      },
      {
        brandName: 'Rybelsus',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily'],
        strengths: [
          { value: 3, unit: 'mg', frequency: 'daily' },
          { value: 7, unit: 'mg', frequency: 'daily' },
          { value: 14, unit: 'mg', frequency: 'daily' }
        ]
      }
    ]
  },
  {
    genericName: 'Insulin Glargine',
    atcClass: 'A10AB05',
    synonyms: ['INS', 'Lantus', 'Basaglar'],
    products: [
      {
        brandName: 'Lantus',
        route: 'subcutaneous',
        form: 'injection-pen',
        allowedIntakeType: 'Injection',
        defaultPlaces: ['at home', 'self administered', 'at clinic'],
        allowedFrequencies: ['daily', 'twice daily'],
        strengths: [
          { value: 100, unit: 'units/ml', frequency: 'daily' }
        ]
      }
    ]
  },

  // Cardiovascular Medications
  {
    genericName: 'Atorvastatin',
    atcClass: 'C10AA05',
    synonyms: ['ATV', 'statin', 'Lipitor'],
    products: [
      {
        brandName: 'Generic Atorvastatin',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily'],
        strengths: [
          { value: 10, unit: 'mg', frequency: 'daily' },
          { value: 20, unit: 'mg', frequency: 'daily' },
          { value: 40, unit: 'mg', frequency: 'daily' },
          { value: 80, unit: 'mg', frequency: 'daily' }
        ]
      }
    ]
  },
  {
    genericName: 'Amlodipine',
    atcClass: 'C08CA01',
    synonyms: ['AML', 'CCB', 'Norvasc'],
    products: [
      {
        brandName: 'Generic Amlodipine',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily'],
        strengths: [
          { value: 2.5, unit: 'mg', frequency: 'daily' },
          { value: 5, unit: 'mg', frequency: 'daily' },
          { value: 10, unit: 'mg', frequency: 'daily' }
        ]
      }
    ]
  },
  {
    genericName: 'Lisinopril',
    atcClass: 'C09AA03',
    synonyms: ['LIS', 'ACE inhibitor', 'Zestril'],
    products: [
      {
        brandName: 'Generic Lisinopril',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily'],
        strengths: [
          { value: 2.5, unit: 'mg', frequency: 'daily' },
          { value: 5, unit: 'mg', frequency: 'daily' },
          { value: 10, unit: 'mg', frequency: 'daily' },
          { value: 20, unit: 'mg', frequency: 'daily' },
          { value: 40, unit: 'mg', frequency: 'daily' }
        ]
      }
    ]
  },

  // Respiratory Medications
  {
    genericName: 'Salbutamol',
    atcClass: 'R03AC02',
    synonyms: ['SAL', 'SABA', 'Ventolin', 'ProAir'],
    products: [
      {
        brandName: 'Ventolin',
        route: 'inhalation',
        form: 'inhaler',
        allowedIntakeType: 'Inhaler',
        defaultPlaces: ['at home', 'self administered', 'at clinic'],
        allowedFrequencies: ['when needed', 'every 4-6 hours', 'before exercise'],
        strengths: [
          { value: 100, unit: 'mcg/dose', frequency: 'when needed' },
          { value: 200, unit: 'mcg/dose', frequency: 'when needed' }
        ]
      }
    ]
  },
  {
    genericName: 'Fluticasone',
    atcClass: 'R03BA01',
    synonyms: ['FLU', 'corticosteroid', 'Flovent', 'Flixotide'],
    products: [
      {
        brandName: 'Flovent',
        route: 'inhalation',
        form: 'inhaler',
        allowedIntakeType: 'Inhaler',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily', 'twice daily'],
        strengths: [
          { value: 50, unit: 'mcg/dose', frequency: 'twice daily' },
          { value: 125, unit: 'mcg/dose', frequency: 'twice daily' },
          { value: 250, unit: 'mcg/dose', frequency: 'twice daily' }
        ]
      }
    ]
  },

  // Gastrointestinal Medications
  {
    genericName: 'Omeprazole',
    atcClass: 'A02BC01',
    synonyms: ['OME', 'PPI', 'Prilosec', 'Losec'],
    products: [
      {
        brandName: 'Generic Omeprazole',
        route: 'oral',
        form: 'capsule',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily'],
        strengths: [
          { value: 10, unit: 'mg', frequency: 'daily' },
          { value: 20, unit: 'mg', frequency: 'daily' },
          { value: 40, unit: 'mg', frequency: 'daily' }
        ]
      }
    ]
  },

  // Antibiotics
  {
    genericName: 'Amoxicillin',
    atcClass: 'J01CA04',
    synonyms: ['AMOX', 'penicillin', 'Amoxil'],
    products: [
      {
        brandName: 'Generic Amoxicillin',
        route: 'oral',
        form: 'capsule',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['twice daily', 'three times daily'],
        strengths: [
          { value: 250, unit: 'mg', frequency: 'three times daily' },
          { value: 500, unit: 'mg', frequency: 'three times daily' }
        ]
      }
    ]
  },

  // Anticoagulants
  {
    genericName: 'Warfarin',
    atcClass: 'B01AA03',
    synonyms: ['WAR', 'anticoagulant', 'Coumadin'],
    products: [
      {
        brandName: 'Generic Warfarin',
        route: 'oral',
        form: 'tablet',
        allowedIntakeType: 'Pill/Tablet',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily'],
        strengths: [
          { value: 1, unit: 'mg', frequency: 'daily' },
          { value: 2, unit: 'mg', frequency: 'daily' },
          { value: 3, unit: 'mg', frequency: 'daily' },
          { value: 4, unit: 'mg', frequency: 'daily' },
          { value: 5, unit: 'mg', frequency: 'daily' }
        ]
      }
    ]
  },

  // Topical Medications
  {
    genericName: 'Diclofenac',
    atcClass: 'M02AA15',
    synonyms: ['NSAID', 'Voltarol', 'Flector'],
    products: [
      {
        brandName: 'Voltarol Gel',
        route: 'topical',
        form: 'gel',
        allowedIntakeType: 'Topical',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['twice daily', 'three times daily', 'four times daily'],
        strengths: [
          { value: 1, unit: '%', frequency: 'three times daily' },
          { value: 2.32, unit: '%', frequency: 'three times daily' }
        ]
      }
    ]
  },

  // Eye Medications
  {
    genericName: 'Latanoprost',
    atcClass: 'S01EE01',
    synonyms: ['Xalatan', 'Monopost'],
    products: [
      {
        brandName: 'Xalatan',
        route: 'ophthalmic',
        form: 'drops',
        allowedIntakeType: 'Drops',
        defaultPlaces: ['at home', 'self administered'],
        allowedFrequencies: ['daily'],
        strengths: [
          { value: 0.005, unit: '%', frequency: 'daily' }
        ]
      }
    ]
  }
];

async function seedNHSData() {
  try {
    console.log('🌱 Starting NHS Medicines A-Z data seeding...');

    // Clear existing data
    await prisma.medicationStrength.deleteMany();
    await prisma.medicationProduct.deleteMany();
    await prisma.medicationValidation.deleteMany();

    console.log('🧹 Cleared existing medication data');

    // Seed medications
    for (const medData of nhsMedications) {
      console.log(`💊 Seeding: ${medData.genericName}`);

      // Create medication using the correct model name
      const medication = await prisma.medicationValidation.create({
        data: {
          genericName: medData.genericName,
          atcClass: medData.atcClass,
          synonyms: JSON.stringify(medData.synonyms)
        }
      });

      // Create products
      for (const productData of medData.products) {
        const product = await prisma.medicationProduct.create({
          data: {
            medicationId: medication.id,
            brandName: productData.brandName,
            route: productData.route,
            form: productData.form,
            allowedIntakeType: productData.allowedIntakeType,
            defaultPlaces: JSON.stringify(productData.defaultPlaces),
            allowedFrequencies: JSON.stringify(productData.allowedFrequencies),
            isActive: true
          }
        });

        // Create strengths
        for (const strengthData of productData.strengths) {
          await prisma.medicationStrength.create({
            data: {
              productId: product.id,
              strengthValue: strengthData.value,
              strengthUnit: strengthData.unit,
              frequency: strengthData.frequency,
              label: `${strengthData.value} ${strengthData.unit} ${strengthData.frequency}`,
              isActive: true
            }
          });
        }
      }
    }

    console.log('✅ NHS Medicines A-Z data seeding completed successfully!');
    
    // Display summary
    const medicationCount = await prisma.medicationValidation.count();
    const productCount = await prisma.medicationProduct.count();
    const strengthCount = await prisma.medicationStrength.count();
    
    console.log('\n📊 Seeding Summary:');
    console.log(`   Medications: ${medicationCount}`);
    console.log(`   Products: ${productCount}`);
    console.log(`   Strengths: ${strengthCount}`);

  } catch (error) {
    console.error('❌ NHS data seeding failed:', error);
    throw error;
  } finally {
    await prisma.$disconnect();
  }
}

// Run seeding if called directly
if (require.main === module) {
  seedNHSData()
    .then(() => {
      console.log('🎉 NHS data seeding completed!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('💥 NHS data seeding failed:', error);
      process.exit(1);
    });
}

module.exports = { seedNHSData };

const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Starting medication validation system seed...');

  // Clear existing data
  await prisma.userMedicationCycle.deleteMany();
  await prisma.medicationValidationRule.deleteMany();
  await prisma.medicationStrength.deleteMany();
  await prisma.medicationProduct.deleteMany();
  await prisma.medicationValidation.deleteMany();

  // 1. ANALGESIC/OTC MEDICATIONS
  console.log('📦 Seeding Analgesic/OTC medications...');

  // Ibuprofen
  const ibuprofen = await prisma.medicationValidation.create({
    data: {
      genericName: 'ibuprofen',
      atcClass: 'M01AE01',
      classHuman: 'NSAID (Non-steroidal anti-inflammatory drug)',
      synonyms: JSON.stringify(['advil', 'nurofen', 'motrin', 'brufen', 'nsaid', 'anti-inflammatory'])
    }
  });

  // Ibuprofen products
  const ibuprofenTablet = await prisma.medicationProduct.create({
    data: {
      medicationId: ibuprofen.id,
      brandName: 'Advil/Nurofen',
      route: 'oral',
      form: 'tablet',
      allowedIntakeType: 'Pill/Tablet',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily', 'twice daily', 'as needed']),
      notes: 'Standard OTC pain relief tablets'
    }
  });

  const ibuprofenSuspension = await prisma.medicationProduct.create({
    data: {
      medicationId: ibuprofen.id,
      brandName: 'Advil/Nurofen Suspension',
      route: 'oral',
      form: 'suspension',
      allowedIntakeType: 'Liquid',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily', 'twice daily', 'as needed']),
      notes: 'Liquid suspension for children and adults who prefer liquid form'
    }
  });

  // Ibuprofen strengths
  await prisma.medicationStrength.createMany({
    data: [
      { productId: ibuprofenTablet.id, strengthValue: 200, strengthUnit: 'mg', frequency: 'as needed', label: '200 mg as needed' },
      { productId: ibuprofenTablet.id, strengthValue: 400, strengthUnit: 'mg', frequency: 'as needed', label: '400 mg as needed' },
      { productId: ibuprofenTablet.id, strengthValue: 600, strengthUnit: 'mg', frequency: 'as needed', label: '600 mg as needed' },
      { productId: ibuprofenTablet.id, strengthValue: 800, strengthUnit: 'mg', frequency: 'as needed', label: '800 mg as needed' },
      { productId: ibuprofenSuspension.id, strengthValue: 100, strengthUnit: 'mg/5mL', frequency: 'as needed', label: '100 mg/5 mL as needed' }
    ]
  });

  // Paracetamol/Acetaminophen
  const paracetamol = await prisma.medicationValidation.create({
    data: {
      genericName: 'paracetamol',
      atcClass: 'N02BE01',
      classHuman: 'Analgesic and antipyretic',
      synonyms: JSON.stringify(['acetaminophen', 'tylenol', 'panadol', 'calpol', 'analgesic', 'antipyretic'])
    }
  });

  const paracetamolTablet = await prisma.medicationProduct.create({
    data: {
      medicationId: paracetamol.id,
      brandName: 'Tylenol/Panadol',
      route: 'oral',
      form: 'tablet',
      allowedIntakeType: 'Pill/Tablet',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily', 'twice daily', 'as needed']),
      notes: 'Standard OTC pain relief and fever reduction'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: paracetamolTablet.id, strengthValue: 500, strengthUnit: 'mg', frequency: 'as needed', label: '500 mg as needed' },
      { productId: paracetamolTablet.id, strengthValue: 1000, strengthUnit: 'mg', frequency: 'as needed', label: '1000 mg as needed' }
    ]
  });

  // 2. DIABETES/ENDOCRINE (GLP-1) MEDICATIONS
  console.log('💉 Seeding Diabetes/Endocrine (GLP-1) medications...');

  // Semaglutide
  const semaglutide = await prisma.medicationValidation.create({
    data: {
      genericName: 'semaglutide',
      atcClass: 'A10BJ02',
      classHuman: 'GLP-1 receptor agonist',
      synonyms: JSON.stringify(['glp1', 'glp-1', 'glp1ra', 'glp-1ra', 'ozempic', 'wegovy', 'rybelsus'])
    }
  });

  // Ozempic (injection)
  const ozempic = await prisma.medicationProduct.create({
    data: {
      medicationId: semaglutide.id,
      brandName: 'Ozempic',
      route: 'subcutaneous',
      form: 'injection-pen',
      allowedIntakeType: 'Injection',
      defaultPlaces: JSON.stringify(['at home', 'at clinic', 'pharmacy administration', 'self administered']),
      allowedFrequencies: JSON.stringify(['weekly']),
      notes: 'Weekly subcutaneous injection for type 2 diabetes'
    }
  });

  // Rybelsus (oral)
  const rybelsus = await prisma.medicationProduct.create({
    data: {
      medicationId: semaglutide.id,
      brandName: 'Rybelsus',
      route: 'oral',
      form: 'tablet',
      allowedIntakeType: 'Pill/Tablet',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily']),
      notes: 'Daily oral tablet for type 2 diabetes'
    }
  });

  // Ozempic strengths
  await prisma.medicationStrength.createMany({
    data: [
      { productId: ozempic.id, strengthValue: 0.25, strengthUnit: 'mg', frequency: 'weekly', label: '0.25 mg once weekly' },
      { productId: ozempic.id, strengthValue: 0.5, strengthUnit: 'mg', frequency: 'weekly', label: '0.5 mg once weekly' },
      { productId: ozempic.id, strengthValue: 1, strengthUnit: 'mg', frequency: 'weekly', label: '1 mg once weekly' },
      { productId: ozempic.id, strengthValue: 2, strengthUnit: 'mg', frequency: 'weekly', label: '2 mg once weekly' }
    ]
  });

  // Rybelsus strengths
  await prisma.medicationStrength.createMany({
    data: [
      { productId: rybelsus.id, strengthValue: 3, strengthUnit: 'mg', frequency: 'daily', label: '3 mg once daily' },
      { productId: rybelsus.id, strengthValue: 7, strengthUnit: 'mg', frequency: 'daily', label: '7 mg once daily' },
      { productId: rybelsus.id, strengthValue: 14, strengthUnit: 'mg', frequency: 'daily', label: '14 mg once daily' }
    ]
  });

  // Liraglutide
  const liraglutide = await prisma.medicationValidation.create({
    data: {
      genericName: 'liraglutide',
      atcClass: 'A10BJ01',
      classHuman: 'GLP-1 receptor agonist',
      synonyms: JSON.stringify(['glp1', 'glp-1', 'glp1ra', 'glp-1ra', 'victoza', 'saxenda'])
    }
  });

  const victoza = await prisma.medicationProduct.create({
    data: {
      medicationId: liraglutide.id,
      brandName: 'Victoza',
      route: 'subcutaneous',
      form: 'injection-pen',
      allowedIntakeType: 'Injection',
      defaultPlaces: JSON.stringify(['at home', 'at clinic', 'pharmacy administration', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily']),
      notes: 'Daily subcutaneous injection for type 2 diabetes'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: victoza.id, strengthValue: 0.6, strengthUnit: 'mg', frequency: 'daily', label: '0.6 mg once daily' },
      { productId: victoza.id, strengthValue: 1.2, strengthUnit: 'mg', frequency: 'daily', label: '1.2 mg once daily' },
      { productId: victoza.id, strengthValue: 1.8, strengthUnit: 'mg', frequency: 'daily', label: '1.8 mg once daily' }
    ]
  });

  // Dulaglutide
  const dulaglutide = await prisma.medicationValidation.create({
    data: {
      genericName: 'dulaglutide',
      atcClass: 'A10BJ03',
      classHuman: 'GLP-1 receptor agonist',
      synonyms: JSON.stringify(['glp1', 'glp-1', 'glp1ra', 'glp-1ra', 'trulicity'])
    }
  });

  const trulicity = await prisma.medicationProduct.create({
    data: {
      medicationId: dulaglutide.id,
      brandName: 'Trulicity',
      route: 'subcutaneous',
      form: 'injection-pen',
      allowedIntakeType: 'Injection',
      defaultPlaces: JSON.stringify(['at home', 'at clinic', 'pharmacy administration', 'self administered']),
      allowedFrequencies: JSON.stringify(['weekly']),
      notes: 'Weekly subcutaneous injection for type 2 diabetes'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: trulicity.id, strengthValue: 0.75, strengthUnit: 'mg', frequency: 'weekly', label: '0.75 mg once weekly' },
      { productId: trulicity.id, strengthValue: 1.5, strengthUnit: 'mg', frequency: 'weekly', label: '1.5 mg once weekly' },
      { productId: trulicity.id, strengthValue: 3, strengthUnit: 'mg', frequency: 'weekly', label: '3 mg once weekly' },
      { productId: trulicity.id, strengthValue: 4.5, strengthUnit: 'mg', frequency: 'weekly', label: '4.5 mg once weekly' }
    ]
  });

  // 3. METABOLIC MEDICATIONS
  console.log('💊 Seeding Metabolic medications...');

  // Metformin
  const metformin = await prisma.medicationValidation.create({
    data: {
      genericName: 'metformin',
      atcClass: 'A10BA02',
      classHuman: 'Biguanide',
      synonyms: JSON.stringify(['glucophage', 'fortamet', 'glumetza', 'biguanide', 'diabetes'])
    }
  });

  const metforminIR = await prisma.medicationProduct.create({
    data: {
      medicationId: metformin.id,
      brandName: 'Generic Metformin (IR)',
      route: 'oral',
      form: 'tablet',
      allowedIntakeType: 'Pill/Tablet',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily', 'twice daily', 'three times daily']),
      notes: 'Immediate release metformin tablets'
    }
  });

  const metforminER = await prisma.medicationProduct.create({
    data: {
      medicationId: metformin.id,
      brandName: 'Generic Metformin (ER)',
      route: 'oral',
      form: 'tablet',
      allowedIntakeType: 'Pill/Tablet',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily']),
      notes: 'Extended release metformin tablets'
    }
  });

  // Metformin IR strengths
  await prisma.medicationStrength.createMany({
    data: [
      { productId: metforminIR.id, strengthValue: 500, strengthUnit: 'mg', frequency: 'twice daily', label: '500 mg twice daily' },
      { productId: metforminIR.id, strengthValue: 850, strengthUnit: 'mg', frequency: 'twice daily', label: '850 mg twice daily' },
      { productId: metforminIR.id, strengthValue: 1000, strengthUnit: 'mg', frequency: 'twice daily', label: '1000 mg twice daily' }
    ]
  });

  // Metformin ER strengths
  await prisma.medicationStrength.createMany({
    data: [
      { productId: metforminER.id, strengthValue: 500, strengthUnit: 'mg', frequency: 'daily', label: '500 mg once daily' },
      { productId: metforminER.id, strengthValue: 1000, strengthUnit: 'mg', frequency: 'daily', label: '1000 mg once daily' }
    ]
  });

  // 4. RESPIRATORY MEDICATIONS
  console.log('🫁 Seeding Respiratory medications...');

  // Salbutamol/Albuterol
  const salbutamol = await prisma.medicationValidation.create({
    data: {
      genericName: 'salbutamol',
      atcClass: 'R03AC02',
      classHuman: 'Short-acting beta-2 agonist (SABA)',
      synonyms: JSON.stringify(['albuterol', 'ventolin', 'proair', 'proventil', 'saba', 'bronchodilator'])
    }
  });

  const salbutamolMDI = await prisma.medicationProduct.create({
    data: {
      medicationId: salbutamol.id,
      brandName: 'Ventolin/ProAir',
      route: 'inhalation',
      form: 'mdi-inhaler',
      allowedIntakeType: 'Inhaler',
      defaultPlaces: JSON.stringify(['at home', 'at clinic', 'self administered']),
      allowedFrequencies: JSON.stringify(['as needed']),
      notes: 'Metered dose inhaler for acute asthma symptoms'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: salbutamolMDI.id, strengthValue: 100, strengthUnit: 'mcg/puff', frequency: 'as needed', label: '100 mcg per puff as needed' }
    ]
  });

  // 5. CARDIOVASCULAR MEDICATIONS
  console.log('❤️ Seeding Cardiovascular medications...');

  // Aspirin
  const aspirin = await prisma.medicationValidation.create({
    data: {
      genericName: 'aspirin',
      atcClass: 'B01AC06',
      classHuman: 'Antiplatelet agent',
      synonyms: JSON.stringify(['acetylsalicylic acid', 'asa', 'antiplatelet', 'blood thinner'])
    }
  });

  const aspirinLowDose = await prisma.medicationProduct.create({
    data: {
      medicationId: aspirin.id,
      brandName: 'Low-dose Aspirin',
      route: 'oral',
      form: 'tablet',
      allowedIntakeType: 'Pill/Tablet',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily']),
      notes: 'Low-dose aspirin for cardiovascular protection'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: aspirinLowDose.id, strengthValue: 81, strengthUnit: 'mg', frequency: 'daily', label: '81 mg once daily' },
      { productId: aspirinLowDose.id, strengthValue: 100, strengthUnit: 'mg', frequency: 'daily', label: '100 mg once daily' }
    ]
  });

  // 6. INSULIN
  console.log('🩸 Seeding Insulin medications...');

  // Insulin Glargine
  const insulinGlargine = await prisma.medicationValidation.create({
    data: {
      genericName: 'insulin glargine',
      atcClass: 'A10AB01',
      classHuman: 'Long-acting insulin analog',
      synonyms: JSON.stringify(['lantus', 'basaglar', 'toujeo', 'long-acting insulin', 'basal insulin'])
    }
  });

  const lantus = await prisma.medicationProduct.create({
    data: {
      medicationId: insulinGlargine.id,
      brandName: 'Lantus',
      route: 'subcutaneous',
      form: 'injection-pen',
      allowedIntakeType: 'Injection',
      defaultPlaces: JSON.stringify(['at home', 'at clinic', 'pharmacy administration', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily']),
      notes: 'Long-acting basal insulin injection'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: lantus.id, strengthValue: 100, strengthUnit: 'units/mL', frequency: 'daily', label: '100 units/mL once daily' }
    ]
  });

  // 7. TOPICAL/OPHTHALMIC MEDICATIONS
  console.log('👁️ Seeding Topical/Ophthalmic medications...');

  // Diclofenac Gel
  const diclofenac = await prisma.medicationValidation.create({
    data: {
      genericName: 'diclofenac',
      atcClass: 'M02AA15',
      classHuman: 'NSAID (Topical)',
      synonyms: JSON.stringify(['voltaren', 'voltarol', 'nsaid', 'anti-inflammatory', 'topical'])
    }
  });

  const diclofenacGel = await prisma.medicationProduct.create({
    data: {
      medicationId: diclofenac.id,
      brandName: 'Voltaren Gel',
      route: 'topical',
      form: 'gel',
      allowedIntakeType: 'Topical',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily', 'twice daily', 'as needed']),
      notes: 'Topical NSAID gel for localized pain relief'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: diclofenacGel.id, strengthValue: 1, strengthUnit: '%', frequency: 'as needed', label: '1% gel as needed' }
    ]
  });

  // Latanoprost
  const latanoprost = await prisma.medicationValidation.create({
    data: {
      genericName: 'latanoprost',
      atcClass: 'S01EE01',
      classHuman: 'Prostaglandin F2α analog',
      synonyms: JSON.stringify(['xalatan', 'prostaglandin', 'glaucoma', 'eye drops'])
    }
  });

  const latanoprostDrops = await prisma.medicationProduct.create({
    data: {
      medicationId: latanoprost.id,
      brandName: 'Xalatan',
      route: 'ophthalmic',
      form: 'drops',
      allowedIntakeType: 'Eye/Ear Drops',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['nightly']),
      notes: 'Eye drops for glaucoma treatment'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: latanoprostDrops.id, strengthValue: 0.005, strengthUnit: '%', frequency: 'nightly', label: '0.005% drops nightly' }
    ]
  });

  // 8. ADDITIONAL COMMON MEDICATIONS
  console.log('💊 Seeding additional common medications...');

  // Lisinopril
  const lisinopril = await prisma.medicationValidation.create({
    data: {
      genericName: 'lisinopril',
      atcClass: 'C09AA03',
      classHuman: 'ACE inhibitor',
      synonyms: JSON.stringify(['prinivil', 'zestril', 'ace inhibitor', 'blood pressure'])
    }
  });

  const lisinoprilTablet = await prisma.medicationProduct.create({
    data: {
      medicationId: lisinopril.id,
      brandName: 'Prinivil/Zestril',
      route: 'oral',
      form: 'tablet',
      allowedIntakeType: 'Pill/Tablet',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily']),
      notes: 'ACE inhibitor for hypertension and heart failure'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: lisinoprilTablet.id, strengthValue: 5, strengthUnit: 'mg', frequency: 'daily', label: '5 mg once daily' },
      { productId: lisinoprilTablet.id, strengthValue: 10, strengthUnit: 'mg', frequency: 'daily', label: '10 mg once daily' },
      { productId: lisinoprilTablet.id, strengthValue: 20, strengthUnit: 'mg', frequency: 'daily', label: '20 mg once daily' },
      { productId: lisinoprilTablet.id, strengthValue: 40, strengthUnit: 'mg', frequency: 'daily', label: '40 mg once daily' }
    ]
  });

  // Atorvastatin
  const atorvastatin = await prisma.medicationValidation.create({
    data: {
      genericName: 'atorvastatin',
      atcClass: 'C10AA05',
      classHuman: 'HMG-CoA reductase inhibitor (Statin)',
      synonyms: JSON.stringify(['lipitor', 'statin', 'cholesterol', 'hmg-coa'])
    }
  });

  const atorvastatinTablet = await prisma.medicationProduct.create({
    data: {
      medicationId: atorvastatin.id,
      brandName: 'Lipitor',
      route: 'oral',
      form: 'tablet',
      allowedIntakeType: 'Pill/Tablet',
      defaultPlaces: JSON.stringify(['at home', 'self administered']),
      allowedFrequencies: JSON.stringify(['daily']),
      notes: 'Statin for cholesterol management'
    }
  });

  await prisma.medicationStrength.createMany({
    data: [
      { productId: atorvastatinTablet.id, strengthValue: 10, strengthUnit: 'mg', frequency: 'daily', label: '10 mg once daily' },
      { productId: atorvastatinTablet.id, strengthValue: 20, strengthUnit: 'mg', frequency: 'daily', label: '20 mg once daily' },
      { productId: atorvastatinTablet.id, strengthValue: 40, strengthUnit: 'mg', frequency: 'daily', label: '40 mg once daily' },
      { productId: atorvastatinTablet.id, strengthValue: 80, strengthUnit: 'mg', frequency: 'daily', label: '80 mg once daily' }
    ]
  });

  // Create validation rules for some products
  console.log('📋 Creating validation rules...');

  await prisma.medicationValidationRule.createMany({
    data: [
      {
        productId: ozempic.id,
        maxDosePerPeriod: '2 mg/week',
        minDosePerPeriod: '0.25 mg/week',
        contraindications: JSON.stringify(['pregnancy', 'diabetic ketoacidosis', 'severe gastrointestinal disease']),
        warnings: JSON.stringify(['nausea', 'vomiting', 'diarrhea', 'pancreatitis risk']),
        version: 1
      },
      {
        productId: rybelsus.id,
        maxDosePerPeriod: '14 mg/day',
        minDosePerPeriod: '3 mg/day',
        contraindications: JSON.stringify(['pregnancy', 'diabetic ketoacidosis', 'severe gastrointestinal disease']),
        warnings: JSON.stringify(['take on empty stomach', 'nausea', 'vomiting']),
        version: 1
      },
      {
        productId: metforminIR.id,
        maxDosePerPeriod: '2550 mg/day',
        minDosePerPeriod: '500 mg/day',
        contraindications: JSON.stringify(['severe kidney disease', 'metabolic acidosis', 'heart failure']),
        warnings: JSON.stringify(['gastrointestinal upset', 'lactic acidosis risk', 'vitamin B12 deficiency']),
        version: 1
      }
    ]
  });

  console.log('✅ Medication validation system seed completed successfully!');
  console.log(`📊 Created ${await prisma.medicationValidation.count()} medications`);
  console.log(`📦 Created ${await prisma.medicationProduct.count()} products`);
  console.log(`💊 Created ${await prisma.medicationStrength.count()} strengths`);
  console.log(`📋 Created ${await prisma.medicationValidationRule.count()} validation rules`);
}

main()
  .catch((e) => {
    console.error('❌ Error seeding database:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

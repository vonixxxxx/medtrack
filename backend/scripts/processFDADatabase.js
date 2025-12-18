/**
 * Pre-process FDA database files into a smaller, searchable index
 * This script processes the large JSON files and creates a compact index
 */

const fs = require('fs');
const path = require('path');

const DATASET_PATH = '/Users/AlexanderSokol/Desktop/medication_dataset';
const OUTPUT_PATH = path.join(__dirname, '../data/fda_medication_index.json');
const DATASET_FILES = [
  'drug-label-0001-of-0013.json',
  'drug-label-0002-of-0013.json',
  'drug-label-0003-of-0013.json',
  'drug-label-0004-of-0013.json',
  'drug-label-0005-of-0013.json',
  'drug-label-0006-of-0013.json',
  'drug-label-0007-of-0013.json',
  'drug-label-0008-of-0013.json',
  'drug-label-0009-of-0013.json',
  'drug-label-0010-of-0013.json',
  'drug-label-0011-of-0013.json',
  'drug-label-0012-of-0013.json',
  'drug-label-0013-of-0013.json'
];

// Same extraction functions as in the service
function extractDrugName(record) {
  if (record.spl_product_data_elements && record.spl_product_data_elements.length > 0) {
    const text = record.spl_product_data_elements[0].toUpperCase();
    const comboMatch = text.match(/^([A-Z][A-Z\s]{2,30}?)(?:\s+AND\s+[A-Z][A-Z\s]+?)?(?:\s+TABLETS?|\s+CAPSULES?|\s+INJECTION)/);
    if (comboMatch) {
      let drugName = comboMatch[1].trim();
      const words = drugName.split(/\s+/).filter(w => {
        const upper = w.toUpperCase();
        return w.length >= 3 && w.length <= 25 && /^[A-Z]+$/.test(upper);
      });
      if (words.length > 0) {
        return words[0].charAt(0) + words[0].slice(1).toLowerCase();
      }
    }
  }
  if (record.description && record.description.length > 0) {
    const desc = record.description[0];
    const match = desc.match(/([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Tablets?|Capsules?|Injection)/);
    if (match) return match[1].trim();
  }
  return null;
}

function extractDrugClass(record) {
  const text = [
    ...(record.mechanism_of_action || []),
    ...(record.description || []),
    ...(record.clinical_pharmacology || [])
  ].join(' ').toLowerCase();
  
  const patterns = [
    { pattern: /ace\s+inhibitor|angiotensin\s+converting\s+enzyme/i, class: 'ACE Inhibitor' },
    { pattern: /beta[\s-]?blocker|beta[\s-]?adrenergic/i, class: 'Beta Blocker' },
    { pattern: /glp[\s-]?1|glucagon[\s-]?like\s+peptide/i, class: 'GLP-1 Receptor Agonist' },
    { pattern: /biguanide/i, class: 'Biguanide' },
    { pattern: /statin|hmg[\s-]?coa/i, class: 'Statin' },
    { pattern: /ssri|selective\s+serotonin/i, class: 'SSRI' },
    { pattern: /nsaid|nonsteroidal/i, class: 'NSAID' },
    { pattern: /proton\s+pump|ppi/i, class: 'Proton Pump Inhibitor' },
    { pattern: /analgesic|antipyretic/i, class: 'Analgesic' },
    { pattern: /opioid|narcotic/i, class: 'Opioid' },
    { pattern: /anticoagulant/i, class: 'Anticoagulant' },
    { pattern: /antibiotic/i, class: 'Antibiotic' },
    { pattern: /diuretic/i, class: 'Diuretic' }
  ];
  
  for (const { pattern, class: drugClass } of patterns) {
    if (pattern.test(text)) return drugClass;
  }
  return null;
}

function extractDosages(record) {
  const dosages = new Set();
  const text = [
    ...(record.description || []),
    ...(record.dosage_and_administration || []),
    ...(record.how_supplied || [])
  ].join(' ');
  
  const matches = text.matchAll(/\b(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|units?|iu|ml|l|%)\b/gi);
  for (const match of matches) {
    if (match[0].length <= 20) dosages.add(match[0].trim());
  }
  return Array.from(dosages).slice(0, 10);
}

function extractDosageForms(record) {
  const forms = new Set();
  const text = [
    ...(record.description || []),
    ...(record.how_supplied || [])
  ].join(' ').toLowerCase();
  
  const matches = text.matchAll(/\b(tablets?|capsules?|injections?|solutions?|creams?|gels?)\b/gi);
  for (const match of matches) {
    forms.add(match[1].charAt(0).toUpperCase() + match[1].slice(1));
  }
  return Array.from(forms).slice(0, 5);
}

async function processFiles() {
  console.log('🚀 Processing FDA database files...\n');
  const drugMap = new Map();
  let totalRecords = 0;
  let totalProcessed = 0;
  
  for (let i = 0; i < DATASET_FILES.length; i++) {
    const filename = DATASET_FILES[i];
    const filePath = path.join(DATASET_PATH, filename);
    
    if (!fs.existsSync(filePath)) {
      console.warn(`⚠️  File not found: ${filePath}`);
      continue;
    }
    
    try {
      console.log(`📂 Processing ${filename} (${i + 1}/${DATASET_FILES.length})...`);
      const fileStartTime = Date.now();
      
      // Read file in chunks
      const fileStats = fs.statSync(filePath);
      console.log(`   File size: ${(fileStats.size / 1024 / 1024).toFixed(2)} MB`);
      
      // Try to parse - if it fails due to size, we'll handle it
      try {
        const fileContent = fs.readFileSync(filePath, 'utf8');
        const data = JSON.parse(fileContent);
        
        if (data.results && Array.isArray(data.results)) {
          totalRecords += data.results.length;
          let fileProcessed = 0;
          
          // Process in batches
          const batchSize = 500;
          for (let j = 0; j < data.results.length; j += batchSize) {
            const batch = data.results.slice(j, Math.min(j + batchSize, data.results.length));
            
            for (const record of batch) {
              const drugName = extractDrugName(record);
              if (drugName && drugName.length >= 3) {
                const key = drugName.toLowerCase();
                
                if (!drugMap.has(key)) {
                  drugMap.set(key, {
                    generic_name: key,
                    display_name: drugName,
                    drug_class: extractDrugClass(record) || 'Unknown',
                    dosage_forms: extractDosageForms(record),
                    typical_strengths: extractDosages(record),
                    record_count: 1
                  });
                  fileProcessed++;
                  totalProcessed++;
                } else {
                  const existing = drugMap.get(key);
                  existing.record_count++;
                  
                  // Merge data
                  const newForms = extractDosageForms(record);
                  newForms.forEach(f => {
                    if (!existing.dosage_forms.includes(f)) {
                      existing.dosage_forms.push(f);
                    }
                  });
                  
                  const newStrengths = extractDosages(record);
                  newStrengths.forEach(s => {
                    if (!existing.typical_strengths.includes(s)) {
                      existing.typical_strengths.push(s);
                    }
                  });
                  
                  if (existing.drug_class === 'Unknown') {
                    const dc = extractDrugClass(record);
                    if (dc) existing.drug_class = dc;
                  }
                }
              }
            }
          }
          
          const fileTime = ((Date.now() - fileStartTime) / 1000).toFixed(2);
          console.log(`   ✓ Processed ${fileProcessed} unique drugs from ${data.results.length} records in ${fileTime}s\n`);
        }
      } catch (parseError) {
        console.log(`   ⚠️  File too large or malformed, skipping: ${parseError.message}\n`);
        continue;
      }
    } catch (error) {
      console.error(`   ❌ Error: ${error.message}\n`);
      continue;
    }
  }
  
  // Save index
  const index = Array.from(drugMap.values());
  console.log(`\n✅ Processing complete!`);
  console.log(`   Total records: ${totalRecords.toLocaleString()}`);
  console.log(`   Unique medications: ${index.length.toLocaleString()}`);
  console.log(`\n💾 Saving index to ${OUTPUT_PATH}...`);
  
  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(index, null, 2));
  const indexSize = (fs.statSync(OUTPUT_PATH).size / 1024 / 1024).toFixed(2);
  console.log(`✓ Index saved (${indexSize} MB)\n`);
  
  return index;
}

if (require.main === module) {
  processFiles()
    .then(() => {
      console.log('✅ Done!');
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Error:', error);
      process.exit(1);
    });
}

module.exports = { processFiles };




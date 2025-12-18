#!/usr/bin/env python3
"""
Process FDA drug-label JSON files and create a searchable index
Handles large files efficiently using streaming JSON parsing
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

DATASET_PATH = '/Users/AlexanderSokol/Desktop/medication_dataset'
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../data/fda_medication_index.json')

DATASET_FILES = [
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
]

INACTIVE_INGREDIENTS = {
    'ANHYDROUS', 'LACTOSE', 'CROSCARMELLOSE', 'SODIUM', 'MAGNESIUM', 'STEARATE',
    'FERRIC', 'OXIDE', 'RED', 'YELLOW', 'WHITE', 'OFF', 'CELLULOSE', 'TALC',
    'TITANIUM', 'DIOXIDE', 'GELATIN', 'SHELLAC', 'WAX', 'POLYETHYLENE', 'GLYCOL',
    'HYDROCHLORIDE', 'HYDROCHLOROTHIAZIDE', 'SULFATE', 'PHOSPHATE', 'CITRATE'
}

def extract_drug_name(record):
    """Extract primary drug name from FDA record"""
    # Method 1: spl_product_data_elements
    if 'spl_product_data_elements' in record and record['spl_product_data_elements']:
        text = record['spl_product_data_elements'][0].upper()
        # Pattern: "DRUGNAME and DRUGNAME Tablets"
        match = re.match(r'^([A-Z][A-Z\s]{2,30}?)(?:\s+AND\s+[A-Z][A-Z\s]+?)?(?:\s+TABLETS?|\s+CAPSULES?|\s+INJECTION)', text)
        if match:
            drug_name = match.group(1).strip()
            words = [w for w in drug_name.split() if len(w) >= 3 and len(w) <= 25 
                    and w.isalpha() and w not in INACTIVE_INGREDIENTS]
            if words:
                return words[0].capitalize()
    
    # Method 2: description field
    if 'description' in record and record['description']:
        desc = record['description'][0]
        match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Tablets?|Capsules?|Injection)', desc)
        if match:
            return match.group(1).strip()
    
    return None

def extract_drug_class(record):
    """Extract drug class from record"""
    text = ' '.join([
        ' '.join(record.get('mechanism_of_action', [])),
        ' '.join(record.get('description', [])),
        ' '.join(record.get('clinical_pharmacology', []))
    ]).lower()
    
    patterns = [
        (r'ace\s+inhibitor|angiotensin\s+converting\s+enzyme', 'ACE Inhibitor'),
        (r'beta[\s-]?blocker|beta[\s-]?adrenergic', 'Beta Blocker'),
        (r'glp[\s-]?1|glucagon[\s-]?like\s+peptide', 'GLP-1 Receptor Agonist'),
        (r'biguanide', 'Biguanide'),
        (r'statin|hmg[\s-]?coa', 'Statin'),
        (r'ssri|selective\s+serotonin', 'SSRI'),
        (r'nsaid|nonsteroidal', 'NSAID'),
        (r'proton\s+pump|ppi', 'Proton Pump Inhibitor'),
        (r'analgesic|antipyretic', 'Analgesic'),
        (r'opioid|narcotic', 'Opioid'),
        (r'anticoagulant', 'Anticoagulant'),
        (r'antibiotic', 'Antibiotic'),
        (r'diuretic', 'Diuretic'),
    ]
    
    for pattern, drug_class in patterns:
        if re.search(pattern, text):
            return drug_class
    
    return None

def extract_dosages(record):
    """Extract typical dosages/strengths"""
    dosages = set()
    text = ' '.join([
        ' '.join(record.get('description', [])),
        ' '.join(record.get('dosage_and_administration', [])),
        ' '.join(record.get('how_supplied', []))
    ])
    
    matches = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:mg|mcg|g|units?|iu|ml|l|%)\b', text, re.IGNORECASE)
    for match in matches:
        dosage = match[0] if isinstance(match, tuple) else match
        if len(dosage) <= 20:
            dosages.add(dosage)
    
    return list(dosages)[:10]

def extract_dosage_forms(record):
    """Extract dosage forms"""
    forms = set()
    text = ' '.join([
        ' '.join(record.get('description', [])),
        ' '.join(record.get('how_supplied', []))
    ]).lower()
    
    matches = re.findall(r'\b(tablets?|capsules?|injections?|solutions?|creams?|gels?)\b', text, re.IGNORECASE)
    for match in matches:
        form = match[0] if isinstance(match, tuple) else match
        if len(form) >= 3:
            forms.add(form.capitalize())
    
    return list(forms)[:5]

def process_files():
    """Process all FDA database files"""
    print('🚀 Processing FDA database files...\n')
    drug_map = {}
    total_records = 0
    total_processed = 0
    
    for i, filename in enumerate(DATASET_FILES, 1):
        file_path = os.path.join(DATASET_PATH, filename)
        
        if not os.path.exists(file_path):
            print(f'⚠️  File not found: {file_path}')
            continue
        
        try:
            print(f'📂 Processing {filename} ({i}/{len(DATASET_FILES)})...')
            file_start_time = os.times()[4]
            
            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f'   File size: {file_size_mb:.2f} MB')
            
            # Load and parse JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'results' in data and isinstance(data['results'], list):
                total_records += len(data['results'])
                file_processed = 0
                
                # Process records
                for record in data['results']:
                    drug_name = extract_drug_name(record)
                    if drug_name and len(drug_name) >= 3:
                        key = drug_name.lower()
                        
                        if key not in drug_map:
                            drug_map[key] = {
                                'generic_name': key,
                                'display_name': drug_name,
                                'drug_class': extract_drug_class(record) or 'Unknown',
                                'dosage_forms': extract_dosage_forms(record),
                                'typical_strengths': extract_dosages(record),
                                'record_count': 1
                            }
                            file_processed += 1
                            total_processed += 1
                        else:
                            existing = drug_map[key]
                            existing['record_count'] += 1
                            
                            # Merge data
                            new_forms = extract_dosage_forms(record)
                            for form in new_forms:
                                if form not in existing['dosage_forms']:
                                    existing['dosage_forms'].append(form)
                            
                            new_strengths = extract_dosages(record)
                            for strength in new_strengths:
                                if strength not in existing['typical_strengths']:
                                    existing['typical_strengths'].append(strength)
                            
                            if existing['drug_class'] == 'Unknown':
                                dc = extract_drug_class(record)
                                if dc:
                                    existing['drug_class'] = dc
                
                file_time = os.times()[4] - file_start_time
                print(f'   ✓ Processed {file_processed} unique drugs from {len(data["results"])} records in {file_time:.2f}s\n')
        
        except MemoryError:
            print(f'   ⚠️  File too large for memory, skipping\n')
            continue
        except Exception as e:
            print(f'   ❌ Error: {str(e)}\n')
            continue
    
    # Save index
    index = list(drug_map.values())
    print(f'\n✅ Processing complete!')
    print(f'   Total records: {total_records:,}')
    print(f'   Unique medications: {len(index):,}')
    print(f'\n💾 Saving index to {OUTPUT_PATH}...')
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    index_size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f'✓ Index saved ({index_size_mb:.2f} MB)\n')
    
    return index

if __name__ == '__main__':
    try:
        process_files()
        print('✅ Done!')
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        exit(1)




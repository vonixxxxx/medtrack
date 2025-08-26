# Comprehensive Drug Database as Fallback
# This provides reliable drug information when external APIs fail

DRUG_DATABASE = {
    # Pain Relief & Anti-inflammatory
    "aspirin": {
        "source": "Local Database",
        "search_term": "aspirin",
        "brand_name": "Aspirin, Bayer Aspirin, Ecotrin",
        "generic_name": "Acetylsalicylic Acid",
        "indications_and_usage": "Used to relieve pain, reduce fever, and as an anti-inflammatory. Also used in low doses to prevent heart attacks and strokes.",
        "warnings": "May cause stomach bleeding, especially in older adults. Avoid if allergic to aspirin. Do not give to children with viral infections due to Reye's syndrome risk.",
        "dosage": "Typical dose: 325-650mg every 4-6 hours as needed. Maximum: 4g per day. Take with food to reduce stomach upset.",
        "side_effects": "Stomach upset, heartburn, nausea, stomach bleeding, ringing in ears, allergic reactions.",
        "manufacturer": "Various manufacturers",
        "license": "Over-the-counter (OTC)",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    "ibuprofen": {
        "source": "Local Database",
        "search_term": "ibuprofen",
        "brand_name": "Advil, Motrin, Nurofen",
        "generic_name": "Ibuprofen",
        "indications_and_usage": "Used to relieve pain, reduce fever, and decrease inflammation. Effective for headaches, muscle pain, arthritis, and menstrual cramps.",
        "warnings": "May increase risk of heart attack and stroke. Can cause stomach bleeding. Avoid if allergic to NSAIDs. Do not use during last 3 months of pregnancy.",
        "dosage": "Typical dose: 200-400mg every 4-6 hours. Maximum: 1200mg per day. Take with food to reduce stomach upset.",
        "side_effects": "Stomach upset, heartburn, nausea, stomach bleeding, dizziness, headache, allergic reactions.",
        "manufacturer": "Various manufacturers",
        "license": "Over-the-counter (OTC)",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    "acetaminophen": {
        "source": "Local Database",
        "search_term": "acetaminophen",
        "brand_name": "Tylenol, Paracetamol, Calpol",
        "generic_name": "Acetaminophen",
        "indications_and_usage": "Used to relieve pain and reduce fever. Safe for most people including children and pregnant women.",
        "warnings": "Overdose can cause severe liver damage. Do not exceed recommended dose. Avoid alcohol while taking. Check other medications for acetaminophen content.",
        "dosage": "Typical dose: 500-1000mg every 4-6 hours. Maximum: 4g per day. Follow package instructions carefully.",
        "side_effects": "Generally well-tolerated. Rare side effects include allergic reactions, liver problems with overdose.",
        "manufacturer": "Various manufacturers",
        "license": "Over-the-counter (OTC)",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    "paracetamol": {
        "source": "Local Database",
        "search_term": "paracetamol",
        "brand_name": "Tylenol, Calpol, Panadol",
        "generic_name": "Acetaminophen",
        "indications_and_usage": "Used to relieve pain and reduce fever. Safe for most people including children and pregnant women.",
        "warnings": "Overdose can cause severe liver damage. Do not exceed recommended dose. Avoid alcohol while taking. Check other medications for paracetamol content.",
        "dosage": "Typical dose: 500-1000mg every 4-6 hours. Maximum: 4g per day. Follow package instructions carefully.",
        "side_effects": "Generally well-tolerated. Rare side effects include allergic reactions, liver problems with overdose.",
        "manufacturer": "Various manufacturers",
        "license": "Over-the-counter (OTC)",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    # Antibiotics
    "amoxicillin": {
        "source": "Local Database",
        "search_term": "amoxicillin",
        "brand_name": "Amoxil, Trimox, Amoxicot",
        "generic_name": "Amoxicillin",
        "indications_and_usage": "Antibiotic used to treat bacterial infections including ear infections, throat infections, pneumonia, and urinary tract infections.",
        "warnings": "Complete the full course of antibiotics. Do not stop early even if feeling better. May cause allergic reactions in penicillin-allergic patients.",
        "dosage": "Typical dose: 250-500mg three times daily. Dosage varies by infection type and patient weight. Take on empty stomach.",
        "side_effects": "Diarrhea, nausea, vomiting, rash, yeast infection, allergic reactions.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    "penicillin": {
        "source": "Local Database",
        "search_term": "penicillin",
        "brand_name": "Penicillin VK, Penicillin G",
        "generic_name": "Penicillin",
        "indications_and_usage": "Antibiotic used to treat bacterial infections. First-line treatment for many common infections.",
        "warnings": "Complete the full course of antibiotics. May cause severe allergic reactions in sensitive patients. Inform doctor of any allergies.",
        "dosage": "Typical dose: 250-500mg four times daily. Dosage varies by infection type. Take on empty stomach.",
        "side_effects": "Diarrhea, nausea, rash, allergic reactions, yeast infection.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    # Diabetes Medications
    "metformin": {
        "source": "Local Database",
        "search_term": "metformin",
        "brand_name": "Glucophage, Fortamet, Glumetza",
        "generic_name": "Metformin Hydrochloride",
        "indications_and_usage": "Used to treat type 2 diabetes. Helps control blood sugar levels and may help with weight loss.",
        "warnings": "Can cause lactic acidosis in rare cases. Avoid excessive alcohol. May need to stop before surgery. Monitor kidney function.",
        "dosage": "Typical starting dose: 500mg twice daily with meals. Maximum: 2550mg per day. Take with food to reduce stomach upset.",
        "side_effects": "Nausea, diarrhea, stomach upset, metallic taste, vitamin B12 deficiency with long-term use.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    "insulin": {
        "source": "Local Database",
        "search_term": "insulin",
        "brand_name": "Humulin, Novolin, Lantus, Humalog",
        "generic_name": "Insulin",
        "indications_and_usage": "Used to treat diabetes by controlling blood sugar levels. Different types available for different needs.",
        "warnings": "Can cause low blood sugar (hypoglycemia). Monitor blood sugar regularly. Rotate injection sites. Never share insulin pens.",
        "dosage": "Dosage varies greatly by individual. Must be prescribed by doctor. Monitor blood sugar levels carefully.",
        "side_effects": "Low blood sugar, weight gain, injection site reactions, allergic reactions.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    # Blood Pressure Medications
    "lisinopril": {
        "source": "Local Database",
        "search_term": "lisinopril",
        "brand_name": "Zestril, Prinivil",
        "generic_name": "Lisinopril",
        "indications_and_usage": "Used to treat high blood pressure, heart failure, and to improve survival after heart attacks.",
        "warnings": "Can cause birth defects. Do not use during pregnancy. May cause severe allergic reactions. Monitor kidney function.",
        "dosage": "Typical starting dose: 10mg once daily. Maximum: 40mg per day. Take at same time each day.",
        "side_effects": "Dry cough, dizziness, headache, fatigue, low blood pressure, kidney problems.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    "amlodipine": {
        "source": "Local Database",
        "search_term": "amlodipine",
        "brand_name": "Norvasc, Amlodipine Besylate",
        "generic_name": "Amlodipine",
        "indications_and_usage": "Used to treat high blood pressure and chest pain (angina). Belongs to calcium channel blocker class.",
        "warnings": "May cause swelling in feet and ankles. Can interact with grapefruit juice. Monitor blood pressure regularly.",
        "dosage": "Typical starting dose: 5mg once daily. Maximum: 10mg per day. Take at same time each day.",
        "side_effects": "Swelling in feet/ankles, dizziness, headache, flushing, fatigue, stomach upset.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    # Mental Health Medications
    "sertraline": {
        "source": "Local Database",
        "search_term": "sertraline",
        "brand_name": "Zoloft",
        "generic_name": "Sertraline Hydrochloride",
        "indications_and_usage": "Used to treat depression, anxiety disorders, obsessive-compulsive disorder, and post-traumatic stress disorder.",
        "warnings": "May increase risk of suicidal thoughts in young people. Do not stop suddenly. May take 4-6 weeks to work.",
        "dosage": "Typical starting dose: 50mg once daily. Maximum: 200mg per day. Take at same time each day.",
        "side_effects": "Nausea, diarrhea, insomnia, sexual problems, weight changes, increased sweating.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    "fluoxetine": {
        "source": "Local Database",
        "search_term": "fluoxetine",
        "brand_name": "Prozac, Sarafem",
        "generic_name": "Fluoxetine Hydrochloride",
        "indications_and_usage": "Used to treat depression, obsessive-compulsive disorder, bulimia nervosa, and panic disorder.",
        "warnings": "May increase risk of suicidal thoughts in young people. Do not stop suddenly. May take 4-6 weeks to work.",
        "dosage": "Typical starting dose: 20mg once daily. Maximum: 80mg per day. Take in morning to avoid insomnia.",
        "side_effects": "Nausea, headache, insomnia, sexual problems, weight changes, nervousness.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    # Cholesterol Medications
    "atorvastatin": {
        "source": "Local Database",
        "search_term": "atorvastatin",
        "brand_name": "Lipitor",
        "generic_name": "Atorvastatin Calcium",
        "indications_and_usage": "Used to lower cholesterol and reduce risk of heart disease and stroke.",
        "warnings": "Can cause muscle problems. Report muscle pain or weakness. Avoid excessive alcohol. May interact with grapefruit juice.",
        "dosage": "Typical starting dose: 10-20mg once daily. Maximum: 80mg per day. Take at same time each day.",
        "side_effects": "Muscle pain, weakness, stomach upset, headache, liver problems, memory problems.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    },
    
    "simvastatin": {
        "source": "Local Database",
        "search_term": "simvastatin",
        "brand_name": "Zocor",
        "generic_name": "Simvastatin",
        "indications_and_usage": "Used to lower cholesterol and reduce risk of heart disease and stroke.",
        "warnings": "Can cause muscle problems. Report muscle pain or weakness. Avoid excessive alcohol. May interact with grapefruit juice.",
        "dosage": "Typical starting dose: 20-40mg once daily. Maximum: 80mg per day. Take in evening.",
        "side_effects": "Muscle pain, weakness, stomach upset, headache, liver problems, memory problems.",
        "manufacturer": "Various manufacturers",
        "license": "Prescription only",
        "source_url": "https://www.fda.gov/drugs"
    }
}

def search_local_database(drug_name: str) -> dict:
    """Search the local drug database for information"""
    # Normalize the search term
    normalized_name = drug_name.lower().strip()
    
    # Direct match
    if normalized_name in DRUG_DATABASE:
        return DRUG_DATABASE[normalized_name].copy()
    
    # Partial matches
    for key, value in DRUG_DATABASE.items():
        if normalized_name in key or key in normalized_name:
            result = value.copy()
            result["search_term"] = drug_name
            return result
    
    # No match found
    return None

def get_all_drugs() -> list:
    """Get a list of all available drugs in the database"""
    return list(DRUG_DATABASE.keys())

def get_drug_categories() -> dict:
    """Get drugs organized by category"""
    categories = {
        "Pain Relief": ["aspirin", "ibuprofen", "acetaminophen", "paracetamol"],
        "Antibiotics": ["amoxicillin", "penicillin"],
        "Diabetes": ["metformin", "insulin"],
        "Blood Pressure": ["lisinopril", "amlodipine"],
        "Mental Health": ["sertraline", "fluoxetine"],
        "Cholesterol": ["atorvastatin", "simvastatin"]
    }
    return categories

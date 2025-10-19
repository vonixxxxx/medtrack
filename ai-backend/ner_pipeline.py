"""
NER Pipeline for MedTrack
Uses SciSpacy and Med7 for medication entity extraction
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class MedicationNERPipeline:
    """Named Entity Recognition pipeline for medication extraction"""
    
    def __init__(self):
        self.medication_patterns = self._load_medication_patterns()
        self.dosage_patterns = self._load_dosage_patterns()
        self.frequency_patterns = self._load_frequency_patterns()
        
    async def initialize(self):
        """Initialize NER models"""
        try:
            logger.info("🔄 Initializing NER pipeline...")
            
            # Using pattern-based extraction for now
            logger.info("✅ NER pipeline initialized (pattern-based)")
            
        except Exception as e:
            logger.error(f"❌ NER pipeline initialization failed: {e}")
            logger.info("Using pattern-based extraction as fallback")
    
    # SciSpacy and Med7 models removed for now - using pattern-based extraction
    
    def _load_medication_patterns(self) -> List[Dict[str, str]]:
        """Load medication name patterns"""
        return [
            # Common medication patterns
            {"pattern": r"\b(?:acetaminophen|paracetamol|tylenol)\b", "type": "medication", "name": "acetaminophen"},
            {"pattern": r"\b(?:ibuprofen|advil|motrin)\b", "type": "medication", "name": "ibuprofen"},
            {"pattern": r"\b(?:aspirin|asa)\b", "type": "medication", "name": "aspirin"},
            {"pattern": r"\b(?:metformin|glucophage)\b", "type": "medication", "name": "metformin"},
            {"pattern": r"\b(?:lisinopril|prinivil|zestril)\b", "type": "medication", "name": "lisinopril"},
            {"pattern": r"\b(?:atorvastatin|lipitor)\b", "type": "medication", "name": "atorvastatin"},
            {"pattern": r"\b(?:omeprazole|prilosec)\b", "type": "medication", "name": "omeprazole"},
            {"pattern": r"\b(?:amlodipine|norvasc)\b", "type": "medication", "name": "amlodipine"},
            {"pattern": r"\b(?:metoprolol|lopressor|toprol)\b", "type": "medication", "name": "metoprolol"},
            {"pattern": r"\b(?:simvastatin|zocor)\b", "type": "medication", "name": "simvastatin"},
            {"pattern": r"\b(?:losartan|cozaar)\b", "type": "medication", "name": "losartan"},
            {"pattern": r"\b(?:hydrochlorothiazide|hctz|microzide)\b", "type": "medication", "name": "hydrochlorothiazide"},
            {"pattern": r"\b(?:sertraline|zoloft)\b", "type": "medication", "name": "sertraline"},
            {"pattern": r"\b(?:tramadol|ultram)\b", "type": "medication", "name": "tramadol"},
            {"pattern": r"\b(?:gabapentin|neurontin)\b", "type": "medication", "name": "gabapentin"},
            {"pattern": r"\b(?:furosemide|lasix)\b", "type": "medication", "name": "furosemide"},
            {"pattern": r"\b(?:carvedilol|coreg)\b", "type": "medication", "name": "carvedilol"},
            {"pattern": r"\b(?:clopidogrel|plavix)\b", "type": "medication", "name": "clopidogrel"},
            {"pattern": r"\b(?:warfarin|coumadin)\b", "type": "medication", "name": "warfarin"},
            {"pattern": r"\b(?:digoxin|lanoxin)\b", "type": "medication", "name": "digoxin"},
            # Generic patterns
            {"pattern": r"\b[A-Z][a-z]+(?:mycin|cin|pam|zole|sartan|pril|statin|pine|olol|sone|zine|prazole)\b", "type": "medication", "name": "generic_medication"},
        ]
    
    def _load_dosage_patterns(self) -> List[Dict[str, str]]:
        """Load dosage patterns"""
        return [
            {"pattern": r"\b(\d+(?:\.\d+)?)\s*(?:mg|milligrams?)\b", "type": "dosage", "unit": "mg"},
            {"pattern": r"\b(\d+(?:\.\d+)?)\s*(?:g|grams?)\b", "type": "dosage", "unit": "g"},
            {"pattern": r"\b(\d+(?:\.\d+)?)\s*(?:ml|milliliters?)\b", "type": "dosage", "unit": "ml"},
            {"pattern": r"\b(\d+(?:\.\d+)?)\s*(?:mcg|micrograms?)\b", "type": "dosage", "unit": "mcg"},
            {"pattern": r"\b(\d+(?:\.\d+)?)\s*(?:units?|iu)\b", "type": "dosage", "unit": "units"},
            {"pattern": r"\b(\d+(?:\.\d+)?)\s*(?:tablets?|pills?|capsules?)\b", "type": "dosage", "unit": "tablets"},
        ]
    
    def _load_frequency_patterns(self) -> List[Dict[str, str]]:
        """Load frequency patterns"""
        return [
            {"pattern": r"\b(?:once|one)\s+(?:daily|a day|per day)\b", "type": "frequency", "value": "once daily"},
            {"pattern": r"\b(?:twice|two)\s+(?:daily|a day|per day)\b", "type": "frequency", "value": "twice daily"},
            {"pattern": r"\b(?:three times|thrice)\s+(?:daily|a day|per day)\b", "type": "frequency", "value": "three times daily"},
            {"pattern": r"\bevery\s+(\d+)\s+(?:hours?|hrs?)\b", "type": "frequency", "value": "every X hours"},
            {"pattern": r"\bevery\s+(\d+)\s+(?:days?)\b", "type": "frequency", "value": "every X days"},
            {"pattern": r"\b(?:as needed|prn|when needed)\b", "type": "frequency", "value": "as needed"},
            {"pattern": r"\b(?:before|after)\s+(?:meals?|eating)\b", "type": "frequency", "value": "with meals"},
        ]
    
    async def extract_medications(self, text: str) -> Dict[str, Any]:
        """Extract medication entities from text"""
        try:
            entities = {
                "medications": [],
                "dosages": [],
                "frequencies": [],
                "raw_entities": []
            }
            
            # Use pattern-based extraction
            pattern_entities = await self._extract_with_patterns(text)
            
            # Merge results
            entities["medications"].extend(pattern_entities["medications"])
            entities["dosages"].extend(pattern_entities["dosages"])
            entities["frequencies"].extend(pattern_entities["frequencies"])
            
            # Remove duplicates
            entities["medications"] = list(set(entities["medications"]))
            entities["dosages"] = list(set(entities["dosages"]))
            entities["frequencies"] = list(set(entities["frequencies"]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Medication extraction failed: {e}")
            return {"medications": [], "dosages": [], "frequencies": [], "raw_entities": []}
    
    # SciSpacy extraction removed - using pattern-based extraction only
    
    async def _extract_with_patterns(self, text: str) -> Dict[str, Any]:
        """Extract entities using regex patterns"""
        entities = {"medications": [], "dosages": [], "frequencies": [], "raw_entities": []}
        
        text_lower = text.lower()
        
        # Extract medications
        for pattern_info in self.medication_patterns:
            matches = re.finditer(pattern_info["pattern"], text_lower, re.IGNORECASE)
            for match in matches:
                entities["medications"].append(match.group().lower())
                entities["raw_entities"].append({
                    "text": match.group(),
                    "label": "MEDICATION",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.7
                })
        
        # Extract dosages
        for pattern_info in self.dosage_patterns:
            matches = re.finditer(pattern_info["pattern"], text_lower, re.IGNORECASE)
            for match in matches:
                entities["dosages"].append(match.group().lower())
                entities["raw_entities"].append({
                    "text": match.group(),
                    "label": "DOSAGE",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8
                })
        
        # Extract frequencies
        for pattern_info in self.frequency_patterns:
            matches = re.finditer(pattern_info["pattern"], text_lower, re.IGNORECASE)
            for match in matches:
                entities["frequencies"].append(match.group().lower())
                entities["raw_entities"].append({
                    "text": match.group(),
                    "label": "FREQUENCY",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8
                })
        
        return entities
    
    def get_entity_visualization(self, text: str) -> str:
        """Get HTML visualization of extracted entities"""
        return "<p>Entity visualization not available (pattern-based extraction)</p>"
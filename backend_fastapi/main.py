from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from typing import Optional, Dict, Any
import json
import re
from bs4 import BeautifulSoup
from drug_database import search_local_database, get_all_drugs, get_drug_categories

# Create FastAPI app instance
app = FastAPI(
    title="Drug Information API",
    description="API for searching drug information using EMC, NHS, and comprehensive local database",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Multiple API sources for better coverage
EMC_BASE_URL = "https://www.medicines.org.uk/emc"
EMC_SEARCH_URL = "https://www.medicines.org.uk/emc/search"
NHS_BASE_URL = "https://www.nhs.uk/medicines"

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message"""
    return {"message": "Enhanced Drug Information API - Use /drug/{name} to search for drugs"}

@app.get("/drug/{drug_name}")
async def get_drug_info(drug_name: str) -> Dict[str, Any]:
    """
    Get drug information from multiple sources including EMC, NHS, and local database
    
    Args:
        drug_name (str): The name of the drug/active ingredient to search for
        
    Returns:
        Dict[str, Any]: JSON response with comprehensive drug information
    """
    
    # Try multiple sources for better coverage
    drug_info = None
    
    # First try EMC search
    try:
        drug_info = await search_emc(drug_name)
        if drug_info:
            return drug_info
    except Exception as e:
        print(f"EMC search failed: {e}")
    
    # Fallback to NHS search
    try:
        drug_info = await search_nhs(drug_name)
        if drug_info:
            return drug_info
    except Exception as e:
        print(f"NHS search failed: {e}")
    
    # Fallback to OpenFDA (limited but sometimes works)
    try:
        drug_info = await search_openfda(drug_name)
        if drug_info:
            return drug_info
    except Exception as e:
        print(f"OpenFDA search failed: {e}")
    
    # Final fallback to local database (guaranteed to work for common drugs)
    local_result = search_local_database(drug_name)
    if local_result:
        return local_result
    
    # If all sources fail, return a comprehensive error with available drugs
    available_drugs = get_all_drugs()
    raise HTTPException(
        status_code=404, 
        detail=f"No drug information found for '{drug_name}'. Available drugs include: {', '.join(available_drugs[:10])}... Try searching for the generic name or active ingredient."
    )

async def search_emc(drug_name: str) -> Optional[Dict[str, Any]]:
    """Search EMC for drug information"""
    try:
        # EMC search endpoint
        search_url = f"{EMC_SEARCH_URL}?q={drug_name}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url)
            response.raise_for_status()
            
            # Parse the search results
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for drug results
            drug_links = soup.find_all('a', href=re.compile(r'/emc/product/\d+'))
            
            if drug_links:
                # Get the first result
                drug_url = f"{EMC_BASE_URL}{drug_links[0]['href']}"
                
                # Fetch detailed drug information
                drug_response = await client.get(drug_url)
                drug_response.raise_for_status()
                
                drug_soup = BeautifulSoup(drug_response.text, 'html.parser')
                
                # Extract drug information
                drug_info = {
                    "source": "EMC (UK)",
                    "search_term": drug_name,
                    "brand_name": extract_text(drug_soup, '.product-name, h1'),
                    "generic_name": extract_text(drug_soup, '.generic-name, .active-ingredient'),
                    "indications_and_usage": extract_text(drug_soup, '.indications, .therapeutic-indications'),
                    "warnings": extract_text(drug_soup, '.warnings, .precautions, .contraindications'),
                    "dosage": extract_text(drug_soup, '.dosage, .administration'),
                    "side_effects": extract_text(drug_soup, '.side-effects, .adverse-reactions'),
                    "manufacturer": extract_text(drug_soup, '.manufacturer, .company'),
                    "license": extract_text(drug_soup, '.license, .marketing-authorization'),
                    "source_url": drug_url
                }
                
                # Clean up the data
                drug_info = {k: v.strip() if v else "Information not available" for k, v in drug_info.items()}
                
                return drug_info
                
    except Exception as e:
        print(f"EMC search error: {e}")
        return None

async def search_nhs(drug_name: str) -> Optional[Dict[str, Any]]:
    """Search NHS for drug information"""
    try:
        # NHS search endpoint
        search_url = f"{NHS_BASE_URL}/{drug_name.lower().replace(' ', '-')}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract NHS drug information
                drug_info = {
                    "source": "NHS (UK)",
                    "search_term": drug_name,
                    "brand_name": extract_text(soup, 'h1, .medication-title'),
                    "generic_name": extract_text(soup, '.generic-name, .active-ingredient'),
                    "indications_and_usage": extract_text(soup, '.uses, .indications, .what-it-treats'),
                    "warnings": extract_text(soup, '.warnings, .precautions, .when-not-to-use'),
                    "dosage": extract_text(soup, '.dosage, .how-to-take'),
                    "side_effects": extract_text(soup, '.side-effects, .adverse-effects'),
                    "source_url": search_url
                }
                
                # Clean up the data
                drug_info = {k: v.strip() if v else "Information not available" for k, v in drug_info.items()}
                
                return drug_info
                
    except Exception as e:
        print(f"NHS search error: {e}")
        return None

async def search_openfda(drug_name: str) -> Optional[Dict[str, Any]]:
    """Fallback to OpenFDA API (limited coverage)"""
    try:
        openfda_url = "https://api.fda.gov/drug/label.json"
        search_params = {
            "search": f"active_ingredient:{drug_name}",
            "limit": 1
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(openfda_url, params=search_params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("results") and len(data["results"]) > 0:
                drug_info = data["results"][0]
                
                return {
                    "source": "OpenFDA (US)",
                    "search_term": drug_name,
                    "brand_name": extract_openfda_field(drug_info, "openfda.brand_name"),
                    "generic_name": extract_openfda_field(drug_info, "openfda.generic_name"),
                    "indications_and_usage": extract_openfda_field(drug_info, "indications_and_usage"),
                    "warnings": extract_openfda_field(drug_info, "warnings"),
                    "dosage": extract_openfda_field(drug_info, "dosage_and_administration"),
                    "side_effects": extract_openfda_field(drug_info, "adverse_reactions"),
                    "source_url": "https://www.fda.gov/drugs"
                }
                
    except Exception as e:
        print(f"OpenFDA search error: {e}")
        return None

def extract_text(soup: BeautifulSoup, selector: str) -> str:
    """Extract text from HTML using CSS selector"""
    try:
        element = soup.select_one(selector)
        if element:
            return element.get_text(strip=True)
        return ""
    except:
        return ""

def extract_openfda_field(drug_info: Dict, field_path: str) -> str:
    """Extract field from OpenFDA response using dot notation"""
    try:
        keys = field_path.split('.')
        value = drug_info
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return "Information not available"
        
        if isinstance(value, list) and len(value) > 0:
            return value[0]
        elif isinstance(value, str):
            return value
        else:
            return "Information not available"
    except:
        return "Information not available"

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running"""
    return {"status": "healthy", "message": "Enhanced Drug Information API is running"}

@app.get("/drugs")
async def get_available_drugs():
    """Get a list of all available drugs in the local database"""
    return {
        "total_drugs": len(get_all_drugs()),
        "drugs": get_all_drugs(),
        "message": "These drugs are guaranteed to return information from the local database"
    }

@app.get("/drugs/categories")
async def get_drug_categories_endpoint():
    """Get drugs organized by therapeutic category"""
    return {
        "categories": get_drug_categories(),
        "message": "Drugs organized by therapeutic category"
    }

@app.get("/drugs/search/{query}")
async def search_drugs(query: str):
    """Search for drugs by partial name match"""
    available_drugs = get_all_drugs()
    matching_drugs = [drug for drug in available_drugs if query.lower() in drug.lower()]
    
    return {
        "query": query,
        "matches": matching_drugs,
        "total_matches": len(matching_drugs),
        "message": f"Found {len(matching_drugs)} drugs matching '{query}'"
    }

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app using uvicorn server
    # host="0.0.0.0" allows external connections
    # port=8001 to avoid conflicts with existing backend on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8001)

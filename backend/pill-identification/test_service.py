#!/usr/bin/env python3
"""
Simple test script for the Pill Identification API

Usage:
    python test_service.py
"""

import requests
import base64
import json
import sys
import os

API_URL = "http://127.0.0.1:8005"


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_info():
    """Test info endpoint"""
    print("\nTesting /info endpoint...")
    try:
        response = requests.get(f"{API_URL}/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_embed(image_path):
    """Test embedding generation"""
    print(f"\nTesting /embed endpoint with {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            f"{API_URL}/embed",
            json={"image": f"data:image/jpeg;base64,{image_data}"}
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Embedding dimension: {result['dimension']}")
            print(f"First 5 values: {result['embedding'][:5]}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_identify(image_path):
    """Test identification"""
    print(f"\nTesting /identify endpoint with {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        response = requests.post(
            f"{API_URL}/identify",
            json={
                "image": f"data:image/jpeg;base64,{image_data}",
                "k": 5,
                "min_confidence": 0.0
            }
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Number of results: {result['num_results']}")
            if result['top_match']:
                print(f"Top match confidence: {result['top_match']['confidence']:.4f}")
                print(f"Top match metadata: {json.dumps(result['top_match']['metadata'], indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Pill Identification API Test Suite")
    print("=" * 60)
    
    # Test basic endpoints
    health_ok = test_health()
    info_ok = test_info()
    
    if not health_ok:
        print("\n❌ Health check failed. Is the service running?")
        print("Start the service with: uvicorn api.app:app --host 127.0.0.1 --port 8005")
        sys.exit(1)
    
    # Test with image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        embed_ok = test_embed(image_path)
        identify_ok = test_identify(image_path)
        
        if embed_ok and identify_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    else:
        print("\n✅ Basic tests passed!")
        print("\nTo test with an image, run:")
        print(f"  python test_service.py /path/to/pill/image.jpg")


if __name__ == '__main__':
    main()








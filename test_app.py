#!/usr/bin/env python3
"""
Test script to verify Flask app works with probability map changes
"""
import requests
import json
from pathlib import Path

# Test image path
TEST_IMAGE = "/Users/amxr666/Desktop/mangrove-carbon-pipeline/TEST IMAGES/Langkawi1.tif"
FLASK_URL = "http://localhost:5000"

def test_app():
    print("=" * 70)
    print("Testing Mangrove Carbon Detection App with Probability Maps")
    print("=" * 70)
    
    # Test 1: Check app is running
    print("\n[TEST 1] Checking if Flask app is running...")
    try:
        response = requests.get(f"{FLASK_URL}/status", timeout=5)
        status = response.json()
        print(f"✅ App Status: {status['status']}")
        print(f"   Device: {status['device']}")
        print(f"   Model channels: {status['model_in_channels']}")
    except Exception as e:
        print(f"❌ App not running: {e}")
        return False
    
    # Test 2: Upload image
    print("\n[TEST 2] Uploading test image...")
    if not Path(TEST_IMAGE).exists():
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return False
    
    try:
        with open(TEST_IMAGE, 'rb') as f:
            files = {'image': f}
            data = {
                'pixel_size': '0.7',
                'carbon_density': '150.0'
            }
            response = requests.post(
                f"{FLASK_URL}/upload",
                files=files,
                data=data,
                timeout=120
            )
        
        result = response.json()
        
        if not result.get('success'):
            print(f"❌ Upload failed: {result.get('error')}")
            return False
        
        print(f"✅ Upload successful!")
        print(f"   Coverage: {result['coveragePercent']}%")
        print(f"   Area: {result['areaHectares']} ha")
        print(f"   Carbon: {result['carbonTons']} tons")
        print(f"   CO2: {result['carbonCO2']} tons")
        
        if result.get('warning'):
            print(f"   ⚠️ Warning: {result['warning']}")
        
        # Save full result for verification
        results_file = Path("/Users/amxr666/Desktop/mangrove-carbon-pipeline/test_results.json")
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n📄 Full results saved to: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_app()
    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED - App is working correctly!")
    else:
        print("❌ TESTS FAILED - Check the error messages above")
    print("=" * 70)

#!/bin/bash

# Test script for medication validation
# Tests multiple medications to ensure they all work

BASE_URL="http://localhost:4000"
ENDPOINT="/api/medications/validateMedication"

echo "Testing Medication Validation Endpoint"
echo "======================================"
echo ""

test_medication() {
    local med=$1
    echo "Testing: $med"
    response=$(curl -s -X POST "$BASE_URL$ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"$med\"}")
    
    found=$(echo "$response" | grep -o '"found":[^,]*' | cut -d':' -f2)
    name=$(echo "$response" | grep -o '"generic_name":"[^"]*"' | cut -d'"' -f4)
    source=$(echo "$response" | grep -o '"source":"[^"]*"' | cut -d'"' -f4)
    
    if [ "$found" = "true" ]; then
        echo "  ✓ Found: $name (source: $source)"
        return 0
    else
        echo "  ✗ Not found"
        echo "  Response: $response"
        return 1
    fi
}

# Test medications
medications=("acetaminophen" "ibuprofen" "xanax" "metformin" "lisinopril" "tylenol" "advil" "ozempic")

all_passed=true
for med in "${medications[@]}"; do
    if ! test_medication "$med"; then
        all_passed=false
    fi
    echo ""
done

if [ "$all_passed" = true ]; then
    echo "======================================"
    echo "✓ All tests passed!"
    exit 0
else
    echo "======================================"
    echo "✗ Some tests failed"
    exit 1
fi



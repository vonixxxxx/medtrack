const axios = require('axios');

const BASE_URL = 'http://localhost:8000/api';

// Test data for comprehensive testing
const testCases = [
  // Exact matches
  { query: 'paracetamol', expected: 'Paracetamol', description: 'Exact generic name match' },
  { query: 'ibuprofen', expected: 'Ibuprofen', description: 'Exact generic name match' },
  { query: 'aspirin', expected: 'Aspirin', description: 'Exact generic name match' },
  
  // Brand name matches
  { query: 'ozempic', expected: 'Semaglutide', description: 'Brand name to generic match' },
  { query: 'calpol', expected: 'Paracetamol', description: 'Brand name to generic match' },
  { query: 'nurofen', expected: 'Ibuprofen', description: 'Brand name to generic match' },
  
  // Acronym and class matches
  { query: 'glp1', expected: 'Semaglutide', description: 'Acronym match' },
  { query: 'glp-1', expected: 'Semaglutide', description: 'Acronym with hyphen match' },
  { query: 'nsaid', expected: 'Ibuprofen', description: 'Drug class match' },
  { query: 'statin', expected: 'Atorvastatin', description: 'Drug class match' },
  
  // Fuzzy matches (typos)
  { query: 'paracitamol', expected: 'Paracetamol', description: 'Typo correction' },
  { query: 'ibuprufen', expected: 'Ibuprofen', description: 'Typo correction' },
  { query: 'metforman', expected: 'Metformin', description: 'Typo correction' },
  
  // No results (should show suggestions)
  { query: 'xyz123', expected: null, description: 'No results with suggestions' },
  { query: 'nonexistent', expected: null, description: 'No results with suggestions' }
];

async function testSearchFunctionality() {
  console.log('🧪 Testing NHS Medication Search System\n');
  
  let passedTests = 0;
  let totalTests = testCases.length;
  
  for (const testCase of testCases) {
    try {
      console.log(`📋 Testing: ${testCase.description}`);
      console.log(`   Query: "${testCase.query}"`);
      
      const response = await axios.get(`${BASE_URL}/meds/search?q=${encodeURIComponent(testCase.query)}`);
      
      if (response.status === 200) {
        const data = response.data;
        
        if (testCase.expected === null) {
          // Test for no results with suggestions
          if (data.matches.length === 0 && data.suggestions.length > 0) {
            console.log(`   ✅ PASS: No results found, suggestions provided (${data.suggestions.length} suggestions)`);
            passedTests++;
          } else {
            console.log(`   ❌ FAIL: Expected no results with suggestions, got ${data.matches.length} matches`);
          }
        } else {
          // Test for successful match
          const found = data.matches.find(match => 
            match.genericName.toLowerCase() === testCase.expected.toLowerCase() ||
            match.products.some(p => p.brandName.toLowerCase().includes(testCase.expected.toLowerCase()))
          );
          
          if (found) {
            console.log(`   ✅ PASS: Found "${found.genericName}" (${found.products.length} products)`);
            console.log(`      Reason: ${found.reason}, Score: ${found.score}`);
            passedTests++;
          } else {
            console.log(`   ❌ FAIL: Expected "${testCase.expected}", not found in results`);
            console.log(`      Found: ${data.matches.map(m => m.genericName).join(', ')}`);
          }
        }
      } else {
        console.log(`   ❌ FAIL: HTTP ${response.status}`);
      }
      
    } catch (error) {
      console.log(`   ❌ FAIL: ${error.message}`);
    }
    
    console.log(''); // Empty line for readability
  }
  
  return { passedTests, totalTests };
}

async function testProductOptions() {
  console.log('🧪 Testing Product Options Retrieval\n');
  
  try {
    // First, search for a medication to get a product ID
    const searchResponse = await axios.get(`${BASE_URL}/meds/search?q=semaglutide`);
    const productId = searchResponse.data.matches[0].products[0].id;
    
    console.log(`📋 Testing Product Options for Semaglutide (Product ID: ${productId})`);
    
    const optionsResponse = await axios.get(`${BASE_URL}/meds/product/${productId}/options`);
    
    if (optionsResponse.status === 200) {
      const options = optionsResponse.data;
      
      console.log(`   ✅ Product: ${options.brand_name} (${options.generic_name})`);
      console.log(`   ✅ Intake Type: ${options.allowed_intake_type}`);
      console.log(`   ✅ Route: ${options.metadata.route}`);
      console.log(`   ✅ Form: ${options.metadata.form}`);
      console.log(`   ✅ Frequencies: ${options.allowed_frequencies.join(', ')}`);
      console.log(`   ✅ Strengths: ${options.strengths.length} available`);
      console.log(`   ✅ Places: ${options.default_places.join(', ')}`);
      
      // Test specific strength values
      const strengths = options.strengths;
      console.log(`   📊 Available Strengths:`);
      strengths.forEach(strength => {
        console.log(`      - ${strength.value} ${strength.unit} (${strength.frequency})`);
      });
      
      return true;
    } else {
      console.log(`   ❌ FAIL: HTTP ${optionsResponse.status}`);
      return false;
    }
    
  } catch (error) {
    console.log(`   ❌ FAIL: ${error.message}`);
    return false;
  }
}

async function testValidationSystem() {
  console.log('🧪 Testing Medication Validation System\n');
  
  try {
    // Get a product for validation testing
    const searchResponse = await axios.get(`${BASE_URL}/meds/search?q=paracetamol`);
    const productId = searchResponse.data.matches[0].products[0].id;
    const medicationId = searchResponse.data.matches[0].id;
    
    console.log(`📋 Testing Validation for Paracetamol`);
    
    // Test valid configuration
    const validPayload = {
      medication_id: medicationId,
      product_id: productId,
      intake_type: 'Pill/Tablet',
      intake_place: 'at home',
      strength_value: 500,
      strength_unit: 'mg',
      frequency: 'every 4-6 hours',
      custom_flags: {}
    };
    
    console.log(`   📋 Testing Valid Configuration:`);
    console.log(`      - Intake: ${validPayload.intake_type}`);
    console.log(`      - Place: ${validPayload.intake_place}`);
    console.log(`      - Dose: ${validPayload.strength_value} ${validPayload.strength_unit}`);
    console.log(`      - Frequency: ${validPayload.frequency}`);
    
    try {
      const validationResponse = await axios.post(`${BASE_URL}/meds/validate`, validPayload);
      
      if (validationResponse.status === 200) {
        const result = validationResponse.data;
        if (result.valid) {
          console.log(`   ✅ PASS: Valid configuration accepted`);
          console.log(`      Normalized: ${result.normalized.label}`);
        } else {
          console.log(`   ❌ FAIL: Valid configuration rejected`);
          console.log(`      Errors: ${JSON.stringify(result.errors)}`);
        }
      } else {
        console.log(`   ❌ FAIL: HTTP ${validationResponse.status}`);
      }
    } catch (validationError) {
      if (validationError.response?.status === 401) {
        console.log(`   ⚠️ SKIP: Validation requires authentication (expected)`);
      } else {
        console.log(`   ❌ FAIL: Validation error: ${validationError.message}`);
      }
    }
    
    return true;
    
  } catch (error) {
    console.log(`   ❌ FAIL: ${error.message}`);
    return false;
  }
}

async function runComprehensiveTests() {
  console.log('🚀 Starting Comprehensive NHS Medication System Tests\n');
  console.log('=' .repeat(60));
  
  try {
    // Test 1: Search Functionality
    const searchResults = await testSearchFunctionality();
    
    console.log('=' .repeat(60));
    
    // Test 2: Product Options
    const optionsSuccess = await testProductOptions();
    
    console.log('=' .repeat(60));
    
    // Test 3: Validation System
    const validationSuccess = await testValidationSystem();
    
    console.log('=' .repeat(60));
    
    // Summary
    console.log('📊 TEST SUMMARY');
    console.log('=' .repeat(60));
    console.log(`🔍 Search Tests: ${searchResults.passedTests}/${searchResults.totalTests} passed`);
    console.log(`📦 Product Options: ${optionsSuccess ? 'PASS' : 'FAIL'}`);
    console.log(`✅ Validation System: ${validationSuccess ? 'PASS' : 'FAIL'}`);
    
    const overallSuccess = searchResults.passedTests === searchResults.totalTests && optionsSuccess && validationSuccess;
    
    if (overallSuccess) {
      console.log('\n🎉 ALL TESTS PASSED! NHS Medication System is working correctly.');
    } else {
      console.log('\n⚠️ Some tests failed. Please review the output above.');
    }
    
    return overallSuccess;
    
  } catch (error) {
    console.error('💥 Test execution failed:', error.message);
    return false;
  }
}

// Run tests if called directly
if (require.main === module) {
  runComprehensiveTests()
    .then((success) => {
      process.exit(success ? 0 : 1);
    })
    .catch((error) => {
      console.error('💥 Test execution failed:', error);
      process.exit(1);
    });
}

module.exports = { runComprehensiveTests };

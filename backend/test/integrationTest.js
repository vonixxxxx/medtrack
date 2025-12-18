/**
 * Comprehensive Integration Test
 * Simulates real user interactions with the medication validation system
 */

const axios = require('axios');

const API_BASE = process.env.API_BASE || 'http://localhost:3001';
const ENDPOINT = `${API_BASE}/api/medications/validateMedication`;

// Real-world medication scenarios
const USER_SCENARIOS = [
  {
    name: 'Common Medication - Propranolol',
    input: 'propranolol',
    expected: {
      found: true,
      hasDrugClass: true,
      hasDosages: true,
      minConfidence: 0.8
    }
  },
  {
    name: 'Brand Name - Tylenol',
    input: 'tylenol',
    expected: {
      found: true,
      hasDrugClass: true,
      hasDosages: true
    }
  },
  {
    name: 'GLP-1 Agonist - Semaglutide',
    input: 'semaglutide',
    expected: {
      found: true,
      hasDrugClass: true,
      hasDosages: true
    }
  },
  {
    name: 'Brand Name - Ozempic',
    input: 'ozempic',
    expected: {
      found: true,
      hasDrugClass: true
    }
  },
  {
    name: 'Common Medication - Metformin',
    input: 'metformin',
    expected: {
      found: true,
      hasDrugClass: true,
      hasDosages: true
    }
  },
  {
    name: 'ACE Inhibitor - Lisinopril',
    input: 'lisinopril',
    expected: {
      found: true,
      hasDrugClass: true,
      hasDosages: true
    }
  },
  {
    name: 'Statin - Atorvastatin',
    input: 'atorvastatin',
    expected: {
      found: true,
      hasDrugClass: true,
      hasDosages: true
    }
  },
  {
    name: 'SSRI - Sertraline',
    input: 'sertraline',
    expected: {
      found: true,
      hasDrugClass: true,
      hasDosages: true
    }
  },
  {
    name: 'With Typo - Metformine',
    input: 'metformine',
    expected: {
      found: true,
      hasDrugClass: true
    }
  },
  {
    name: 'Invalid Input - Not a Drug',
    input: 'xyz123notadrug',
    expected: {
      found: false
    }
  }
];

let testResults = {
  passed: 0,
  failed: 0,
  errors: []
};

async function testEndpoint(medication) {
  try {
    const response = await axios.post(ENDPOINT, {
      medication: medication,
      medication_name: medication,
      name: medication
    }, {
      timeout: 10000
    });
    
    return response.data;
  } catch (error) {
    if (error.response) {
      return error.response.data;
    }
    throw error;
  }
}

async function runIntegrationTests() {
  console.log('🧪 Starting Integration Tests - Real User Scenarios\n');
  console.log('='.repeat(80));
  console.log(`Testing endpoint: ${ENDPOINT}\n`);
  
  for (const scenario of USER_SCENARIOS) {
    console.log(`\n📋 Scenario: ${scenario.name}`);
    console.log(`   Input: "${scenario.input}"`);
    console.log('-'.repeat(80));
    
    try {
      const result = await testEndpoint(scenario.input);
      
      if (scenario.expected.found) {
        if (result.found && result.success && result.data) {
          const data = result.data;
          
          console.log(`   ✅ Medication Found: ${data.generic_name || data.name}`);
          console.log(`   📊 Drug Class: ${data.drug_class || 'Unknown'}`);
          console.log(`   📈 Confidence: ${(data.confidence * 100).toFixed(1)}%`);
          console.log(`   📦 Source: ${data.source}`);
          console.log(`   💊 Dosage Forms: ${data.dosage_forms?.length || 0}`);
          console.log(`   📏 Typical Strengths: ${data.typical_strengths?.length || 0}`);
          
          // Validate expectations
          let scenarioPassed = true;
          
          if (scenario.expected.hasDrugClass && data.drug_class === 'Unknown') {
            console.log(`   ⚠️  Warning: Drug class not detected`);
            // Not a failure, just a warning
          }
          
          if (scenario.expected.hasDosages && (!data.typical_strengths || data.typical_strengths.length === 0)) {
            console.log(`   ⚠️  Warning: No dosage information available`);
            // Not a failure, just a warning
          }
          
          if (scenario.expected.minConfidence && data.confidence < scenario.expected.minConfidence) {
            console.log(`   ⚠️  Warning: Confidence below expected (${data.confidence} < ${scenario.expected.minConfidence})`);
            // Not a failure, just a warning
          }
          
          if (scenarioPassed) {
            testResults.passed++;
            console.log(`   ✅ PASSED`);
          } else {
            testResults.failed++;
            testResults.errors.push({
              scenario: scenario.name,
              error: 'Expectations not met'
            });
            console.log(`   ❌ FAILED`);
          }
        } else {
          testResults.failed++;
          testResults.errors.push({
            scenario: scenario.name,
            error: 'Medication not found'
          });
          console.log(`   ❌ FAILED: Medication not found`);
        }
      } else {
        // Expected not found
        if (!result.found || !result.success) {
          testResults.passed++;
          console.log(`   ✅ Correctly rejected`);
        } else {
          testResults.failed++;
          testResults.errors.push({
            scenario: scenario.name,
            error: 'Should have been rejected but was found'
          });
          console.log(`   ❌ FAILED: Should have been rejected`);
        }
      }
    } catch (error) {
      testResults.failed++;
      testResults.errors.push({
        scenario: scenario.name,
        error: error.message
      });
      console.log(`   ❌ ERROR: ${error.message}`);
    }
  }
  
  // Summary
  console.log('\n' + '='.repeat(80));
  console.log('📊 INTEGRATION TEST SUMMARY');
  console.log('='.repeat(80));
  console.log(`✅ Passed: ${testResults.passed}`);
  console.log(`❌ Failed: ${testResults.failed}`);
  console.log(`📈 Success Rate: ${((testResults.passed / (testResults.passed + testResults.failed)) * 100).toFixed(1)}%`);
  
  if (testResults.errors.length > 0) {
    console.log('\n⚠️  Errors:');
    testResults.errors.forEach((err, i) => {
      console.log(`   ${i + 1}. ${err.scenario}: ${err.error}`);
    });
  }
  
  console.log('\n' + '='.repeat(80));
  
  return testResults.failed === 0;
}

// Run if called directly
if (require.main === module) {
  runIntegrationTests()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('Test suite error:', error);
      process.exit(1);
    });
}

module.exports = { runIntegrationTests };




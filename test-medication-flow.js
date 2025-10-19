#!/usr/bin/env node

const axios = require('axios');

async function testMedicationFlow() {
  console.log('🧪 Testing MedTrack Medication Flow...\n');

  try {
    // Test 1: Backend API
    console.log('1. Testing Backend API...');
    const response = await axios.post('http://localhost:4000/api/medications/validateMedication', {
      medication_name: 'panadol'
    });
    
    if (response.data.success) {
      console.log('✅ Backend API working');
      console.log(`   Found: ${response.data.data.generic_name}`);
      console.log(`   Confidence: ${response.data.data.confidence}`);
    } else {
      console.log('❌ Backend API failed');
      return;
    }

    // Test 2: Frontend accessibility
    console.log('\n2. Testing Frontend...');
    try {
      const frontendResponse = await axios.get('http://localhost:3000', {
        headers: { 'User-Agent': 'Mozilla/5.0' },
        timeout: 5000
      });
      if (frontendResponse.status === 200 && frontendResponse.data.includes('MedTrack')) {
        console.log('✅ Frontend accessible');
      } else {
        console.log('❌ Frontend not accessible');
        return;
      }
    } catch (error) {
      console.log('❌ Frontend not accessible:', error.message);
      return;
    }

    // Test 3: Component loading
    console.log('\n3. Testing Component Loading...');
    const componentResponse = await axios.get('http://localhost:3000/src/components/ProductionMedicationChat.jsx');
    if (componentResponse.status === 200) {
      console.log('✅ ProductionMedicationChat component loads');
    } else {
      console.log('❌ Component failed to load');
      return;
    }

    console.log('\n🎉 All tests passed! The medication flow is ready.');
    console.log('\n📋 Next steps:');
    console.log('   1. Open http://localhost:3000 in your browser');
    console.log('   2. Click "Add New Medication"');
    console.log('   3. Try adding medications like "panadol", "adderall", "metformin"');
    console.log('   4. Test the conversational flow and metric selection');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('   Status:', error.response.status);
      console.error('   Data:', error.response.data);
    }
  }
}

testMedicationFlow();
#!/usr/bin/env node

const axios = require('axios');

const NODE_BACKEND_URL = 'http://localhost:4000';
const PYTHON_BACKEND_URL = 'http://localhost:5002';
const FRONTEND_URL = 'http://localhost:3000';

async function testAISystem() {
  console.log('🧪 MedTrack AI Stack Comprehensive Test\n');
  
  let passedTests = 0;
  let totalTests = 0;
  
  // Test 1: Python AI Backend Health
  totalTests++;
  try {
    console.log('1. Testing Python AI Backend Health...');
    const response = await axios.get(`${PYTHON_BACKEND_URL}/health`);
    if (response.data.status === 'healthy' && response.data.ai_available) {
      console.log('✅ Python AI Backend is healthy');
      passedTests++;
    } else {
      console.log('❌ Python AI Backend is not healthy');
    }
  } catch (error) {
    console.log('❌ Python AI Backend health check failed:', error.message);
  }
  
  // Test 2: Python AI Backend Status
  totalTests++;
  try {
    console.log('\n2. Testing Python AI Backend Status...');
    const response = await axios.get(`${PYTHON_BACKEND_URL}/api/ai/status`);
    if (response.data.available) {
      console.log('✅ Python AI Backend status: Available');
      console.log(`   Models: ${response.data.models.join(', ')}`);
      passedTests++;
    } else {
      console.log('❌ Python AI Backend status: Not available');
    }
  } catch (error) {
    console.log('❌ Python AI Backend status check failed:', error.message);
  }
  
  // Test 3: Node.js Backend AI Status
  totalTests++;
  try {
    console.log('\n3. Testing Node.js Backend AI Status...');
    const response = await axios.get(`${NODE_BACKEND_URL}/api/ai/status`);
    if (response.data.available) {
      console.log('✅ Node.js Backend AI status: Available');
      console.log(`   Message: ${response.data.message}`);
      passedTests++;
    } else {
      console.log('❌ Node.js Backend AI status: Not available');
    }
  } catch (error) {
    console.log('❌ Node.js Backend AI status check failed:', error.message);
  }
  
  // Test 4: AI Chat Functionality
  totalTests++;
  try {
    console.log('\n4. Testing AI Chat Functionality...');
    const response = await axios.post(`${NODE_BACKEND_URL}/api/ai/chat`, {
      message: 'I need help with my metformin medication',
      context: {},
      type: 'medication'
    });
    if (response.data.response && response.data.confidence > 0) {
      console.log('✅ AI Chat is working');
      console.log(`   Response: ${response.data.response.substring(0, 100)}...`);
      passedTests++;
    } else {
      console.log('❌ AI Chat is not working properly');
    }
  } catch (error) {
    console.log('❌ AI Chat test failed:', error.message);
  }
  
  // Test 5: Medication Validation
  totalTests++;
  try {
    console.log('\n5. Testing Medication Validation...');
    const response = await axios.post(`${NODE_BACKEND_URL}/api/ai/validate`, {
      medication: 'metformin',
      dosage: '500mg',
      frequency: 'twice daily',
      user_context: {}
    });
    if (response.data.isValid !== undefined && response.data.confidence > 0) {
      console.log('✅ Medication Validation is working');
      console.log(`   Valid: ${response.data.isValid}, Confidence: ${response.data.confidence}`);
      passedTests++;
    } else {
      console.log('❌ Medication Validation is not working properly');
    }
  } catch (error) {
    console.log('❌ Medication Validation test failed:', error.message);
  }
  
  // Test 6: Medication Search
  totalTests++;
  try {
    console.log('\n6. Testing Medication Search...');
    const response = await axios.post(`${NODE_BACKEND_URL}/api/ai/search-med`, {
      query: 'metformin',
      limit: 5,
      min_confidence: 0.5
    });
    if (response.data.results && response.data.results.length > 0) {
      console.log('✅ Medication Search is working');
      console.log(`   Found ${response.data.results.length} results`);
      passedTests++;
    } else {
      console.log('❌ Medication Search is not working properly');
    }
  } catch (error) {
    console.log('❌ Medication Search test failed:', error.message);
  }
  
  // Test 7: Health Report Generation
  totalTests++;
  try {
    console.log('\n7. Testing Health Report Generation...');
    const response = await axios.post(`${NODE_BACKEND_URL}/api/ai/health-report`, {
      user_data: {
        medications: ['metformin', 'lisinopril'],
        metrics: { blood_sugar: 120, systolic: 130, diastolic: 85 },
        adherence: 95
      }
    });
    if (response.data.adherence && response.data.insights) {
      console.log('✅ Health Report Generation is working');
      console.log(`   Adherence: ${response.data.adherence}, Trend: ${response.data.trend}`);
      passedTests++;
    } else {
      console.log('❌ Health Report Generation is not working properly');
    }
  } catch (error) {
    console.log('❌ Health Report Generation test failed:', error.message);
  }
  
  // Test 8: Frontend Accessibility
  totalTests++;
  try {
    console.log('\n8. Testing Frontend Accessibility...');
    const response = await axios.get(`${FRONTEND_URL}`, { 
      timeout: 5000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
      }
    });
    if (response.status === 200 && response.data.includes('MedTrack')) {
      console.log('✅ Frontend is accessible');
      passedTests++;
    } else {
      console.log('❌ Frontend is not accessible');
    }
  } catch (error) {
    console.log('❌ Frontend accessibility test failed:', error.message);
  }
  
  // Test 9: Concurrent User Simulation
  totalTests++;
  try {
    console.log('\n9. Testing Concurrent User Support...');
    const promises = [];
    for (let i = 0; i < 5; i++) {
      promises.push(
        axios.post(`${NODE_BACKEND_URL}/api/ai/chat`, {
          message: `Test message ${i + 1}`,
          context: {},
          type: 'general'
        })
      );
    }
    
    const results = await Promise.all(promises);
    const successCount = results.filter(r => r.data.response).length;
    
    if (successCount === 5) {
      console.log('✅ Concurrent user support is working');
      passedTests++;
    } else {
      console.log(`❌ Concurrent user support failed: ${successCount}/5 requests succeeded`);
    }
  } catch (error) {
    console.log('❌ Concurrent user test failed:', error.message);
  }
  
  // Test 10: Error Handling
  totalTests++;
  try {
    console.log('\n10. Testing Error Handling...');
    const response = await axios.post(`${NODE_BACKEND_URL}/api/ai/chat`, {
      message: '',
      context: {},
      type: 'general'
    });
    console.log('❌ Error handling test failed: Should have returned error for empty message');
  } catch (error) {
    if (error.response && error.response.status >= 400) {
      console.log('✅ Error handling is working (properly rejected invalid request)');
      passedTests++;
    } else {
      console.log('❌ Error handling test failed:', error.message);
    }
  }
  
  // Summary
  console.log('\n📊 Test Results Summary:');
  console.log(`Passed: ${passedTests}/${totalTests} tests`);
  console.log(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
  
  if (passedTests === totalTests) {
    console.log('\n🎉 All tests passed! MedTrack AI Stack is fully functional.');
    console.log('\n✅ Features Working:');
    console.log('   • Python AI Backend (FastAPI)');
    console.log('   • Node.js Backend Integration');
    console.log('   • AI Chat with medication context');
    console.log('   • Medication validation and search');
    console.log('   • Health report generation');
    console.log('   • Frontend accessibility');
    console.log('   • Concurrent user support');
    console.log('   • Error handling');
    
    console.log('\n🚀 MedTrack AI Stack is ready for production use!');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some tests failed. Please check the errors above.');
    process.exit(1);
  }
}

testAISystem().catch(error => {
  console.error('Test suite failed:', error);
  process.exit(1);
});
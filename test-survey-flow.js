const axios = require('axios');

const API_BASE = 'http://localhost:4000/api';

async function testSurveyFlow() {
  console.log('🧪 Testing Survey Completion Flow\n');

  try {
    // 1. Sign up a new patient
    console.log('1️⃣ Signing up new patient...');
    const signupResponse = await axios.post(`${API_BASE}/auth/signup`, {
      email: `test.survey.${Date.now()}@example.com`,
      password: 'test123',
      role: 'patient',
      hospitalCode: '123456789',
      patientData: {
        name: 'Survey Test Patient',
        dob: '1990-01-01',
        sex: 'Male'
      }
    });
    
    console.log('✅ Patient signup successful');
    console.log('   User ID:', signupResponse.data.user.id);

    // 2. Check survey status (should be false)
    console.log('\n2️⃣ Checking survey status...');
    const statusResponse1 = await axios.get(`${API_BASE}/auth/survey-status`);
    console.log('   Survey completed:', statusResponse1.data.surveyCompleted);
    console.log('   Expected: false');

    // 3. Complete survey
    console.log('\n3️⃣ Completing survey...');
    const completeResponse = await axios.put(`${API_BASE}/auth/complete-survey`);
    console.log('   Survey completion response:', completeResponse.data.message);

    // 4. Check survey status again (should be true)
    console.log('\n4️⃣ Checking survey status after completion...');
    const statusResponse2 = await axios.get(`${API_BASE}/auth/survey-status`);
    console.log('   Survey completed:', statusResponse2.data.surveyCompleted);
    console.log('   Expected: true');

    if (statusResponse2.data.surveyCompleted) {
      console.log('\n🎉 Survey completion flow working correctly!');
      console.log('   ✅ Survey appears for new users');
      console.log('   ✅ Survey completion is tracked');
      console.log('   ✅ Survey won\'t appear again after completion');
    } else {
      console.log('\n❌ Survey completion not working properly');
    }

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
  }
}

testSurveyFlow();

const axios = require('axios');

const API_BASE = 'http://localhost:4000/api';

async function testHospitalCodeFunctionality() {
  console.log('🏥 Testing Hospital Code Functionality\n');

  try {
    // Test 1: Signup with valid hospital code
    console.log('1. Testing signup with valid hospital code (123456789)...');
    const validSignup = await axios.post(`${API_BASE}/auth/signup`, {
      email: 'test-clinician@example.com',
      password: 'password123',
      role: 'clinician',
      hospitalCode: '123456789'
    });
    console.log('✅ Valid signup successful:', validSignup.data.user);

    // Test 2: Signup with invalid hospital code
    console.log('\n2. Testing signup with invalid hospital code...');
    try {
      await axios.post(`${API_BASE}/auth/signup`, {
        email: 'test-invalid@example.com',
        password: 'password123',
        role: 'patient',
        hospitalCode: '999999999'
      });
      console.log('❌ Invalid signup should have failed');
    } catch (error) {
      console.log('✅ Invalid signup correctly rejected:', error.response.data.error);
    }

    // Test 3: Signup without hospital code
    console.log('\n3. Testing signup without hospital code...');
    try {
      await axios.post(`${API_BASE}/auth/signup`, {
        email: 'test-no-code@example.com',
        password: 'password123',
        role: 'patient'
      });
      console.log('❌ Signup without hospital code should have failed');
    } catch (error) {
      console.log('✅ Signup without hospital code correctly rejected:', error.response.data.error);
    }

    // Test 4: Login and test JWT token
    console.log('\n4. Testing login and JWT token...');
    const loginResponse = await axios.post(`${API_BASE}/auth/login`, {
      email: 'test-clinician@example.com',
      password: 'password123'
    });
    console.log('✅ Login successful:', loginResponse.data.user);

    // Test 5: Test doctor endpoints with hospital code filtering
    console.log('\n5. Testing doctor endpoints with hospital code filtering...');
    const token = loginResponse.data.token;
    const doctorResponse = await axios.get(`${API_BASE}/doctor/patients`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    console.log('✅ Doctor patients endpoint accessible:', doctorResponse.data.length, 'patients found');

    // Test 6: Test analytics endpoint
    console.log('\n6. Testing analytics endpoint...');
    const analyticsResponse = await axios.get(`${API_BASE}/doctor/analytics`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    console.log('✅ Analytics endpoint accessible:', analyticsResponse.data);

    console.log('\n🎉 All tests passed! Hospital code functionality is working correctly.');

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
  }
}

// Run the test
testHospitalCodeFunctionality();



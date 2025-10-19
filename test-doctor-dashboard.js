const axios = require('axios');

const API_BASE = 'http://localhost:4000/api';

async function testDoctorDashboard() {
  console.log('🧪 Testing Doctor Dashboard End-to-End Flow...\n');

  try {
    const timestamp = Date.now();
    
    // 1. Test clinician signup
    console.log('1. Testing clinician signup...');
    const clinicianSignup = await axios.post(`${API_BASE}/auth/signup`, {
      email: `test-clinician-${timestamp}@example.com`,
      password: 'password123',
      role: 'clinician',
      hospitalCode: '123456789'
    });
    console.log('✅ Clinician signup successful:', clinicianSignup.data.user.role);

    // 2. Test patient signup
    console.log('\n2. Testing patient signup...');
    const patientSignup = await axios.post(`${API_BASE}/auth/signup`, {
      email: `test-patient-${timestamp}@example.com`,
      password: 'password123',
      role: 'patient',
      hospitalCode: '123456789'
    });
    console.log('✅ Patient signup successful:', patientSignup.data.user.role);

    // 3. Test clinician login
    console.log('\n3. Testing clinician login...');
    const clinicianLogin = await axios.post(`${API_BASE}/auth/login`, {
      email: `test-clinician-${timestamp}@example.com`,
      password: 'password123'
    });
    const clinicianToken = clinicianLogin.data.token;
    console.log('✅ Clinician login successful');

    // 4. Test getting patients (should be empty initially)
    console.log('\n4. Testing get patients...');
    const patientsResponse = await axios.get(`${API_BASE}/doctor/patients`, {
      headers: { Authorization: `Bearer ${clinicianToken}` }
    });
    console.log('✅ Get patients successful, count:', patientsResponse.data.length);

    // 5. Test medical history parsing
    console.log('\n5. Testing medical history parsing...');
    const parseResponse = await axios.post(`${API_BASE}/doctor/parse-history`, {
      medicalNotes: 'Patient has type 2 diabetes, hypertension, and high cholesterol. Also shows signs of obesity.'
    }, {
      headers: { Authorization: `Bearer ${clinicianToken}` }
    });
    console.log('✅ Medical history parsing successful, conditions found:', parseResponse.data.conditions.length);

    // 6. Test HbA1c adjustment calculation
    console.log('\n6. Testing HbA1c adjustment calculation...');
    const hba1cResponse = await axios.post(`${API_BASE}/doctor/hba1c-adjust`, {
      measuredHbA1cPercent: 7.5,
      weightKg: 80,
      medications: {
        metformin: 1000,
        insulin: 20
      }
    }, {
      headers: { Authorization: `Bearer ${clinicianToken}` }
    });
    console.log('✅ HbA1c adjustment calculation successful:', hba1cResponse.data);

    console.log('\n🎉 All tests passed! Doctor Dashboard is fully functional.');

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
    process.exit(1);
  }
}

testDoctorDashboard();

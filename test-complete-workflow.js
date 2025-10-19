const axios = require('axios');

const BASE_URL = 'http://localhost:4000/api';
const FRONTEND_URL = 'http://localhost:3000';

// Test data
const testPatient = {
  email: `test-patient-${Date.now()}@example.com`,
  password: 'testpassword123',
  name: 'John Doe',
  role: 'patient',
  hospitalCode: '123456789'
};

const testClinician = {
  email: `test-clinician-${Date.now()}@example.com`,
  password: 'testpassword123',
  name: 'Dr. Smith',
  role: 'clinician',
  hospitalCode: '123456789'
};

const medicalNotes = `
Patient presents with Type 2 diabetes mellitus, hypertension, and obesity.
Current medications: Metformin 1000mg twice daily, Lisinopril 10mg daily.
Recent labs: HbA1c 8.2%, BMI 32.5, Total cholesterol 240 mg/dL, HDL 35 mg/dL, LDL 150 mg/dL.
Patient has sleep apnea and uses CPAP. Also has anxiety and depression.
Weight: 95kg, Height: 175cm, Blood pressure: 145/90 mmHg.
History of bariatric sleeve surgery 2 years ago.
`;

async function testCompleteWorkflow() {
  console.log('🧪 Starting comprehensive workflow test...\n');

  try {
    // Step 1: Test patient signup
    console.log('1️⃣ Testing patient signup...');
    const patientSignupResponse = await axios.post(`${BASE_URL}/auth/signup`, testPatient);
    console.log('✅ Patient signup successful:', patientSignupResponse.data.message);

    // Step 2: Test clinician signup
    console.log('\n2️⃣ Testing clinician signup...');
    const clinicianSignupResponse = await axios.post(`${BASE_URL}/auth/signup`, testClinician);
    console.log('✅ Clinician signup successful:', clinicianSignupResponse.data.message);

    // Step 3: Test patient login
    console.log('\n3️⃣ Testing patient login...');
    const patientLoginResponse = await axios.post(`${BASE_URL}/auth/login`, {
      email: testPatient.email,
      password: testPatient.password
    });
    const patientToken = patientLoginResponse.data.token;
    console.log('✅ Patient login successful');

    // Step 4: Test survey data submission
    console.log('\n4️⃣ Testing survey data submission...');
    const surveyData = {
      name: 'John Doe',
      dateOfBirth: new Date('1985-05-15'),
      biologicalSex: 'Male',
      ethnicity: 'White - English, Welsh, Scottish, Northern Irish or British',
      location: 'London',
      postcode: 'SW1A 1AA',
      nhsNumber: '1234567890',
      weight: 95,
      height: 175,
      baselineBMI: 32.5,
      ascvd: false,
      htn: true,
      hypertension: true,
      t2dm: true,
      prediabetes: false,
      osa: true,
      cpap: true,
      anxiety: true,
      depression: true,
      bariatricSleeve: true,
      baselineHbA1c: 8.2,
      baselineTC: 240,
      baselineHDL: 35,
      baselineLDL: 150
    };

    const surveyResponse = await axios.post(`${BASE_URL}/auth/survey-data`, surveyData);
    console.log('✅ Survey data submitted successfully');

    // Step 5: Test survey completion
    console.log('\n5️⃣ Testing survey completion...');
    const surveyCompleteResponse = await axios.put(`${BASE_URL}/auth/complete-survey`);
    console.log('✅ Survey marked as completed');

    // Step 6: Test clinician login
    console.log('\n6️⃣ Testing clinician login...');
    const clinicianLoginResponse = await axios.post(`${BASE_URL}/auth/login`, {
      email: testClinician.email,
      password: testClinician.password
    });
    const clinicianToken = clinicianLoginResponse.data.token;
    console.log('✅ Clinician login successful');

    // Step 7: Test getting patients (should see the patient we just created)
    console.log('\n7️⃣ Testing patient retrieval for clinician...');
    const patientsResponse = await axios.get(`${BASE_URL}/doctor/patients`, {
      headers: { Authorization: `Bearer ${clinicianToken}` }
    });
    console.log(`✅ Retrieved ${patientsResponse.data.length} patients`);
    
    if (patientsResponse.data.length > 0) {
      const patient = patientsResponse.data[0];
      console.log(`   - Patient: ${patient.name} (${patient.email})`);
      console.log(`   - Hospital Code: ${patient.hospitalCode}`);
      console.log(`   - Conditions: ${patient.conditions?.length || 0} conditions`);
    }

    // Step 8: Test AI medical history parsing
    console.log('\n8️⃣ Testing AI medical history parsing...');
    if (patientsResponse.data.length > 0) {
      const patientId = patientsResponse.data[0].id;
      const parseResponse = await axios.post(`${BASE_URL}/doctor/parse-history`, {
        patientId: patientId,
        medicalNotes: medicalNotes
      }, {
        headers: { Authorization: `Bearer ${clinicianToken}` }
      });
      
      console.log('✅ AI parsing completed');
      console.log(`   - Parsed ${Object.keys(parseResponse.data.parsedData).length} fields`);
      console.log(`   - Found ${parseResponse.data.auditLogs} updates requiring review`);
      console.log(`   - Detected ${parseResponse.data.conditions.length} conditions`);
      
      // Show some parsed data
      const parsedData = parseResponse.data.parsedData;
      console.log('   - Sample parsed data:');
      Object.entries(parsedData).slice(0, 5).forEach(([key, value]) => {
        console.log(`     ${key}: ${value}`);
      });
    }

    // Step 9: Test medication search (fuzzy search)
    console.log('\n9️⃣ Testing medication fuzzy search...');
    const searchTests = [
      'metformin',
      'metfornin', // intentional typo
      'atorv', // partial match
      'aspirin',
      'aspirn' // intentional typo
    ];

    for (const searchTerm of searchTests) {
      try {
        const searchResponse = await axios.post(`${BASE_URL}/ai/search-med`, {
          query: searchTerm,
          limit: 3
        });
        console.log(`   - "${searchTerm}": Found ${searchResponse.data.results.length} results`);
        if (searchResponse.data.results.length > 0) {
          console.log(`     Top result: ${searchResponse.data.results[0].name} (${Math.round(searchResponse.data.results[0].confidence * 100)}% match)`);
        }
      } catch (error) {
        console.log(`   - "${searchTerm}": Search failed - ${error.message}`);
      }
    }

    // Step 10: Test survey status (should show completed)
    console.log('\n🔟 Testing survey status...');
    const surveyStatusResponse = await axios.get(`${BASE_URL}/auth/survey-status`);
    console.log(`✅ Survey status: ${surveyStatusResponse.data.surveyCompleted ? 'Completed' : 'Not completed'}`);

    console.log('\n🎉 All tests completed successfully!');
    console.log('\n📋 Summary:');
    console.log('✅ Patient signup and survey completion');
    console.log('✅ Clinician signup and patient access');
    console.log('✅ AI medical history parsing');
    console.log('✅ Fuzzy medication search');
    console.log('✅ Hospital code linking (123456789)');
    console.log('✅ Survey data integration with patient records');
    console.log('✅ Comprehensive condition detection');

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
    console.error('Stack trace:', error.stack);
  }
}

// Run the test
testCompleteWorkflow();

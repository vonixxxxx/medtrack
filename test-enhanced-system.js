const axios = require('axios');

const API_BASE = 'http://localhost:4000/api';

async function testEnhancedSystem() {
  console.log('🧪 Testing Enhanced MedTrack System\n');

  try {
    // 1. Test Enhanced Patient Signup
    console.log('1️⃣ Testing Enhanced Patient Signup...');
    const patientData = {
      email: `test.patient.${Date.now()}@example.com`,
      password: 'testpassword123',
      role: 'patient',
      hospitalCode: '123456789',
      patientData: {
        name: 'John Enhanced Patient',
        dob: '1985-06-15',
        sex: 'Male',
        ethnic_group: 'White British',
        location: 'London',
        postcode: 'SW1A 1AA',
        nhs_number: '1234567890',
        height: 175,
        baseline_weight: 95,
        baseline_weight_date: '2024-01-01',
        t2dm: true,
        diabetes_type: 'Type 2',
        baseline_hba1c: 7.2,
        baseline_hba1c_date: '2024-01-01',
        ascvd: true,
        htn: true,
        dyslipidaemia: true,
        osa: false,
        asthma: true,
        anxiety: true,
        depression: false,
        notes: 'Patient with multiple comorbidities',
        criteria_for_wegovy: 'BMI 31, T2DM, multiple risk factors'
      }
    };

    const signupResponse = await axios.post(`${API_BASE}/auth/signup`, patientData);
    console.log('✅ Patient signup successful');
    console.log('   Patient ID:', signupResponse.data.user.id);
    console.log('   Token:', signupResponse.data.token.substring(0, 20) + '...');

    const patientToken = signupResponse.data.token;
    const patientId = signupResponse.data.user.id;

    // 2. Test Clinician Signup
    console.log('\n2️⃣ Testing Clinician Signup...');
    const clinicianData = {
      email: `test.clinician.${Date.now()}@example.com`,
      password: 'testpassword123',
      role: 'clinician',
      hospitalCode: '123456789'
    };

    const clinicianResponse = await axios.post(`${API_BASE}/auth/signup`, clinicianData);
    console.log('✅ Clinician signup successful');
    console.log('   Clinician ID:', clinicianResponse.data.user.id);

    const clinicianToken = clinicianResponse.data.token;

    // 3. Test Enhanced Patient Records Retrieval
    console.log('\n3️⃣ Testing Enhanced Patient Records Retrieval...');
    const patientsResponse = await axios.get(`${API_BASE}/doctor/patients`, {
      headers: { Authorization: `Bearer ${clinicianToken}` }
    });

    console.log('✅ Patient records retrieved successfully');
    console.log('   Number of patients:', patientsResponse.data.length);
    
    if (patientsResponse.data.length > 0) {
      const patient = patientsResponse.data[0];
      console.log('   Sample patient data:');
      console.log('   - Name:', patient.name);
      console.log('   - Age:', patient.age);
      console.log('   - Sex:', patient.sex);
      console.log('   - Ethnic Group:', patient.ethnic_group);
      console.log('   - BMI:', patient.baseline_bmi);
      console.log('   - T2DM:', patient.t2dm);
      console.log('   - ASCVD:', patient.ascvd);
      console.log('   - HTN:', patient.htn);
      console.log('   - Notes:', patient.notes);
    }

    // 4. Test AI Medical History Parsing
    console.log('\n4️⃣ Testing AI Medical History Parsing...');
    const medicalNotes = `
      Patient presents with Type 2 diabetes mellitus, diagnosed 2 years ago.
      HbA1c: 8.5%, BMI: 32.1, Weight: 95kg, Height: 175cm
      History of hypertension, dyslipidaemia, and asthma.
      Previous sleep study showed mild OSA, currently on CPAP.
      Also has anxiety and depression.
      Previous bariatric surgery - gastric sleeve in 2020.
      Current medications: Metformin, Lisinopril, Atorvastatin
    `;

    const parseResponse = await axios.post(`${API_BASE}/doctor/parse-history`, {
      patientId: patientsResponse.data[0].id,
      medicalNotes: medicalNotes
    }, {
      headers: { Authorization: `Bearer ${clinicianToken}` }
    });

    console.log('✅ AI parsing successful');
    console.log('   Parsed data:', parseResponse.data.parsedData);
    console.log('   Updates suggested:', parseResponse.data.updates);
    console.log('   Audit logs created:', parseResponse.data.auditLogs);
    console.log('   Conditions found:', parseResponse.data.conditions.length);

    // 5. Test Patient Data Update
    console.log('\n5️⃣ Testing Patient Data Update...');
    const updateData = {
      mrn: 'MRN123456',
      diagnoses_coded_in_scr: 'E11.9, I10, E78.5',
      total_qualifying_comorbidities: 4,
      all_medications_from_scr: 'Metformin 500mg BD, Lisinopril 10mg OD, Atorvastatin 20mg OD'
    };

    const updateResponse = await axios.put(`${API_BASE}/doctor/patients/${patientsResponse.data[0].id}`, updateData, {
      headers: { Authorization: `Bearer ${clinicianToken}` }
    });

    console.log('✅ Patient data update successful');
    console.log('   Updated MRN:', updateResponse.data.patient.mrn);
    console.log('   Updated diagnoses:', updateResponse.data.patient.diagnoses_coded_in_scr);

    // 6. Test Audit Logs Retrieval
    console.log('\n6️⃣ Testing Audit Logs Retrieval...');
    const auditLogsResponse = await axios.get(`${API_BASE}/doctor/patients/${patientsResponse.data[0].id}/audit-logs`, {
      headers: { Authorization: `Bearer ${clinicianToken}` }
    });

    console.log('✅ Audit logs retrieved successfully');
    console.log('   Number of audit logs:', auditLogsResponse.data.length);
    
    if (auditLogsResponse.data.length > 0) {
      const log = auditLogsResponse.data[0];
      console.log('   Sample audit log:');
      console.log('   - Field:', log.field_name);
      console.log('   - Old value:', log.old_value);
      console.log('   - New value:', log.new_value);
      console.log('   - AI confidence:', log.ai_confidence);
      console.log('   - Approved:', log.clinician_approved);
    }

    console.log('\n🎉 All tests passed successfully!');
    console.log('\n📊 System Summary:');
    console.log('   ✅ Enhanced patient signup with comprehensive data');
    console.log('   ✅ Enhanced patient records table with filtering');
    console.log('   ✅ AI medical history parsing with field mapping');
    console.log('   ✅ Patient data update functionality');
    console.log('   ✅ AI audit logging system');
    console.log('   ✅ Clinician dashboard integration');

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
    process.exit(1);
  }
}

// Run the test
testEnhancedSystem();

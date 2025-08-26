import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { User, Heart, Activity, Droplets, Scale, FileText, Download, Eye, EyeOff } from 'lucide-react';

const MedTrackDashboard = ({ userData }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [expandedSections, setExpandedSections] = useState({});

  const tabs = [
    { id: 'overview', name: 'Overview' },
    { id: 'demographics', name: 'Demographics' },
    { id: 'medications', name: 'Medications' },
    { id: 'metrics', name: 'Metrics' }
  ];

  const calculateAge = (dateOfBirth) => {
    if (!dateOfBirth) return null;
    const today = new Date();
    const birthDate = new Date(dateOfBirth);
    const age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();

    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      return age - 1;
    }

    const dayDiff = today.getDate() - birthDate.getDate();
    const decimalAge = age + (monthDiff * 30 + dayDiff) / 365.25;
    return parseFloat(decimalAge.toFixed(1));
  };

  const getBMICategory = (bmi) => {
    if (bmi < 18.5) return { category: 'Underweight', color: 'text-blue-600', bgColor: 'bg-blue-50' };
    if (bmi < 25) return { category: 'Normal weight', color: 'text-green-600', bgColor: 'bg-green-50' };
    if (bmi < 30) return { category: 'Overweight', color: 'text-yellow-600', bgColor: 'bg-yellow-50' };
    if (bmi < 35) return { category: 'Obese Class I', color: 'text-orange-600', bgColor: 'bg-orange-50' };
    if (bmi < 40) return { category: 'Obese Class II', color: 'text-red-600', bgColor: 'bg-red-50' };
    return { category: 'Obese Class III', color: 'text-red-800', bgColor: 'bg-red-100' };
  };

  const getWHRCategory = (whr, biologicalSex) => {
    if (biologicalSex === 'male') {
      if (whr < 0.9) return { category: 'Low risk', color: 'text-green-600', bgColor: 'bg-green-50' };
      if (whr < 1.0) return { category: 'Moderate risk', color: 'text-yellow-600', bgColor: 'bg-yellow-50' };
      return { category: 'High risk', color: 'text-red-600', bgColor: 'bg-red-50' };
    } else {
      if (whr < 0.8) return { category: 'Low risk', color: 'text-green-600', bgColor: 'bg-green-50' };
      if (whr < 0.85) return { category: 'Moderate risk', color: 'text-yellow-600', bgColor: 'bg-yellow-50' };
      return { category: 'High risk', color: 'text-red-600', bgColor: 'bg-red-50' };
    }
  };

  const getAUDITCategory = (score) => {
    if (score <= 7) return { category: 'Low risk', color: 'text-green-600', bgColor: 'bg-green-50' };
    if (score <= 15) return { category: 'Medium risk', color: 'text-yellow-600', bgColor: 'bg-yellow-50' };
    if (score <= 19) return { category: 'High risk', color: 'text-orange-600', bgColor: 'bg-orange-50' };
    return { category: 'Very high risk', color: 'text-red-600', bgColor: 'bg-red-50' };
  };

  const getIPAQCategory = (score) => {
    if (score < 600) return { category: 'Low activity', color: 'text-red-600', bgColor: 'bg-red-50' };
    if (score < 3000) return { category: 'Moderate activity', color: 'text-yellow-600', bgColor: 'bg-yellow-50' };
    return { category: 'High activity', color: 'text-green-600', bgColor: 'bg-green-50' };
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleCSVExport = () => {
    const csvData = {
      'Date of Birth': userData.dateOfBirth ? format(new Date(userData.dateOfBirth), 'yyyy-MM-dd') : '',
      'Age': calculateAge(userData.dateOfBirth),
      'Biological Sex': userData.biologicalSex,
      'Ethnicity': userData.ethnicity,
      'BMI': userData.bmi,
      'Weight (kg)': userData.weight,
      'Height (m)': userData.height,
      'Waist Circumference (cm)': userData.waistCircumference,
      'Hip Circumference (cm)': userData.hipCircumference,
      'Neck Circumference (cm)': userData.neckCircumference,
      'WHR': userData.whr,
      'WHtR': userData.whtr,
      'BRI': userData.bri,
      'Blood Pressure': userData.bloodPressure,
      'AUDIT Score': userData.auditScore,
      'IPAQ Score': userData.ipaqScore,
      'Smoking Status': userData.smokingStatus,
      'Pack Years': userData.packYears,
      'Vaping Device Info': userData.vapingInfo?.deviceInfo || '',
      'Nicotine (mg/mL)': userData.vapingInfo?.nicotineMg || '',
      'PG/VG Ratio': userData.vapingInfo?.pgVgRatio || '',
      'Vaping Usage Pattern': userData.vapingInfo?.usagePattern || '',
      'PSECDI Score': userData.vapingInfo?.psecdiScore || '',
      'Readiness to Quit': userData.vapingInfo?.readinessToQuit || '',
      'IIEF-5 Score': userData.iiefScore,
      'Low Testosterone Symptoms': userData.lowTestosteroneSymptoms?.join(', ') || '',
      'Red Flag Questions': JSON.stringify(userData.redFlagQuestions) || '',
      'Has Menses': userData.hasMenses,
      'Age at Menarche': userData.ageAtMenarche,
      'Menstrual Regularity': userData.menstrualRegularity,
      'Last Menstrual Period': userData.lastMenstrualPeriod ? format(new Date(userData.lastMenstrualPeriod), 'yyyy-MM-dd') : '',
      'Cycle Length (days)': userData.cycleLength,
      'Period Duration (days)': userData.periodDuration,
      'Uses Contraception': userData.usesContraception,
      'Contraception Type': userData.contraceptionType,
      'Previous Pregnancies': userData.hasPreviousPregnancies,
      'Perimenopausal': userData.isPerimenopausal,
      'Postmenopausal': userData.isPostmenopausal,
      'Age at Menopause': userData.ageAtMenopause,
      'Menopause Type': userData.menopauseType,
      'On HRT': userData.onHRT,
      'HRT Type': userData.hrtType
    };

    const csvContent = Object.entries(csvData)
      .map(([key, value]) => `"${key}","${value}"`)
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medtrack_data_${format(new Date(), 'yyyy-MM-dd')}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (!userData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  const age = calculateAge(userData.dateOfBirth);
  const bmiCategory = userData.bmi ? getBMICategory(userData.bmi) : null;
  const whrCategory = userData.whr ? getWHRCategory(userData.whr, userData.biologicalSex) : null;
  const auditCategory = userData.auditScore ? getAUDITCategory(userData.auditScore) : null;
  const ipaqCategory = userData.ipaqScore ? getIPAQCategory(userData.ipaqScore) : null;

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome back, {userData.firstName || 'User'}!</h1>
          <p className="text-gray-600">Here's your health overview for today</p>
        </div>

        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <div className="bg-white">
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Key Metrics Cards */}
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="flex items-center">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <User className="w-6 h-6 text-blue-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Age</p>
                    <p className="text-2xl font-semibold text-gray-900">{userData.age || 'N/A'}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="flex items-center">
                  <div className="p-2 bg-green-100 rounded-lg">
                    <Scale className="w-6 h-6 text-green-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">BMI</p>
                    <p className="text-2xl font-semibold text-gray-900">{userData.bmi || 'N/A'}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="flex items-center">
                  <div className="p-2 bg-purple-100 rounded-lg">
                    <Heart className="w-6 h-6 text-purple-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">Blood Pressure</p>
                    <p className="text-2xl font-semibold text-gray-900">{userData.bloodPressure || 'N/A'}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="flex items-center">
                  <div className="p-2 bg-orange-100 rounded-lg">
                    <Activity className="w-6 h-6 text-orange-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600">IPAQ Score</p>
                    <p className="text-2xl font-semibold text-gray-900">{userData.ipaqScore || 'N/A'}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'demographics' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Demographics Card */}
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">Demographics</h3>
                  <User className="w-5 h-5 text-blue-600" />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Age:</span>
                    <span className="text-sm font-medium text-gray-800">{userData.age || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Sex:</span>
                    <span className="text-sm font-medium text-gray-800">{userData.biologicalSex || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Ethnicity:</span>
                    <span className="text-sm font-medium text-gray-800">{userData.ethnicity || 'N/A'}</span>
                  </div>
                </div>
              </div>

              {/* Female Health Card */}
              {userData.biologicalSex === 'female' && (
                <div className="bg-white border border-gray-200 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-800">Female Health</h3>
                    <Heart className="w-5 h-5 text-pink-600" />
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Menses:</span>
                      <span className="text-sm font-medium text-gray-800">{userData.hasMenses ? 'Yes' : 'No'}</span>
                    </div>
                    {userData.hasMenses && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Cycle Length:</span>
                          <span className="text-sm font-medium text-gray-800">{userData.cycleLength || 'N/A'} days</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Contraception:</span>
                          <span className="text-sm font-medium text-gray-800">{userData.usesContraception ? 'Yes' : 'No'}</span>
                        </div>
                      </>
                    )}
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Pregnancies:</span>
                      <span className="text-sm font-medium text-gray-800">{userData.hasPreviousPregnancies ? 'Yes' : 'No'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Menopause:</span>
                      <span className="text-sm font-medium text-gray-800">
                        {userData.isPostmenopausal ? 'Post' : userData.isPerimenopausal ? 'Peri' : 'No'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Male Health Card */}
              {userData.biologicalSex === 'male' && (
                <div className="bg-white border border-gray-200 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-800">Male Health</h3>
                    <Activity className="w-5 h-5 text-blue-600" />
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">IIEF-5 Score:</span>
                      <span className="text-sm font-medium text-gray-800">{userData.iiefScore || 'N/A'}/25</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Testosterone Symptoms:</span>
                      <span className="text-sm font-medium text-gray-800">
                        {userData.lowTestosteroneSymptoms?.length > 0 ? 'Yes' : 'No'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Lifestyle Card */}
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">Lifestyle</h3>
                  <Droplets className="w-5 h-5 text-green-600" />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">AUDIT Score:</span>
                    <span className="text-sm font-medium text-gray-800">{userData.auditScore || 'N/A'}/40</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Smoking:</span>
                    <span className="text-sm font-medium text-gray-800">{userData.smokingStatus || 'N/A'}</span>
                  </div>
                  {userData.smokingStatus === 'Current smoker' && (
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Pack Years:</span>
                      <span className="text-sm font-medium text-gray-800">
                        {userData.packYears ? userData.packYears.toFixed(1) : 'N/A'}
                      </span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">IPAQ Score:</span>
                    <span className="text-sm font-medium text-gray-800">{userData.ipaqScore || 'N/A'}</span>
                  </div>
                </div>
              </div>

              {/* Anthropometrics Card */}
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">Anthropometrics</h3>
                  <Scale className="w-5 h-5 text-purple-600" />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Weight:</span>
                    <span className="text-sm font-medium text-gray-800">{userData.weight || 'N/A'} kg</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Height:</span>
                    <span className="text-sm font-medium text-gray-800">{userData.height || 'N/A'} m</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">BMI:</span>
                    <span className="text-sm font-medium text-gray-800">{userData.bmi || 'N/A'}</span>
                  </div>
                  {userData.waistCircumference && userData.hipCircumference && (
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">WHR:</span>
                      <span className="text-sm font-medium text-gray-800">
                        {(userData.waistCircumference / userData.hipCircumference).toFixed(2)}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Medications Card */}
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">Medications</h3>
                  <FileText className="w-5 h-5 text-indigo-600" />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Total:</span>
                    <span className="text-sm font-medium text-gray-800">
                      {userData.medications?.length || 0} medications
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Active:</span>
                    <span className="text-sm font-medium text-gray-800">
                      {userData.medications?.filter(m => m.isActive)?.length || 0} active
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'medications' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold text-gray-800">Your Medications</h2>
                <button className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md hover:from-blue-700 hover:to-indigo-700">
                  Add Medication
                </button>
              </div>
              
              {userData.medications?.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {userData.medications.map((med) => (
                    <div key={med.id} className="bg-white border border-gray-200 rounded-xl p-6">
                      <div className="flex items-start justify-between mb-3">
                        <h3 className="text-lg font-semibold text-gray-800">{med.name}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          med.isActive ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {med.isActive ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm text-gray-600">Dosage: {med.dosage}</p>
                        <p className="text-sm text-gray-600">Frequency: {med.frequency}</p>
                        <p className="text-sm text-gray-600">Started: {new Date(med.startDate).toLocaleDateString()}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No medications yet</h3>
                  <p className="text-gray-500">Add your first medication to get started</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'metrics' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold text-gray-800">Health Metrics</h2>
                <button className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md hover:from-blue-700 hover:to-indigo-700">
                  Add Metric
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="bg-white border border-gray-200 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Weight Trend</h3>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-gray-900">{userData.weight || 'N/A'}</p>
                    <p className="text-sm text-gray-500">kg</p>
                  </div>
                </div>
                
                <div className="bg-white border border-gray-200 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Blood Pressure</h3>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-gray-900">{userData.bloodPressure || 'N/A'}</p>
                    <p className="text-sm text-gray-500">mmHg</p>
                  </div>
                </div>
                
                <div className="bg-white border border-gray-200 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">BMI</h3>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-gray-900">{userData.bmi || 'N/A'}</p>
                    <p className="text-sm text-gray-500">kg/m²</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MedTrackDashboard;

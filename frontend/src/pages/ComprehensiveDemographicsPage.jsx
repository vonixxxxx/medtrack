import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Navigation from '../components/Navigation';
import IIEF5Screening from '../components/IIEF5Screening';
import AUDITScreening from '../components/AUDITScreening';
import HeartRiskCalculator from '../components/HeartRiskCalculator';
import IPAQScreening from '../components/IPAQScreening';

export default function ComprehensiveDemographicsPage() {
  const [activeAssessment, setActiveAssessment] = useState(null);
  const [assessmentResults, setAssessmentResults] = useState({
    iief5: null,
    audit: null,
    heartRisk: null,
    ipaq: null
  });

  // Get user data from localStorage (signup data + any updated metrics)
  const [userData, setUserData] = useState({
    firstName: '',
    age: null,
    biologicalSex: '',
    ethnicity: '',
    bmi: null,
    bloodPressure: '',
    ipaqScore: null,
    iiefScore: null,
    auditScore: null,
    weight: null,
    height: null,
    waistCircumference: null,
    hipCircumference: null,
    hasMenses: null,
    cycleLength: null,
    usesContraception: null,
    contraceptionType: '',
    hasPreviousPregnancies: null,
    isPerimenopausal: null,
    isPostmenopausal: null,
    lowTestosteroneSymptoms: [],
    redFlagQuestions: {},
    smokingStatus: '',
    medications: []
  });

  useEffect(() => {
    // Load user data from localStorage - prioritize signup data, then any updated metrics
    const loadUserData = () => {
      try {
        // First try to load signup data
        const signupData = localStorage.getItem('signupData');
        if (signupData) {
          const parsedSignupData = JSON.parse(signupData);
          
          // Calculate age from date of birth
          let calculatedAge = null;
          if (parsedSignupData.dateOfBirth) {
            calculatedAge = calculateAge(parsedSignupData.dateOfBirth);
          }
          
          // Map signup data to userData structure
          const mappedData = {
            firstName: parsedSignupData.firstName || '',
            age: calculatedAge,
            biologicalSex: parsedSignupData.biologicalSex || '',
            ethnicity: parsedSignupData.ethnicity || '',
            weight: parsedSignupData.weight || null,
            height: parsedSignupData.height || null,
            waistCircumference: parsedSignupData.waistCircumference || null,
            hipCircumference: parsedSignupData.hipCircumference || null,
            bloodPressure: parsedSignupData.bloodPressure || '',
            ipaqScore: parsedSignupData.ipaqScore || null,
            iiefScore: parsedSignupData.iiefScore || null,
            auditScore: parsedSignupData.auditScore || null,
            hasMenses: parsedSignupData.hasMenses || null,
            cycleLength: parsedSignupData.cycleLength || null,
            usesContraception: parsedSignupData.usesContraception || null,
            contraceptionType: parsedSignupData.contraceptionType || '',
            hasPreviousPregnancies: parsedSignupData.hasPreviousPregnancies || null,
            isPerimenopausal: parsedSignupData.isPerimenopausal || null,
            isPostmenopausal: parsedSignupData.isPostmenopausal || null,
            lowTestosteroneSymptoms: parsedSignupData.lowTestosteroneSymptoms || [],
            redFlagQuestions: parsedSignupData.redFlagQuestions || {},
            smokingStatus: parsedSignupData.smokingStatus || '',
            medications: parsedSignupData.medications || []
          };
          
          setUserData(prev => ({ ...prev, ...mappedData }));
          
          // Set assessment results from signup (these should remain static)
          setAssessmentResults({
            iief5: parsedSignupData.iiefScore || null,
            audit: parsedSignupData.auditScore || null,
            heartRisk: null, // Heart risk is calculated, not stored from signup
            ipaq: parsedSignupData.ipaqScore || null
          });
        }
        
        // Then try to load any updated metrics (these can change)
        const updatedMetrics = localStorage.getItem('medtrack_updated_metrics');
        if (updatedMetrics) {
          const parsedMetrics = JSON.parse(updatedMetrics);
          setUserData(prev => ({ ...prev, ...parsedMetrics }));
        }
        
      } catch (error) {
        console.log('Error loading user data:', error);
      }
    };

    loadUserData();
  }, []);

  // Helper function to calculate age from date of birth
  const calculateAge = (dateOfBirth) => {
    const today = new Date();
    const birthDate = new Date(dateOfBirth);
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    
    return age;
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  // Calculate BMI if weight and height are available
  const calculateBMI = () => {
    if (userData.weight && userData.height) {
      return (userData.weight / (userData.height * userData.height)).toFixed(1);
    }
    return null;
  };

  // Calculate WHR if measurements are available
  const calculateWHR = () => {
    if (userData.waistCircumference && userData.hipCircumference) {
      return (userData.waistCircumference / userData.hipCircumference).toFixed(2);
    }
    return null;
  };

  const bmi = calculateBMI();
  const whr = calculateWHR();

  // Function to update metrics (called when user logs new data)
  const updateMetric = (metricType, value) => {
    const updatedData = { ...userData, [metricType]: value };
    setUserData(updatedData);
    
    // Save updated metrics to localStorage (separate from signup data)
    try {
      localStorage.setItem('medtrack_updated_metrics', JSON.stringify(updatedData));
    } catch (error) {
      console.log('Error saving updated metrics:', error);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <Navigation />
      
      <motion.div 
        className="max-w-7xl mx-auto px-6 py-8"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Header */}
        <motion.div 
          className="mb-8 text-center"
          variants={cardVariants}
        >
          <h1 className="text-4xl font-bold text-blue-400 mb-3">Health Profile</h1>
        </motion.div>

        {/* Key Metrics Overview */}
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
          variants={cardVariants}
        >
          <div className="bg-white rounded-2xl p-6 text-center shadow-lg border border-gray-200">
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {userData.age ? userData.age : '--'}
            </div>
            <div className="text-sm font-medium text-gray-400">Age</div>
          </div>
          
          <div className="bg-white rounded-2xl p-6 text-center shadow-lg border border-gray-200">
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {bmi ? bmi : '--'}
            </div>
            <div className="text-sm font-medium text-gray-400">BMI</div>
          </div>
          
          <div className="bg-white rounded-2xl p-6 text-center shadow-lg border border-gray-200">
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {userData.bloodPressure ? userData.bloodPressure : '--'}
            </div>
            <div className="text-sm font-medium text-gray-400">Blood Pressure</div>
          </div>
          
          <div className="bg-white rounded-2xl p-6 text-center shadow-lg border border-gray-200">
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {userData.ipaqScore ? userData.ipaqScore : '--'}
            </div>
            <div className="text-sm font-medium text-gray-400">IPAQ Score</div>
          </div>
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            {/* Demographics Section */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Demographics
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">Age:</span>
                  <span className="text-black font-semibold">
                    {userData.age ? userData.age : 'Not specified'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">Sex:</span>
                  <span className="text-black font-semibold capitalize">
                    {userData.biologicalSex ? userData.biologicalSex : 'Not specified'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-600 font-medium">Ethnicity:</span>
                  <span className="text-black font-semibold">
                    {userData.ethnicity ? userData.ethnicity : 'Not specified'}
                  </span>
                </div>
              </div>
            </motion.div>

            {/* Gender-Specific Health */}
            {userData.biologicalSex === 'female' && (
              <motion.div 
                className="bg-white border border-gray-200 rounded-2xl p-6"
                variants={cardVariants}
              >
                <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                  <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                  Female Health
                </h2>
                <div className="space-y-3">
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600 font-medium">Menses:</span>
                    <span className="text-blue-900 font-semibold">
                      {userData.hasMenses !== null ? (userData.hasMenses ? 'Yes' : 'No') : 'Not specified'}
                    </span>
                  </div>
                  {userData.hasMenses && (
                    <>
                      <div className="flex justify-between items-center py-2 border-b border-gray-100">
                        <span className="text-gray-600 font-medium">Cycle Length:</span>
                        <span className="text-blue-900 font-semibold">
                          {userData.cycleLength ? `${userData.cycleLength} days` : 'Not specified'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-2 border-b border-gray-100">
                        <span className="text-gray-600 font-medium">Contraception:</span>
                        <span className="text-blue-900 font-semibold">
                          {userData.usesContraception !== null ? (userData.usesContraception ? 'Yes' : 'No') : 'Not specified'}
                        </span>
                      </div>
                      {userData.usesContraception && (
                        <div className="flex justify-between items-center py-2 border-b border-gray-100">
                          <span className="text-gray-600 font-medium">Type:</span>
                          <span className="text-blue-900 font-semibold">
                            {userData.contraceptionType || 'Not specified'}
                          </span>
                        </div>
                      )}
                    </>
                  )}
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600 font-medium">Pregnancies:</span>
                    <span className="text-blue-900 font-semibold">
                      {userData.hasPreviousPregnancies !== null ? (userData.hasPreviousPregnancies ? 'Yes' : 'No') : 'Not specified'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-gray-600 font-medium">Menopause:</span>
                    <span className="text-blue-900 font-semibold">
                      {userData.isPostmenopausal ? 'Post' : userData.isPerimenopausal ? 'Peri' : 'Not specified'}
                    </span>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Lifestyle Section */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Lifestyle
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">AUDIT Score:</span>
                  <span className="text-black font-semibold">
                    {userData.auditScore ? `${userData.auditScore}/40` : 'Not assessed'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">Smoking:</span>
                  <span className="text-black font-semibold capitalize">
                    {userData.smokingStatus ? userData.smokingStatus : 'Not specified'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-600 font-medium">IPAQ Score:</span>
                  <span className="text-black font-semibold">
                    {userData.ipaqScore ? userData.ipaqScore : 'Not specified'}
                  </span>
                </div>
              </div>
            </motion.div>

            {/* Anthropometrics Section */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Anthropometrics
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">Weight:</span>
                  <span className="text-black font-semibold">
                    {userData.weight ? `${userData.weight} kg` : 'Not recorded'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">Height:</span>
                  <span className="text-black font-semibold">
                    {userData.height ? `${userData.height} m` : 'Not recorded'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">BMI:</span>
                  <span className="text-black font-semibold">
                    {bmi ? bmi : 'Not calculated'}
                  </span>
                </div>
                {whr && (
                  <div className="flex justify-between items-center py-2">
                    <span className="text-gray-600 font-medium">WHR:</span>
                    <span className="text-black font-semibold">{whr}</span>
                  </div>
                )}
              </div>
            </motion.div>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Metric History Section */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Current Metrics
              </h2>
              <div className="mb-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-white rounded-lg p-4 border border-gray-200 text-center shadow-sm">
                    <div className="text-2xl font-bold text-blue-400 mb-2">
                      {userData.height ? userData.height : '--'}
                    </div>
                    <div className="text-sm font-medium text-gray-600">Height (m)</div>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4 border border-gray-200 text-center shadow-sm">
                    <div className="text-2xl font-bold text-blue-400 mb-2">
                      {userData.weight ? userData.weight : '--'}
                    </div>
                    <div className="text-sm font-medium text-gray-600">Weight (kg)</div>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4 border border-gray-200 text-center shadow-sm">
                    <div className="text-2xl font-bold text-blue-400 mb-2">
                      {userData.bloodPressure ? userData.bloodPressure : '--'}
                    </div>
                    <div className="text-sm font-medium text-gray-600">Blood Pressure</div>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4 border border-gray-200 text-center shadow-sm">
                    <div className="text-2xl font-bold text-blue-400 mb-2">
                      {bmi ? bmi : '--'}
                    </div>
                    <div className="text-sm font-medium text-gray-600">BMI</div>
                  </div>
                </div>
              </div>

              {/* Update Metrics Form */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-blue-400 mb-4">Update Metrics</h3>
                <div className="bg-gray-50 rounded-xl p-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Weight (kg)</label>
                      <input
                        type="number"
                        step="0.1"
                        value={userData.weight || ''}
                        onChange={(e) => updateMetric('weight', parseFloat(e.target.value) || null)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:border-blue-400 focus:outline-none"
                        placeholder="Enter weight"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Height (m)</label>
                      <input
                        type="number"
                        step="0.01"
                        value={userData.height || ''}
                        onChange={(e) => updateMetric('height', parseFloat(e.target.value) || null)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:border-blue-400 focus:outline-none"
                        placeholder="Enter height"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Blood Pressure</label>
                      <input
                        type="text"
                        value={userData.bloodPressure || ''}
                        onChange={(e) => updateMetric('bloodPressure', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:border-blue-400 focus:outline-none"
                        placeholder="e.g., 120/80"
                      />
                    </div>
                  </div>
                  <div className="mt-4 text-center">
                    <p className="text-sm text-gray-600">
                      Updates are automatically saved and will recalculate BMI
                    </p>
                  </div>
                </div>
              </div>

              {/* Enhanced Metric History Table */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-blue-400 mb-4">Historical Data</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-base">
                    <thead>
                      <tr className="border-b-2 border-blue-400">
                        <th className="text-left py-4 px-4 font-bold text-gray-600 bg-white rounded-tl-lg">Date</th>
                        <th className="text-left py-4 px-3 font-bold text-gray-600 bg-white">Height</th>
                        <th className="text-left py-4 px-3 font-bold text-gray-600 bg-white">Weight</th>
                        <th className="text-left py-4 px-3 font-bold text-gray-600 bg-white">BMI</th>
                        <th className="text-left py-4 px-3 font-bold text-gray-600 bg-white">BP</th>
                        <th className="text-left py-4 px-3 font-bold text-gray-600 bg-white">IPAQ</th>
                        <th className="text-left py-4 px-3 font-bold text-gray-600 bg-white">IIEF-5</th>
                        <th className="text-left py-4 px-3 font-bold text-gray-600 bg-white">AUDIT</th>
                        <th className="text-left py-4 px-4 font-bold text-gray-600 bg-white rounded-tr-lg">Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-blue-400 hover:bg-blue-50 transition-colors">
                        <td className="py-4 px-4 text-gray-600 font-medium bg-white">Today</td>
                        <td className="py-4 px-3 text-gray-600 bg-white">
                          {userData.height ? `${userData.height}m` : '--'}
                        </td>
                        <td className="py-4 px-3 text-gray-600 bg-white">
                          {userData.weight ? `${userData.weight}kg` : '--'}
                        </td>
                        <td className="py-4 px-3 text-gray-600 bg-white">
                          {bmi ? bmi : '--'}
                        </td>
                        <td className="py-4 px-3 text-gray-600 bg-white">
                          {userData.bloodPressure ? userData.bloodPressure : '--'}
                        </td>
                        <td className="py-4 px-3 text-gray-600 bg-white">
                          {userData.ipaqScore ? userData.ipaqScore : '--'}
                        </td>
                        <td className="py-4 px-3 text-gray-600 bg-white">
                          {userData.iiefScore ? userData.iiefScore : '--'}
                        </td>
                        <td className="py-4 px-3 text-gray-600 bg-white">
                          {userData.auditScore ? userData.auditScore : '--'}
                        </td>
                        <td className="py-4 px-4 text-gray-600 bg-white">--</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </motion.div>

            {/* Medications Section */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Medications
              </h2>
              <div className="space-y-4 mb-4">
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">Total:</span>
                  <span className="text-black font-semibold">
                    {userData.medications?.length || 0} medications
                  </span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-600 font-medium">Active:</span>
                  <span className="text-black font-semibold">
                    {userData.medications?.filter(m => m.isActive)?.length || 0} active
                  </span>
                </div>
              </div>
              
              {userData.medications?.length > 0 ? (
                <div className="space-y-4">
                  {userData.medications.map((med) => (
                    <div key={med.id} className="bg-gradient-to-br from-blue-100 to-blue-200 rounded-xl p-4 border border-blue-200">
                      <div className="flex items-start justify-between mb-3">
                        <h3 className="text-lg font-semibold text-black">{med.name}</h3>
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                          med.isActive ? 'bg-blue-400 text-white' : 'bg-blue-100 text-black'
                        }`}>
                          {med.isActive ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Dosage:</span>
                          <span className="text-black font-medium ml-2">{med.dosage}</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Frequency:</span>
                          <span className="text-black font-medium ml-2">{med.frequency}</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Started:</span>
                          <span className="text-black font-medium ml-2">
                            {new Date(med.startDate).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                  <p className="text-lg font-medium">No medications recorded</p>
                  <p className="text-sm">Add your medications to track them here</p>
                </div>
              )}
            </motion.div>

            {/* Clinical Screening Section */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-10 bg-blue-300 rounded-full mr-3"></div>
                Clinical Assessments
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-black">IIEF-5</h3>
                    <span title="IIEF-5 assesses erectile function (score 5-25; higher is better)." className="text-blue-400 cursor-help">?</span>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">Erectile function screening (5 questions)</p>
                  {assessmentResults.iief5 ? (
                    <div className="text-center">
                      <div className="bg-blue-50 rounded-lg p-3 mb-3">
                        <p className="text-lg font-bold text-blue-600">{assessmentResults.iief5}</p>
                        <p className="text-sm text-gray-600">Score from signup</p>
                      </div>
                      <p className="text-xs text-gray-500">Assessment completed during registration</p>
                    </div>
                  ) : (
                    <div className="text-center py-4">
                      <p className="text-sm text-gray-500">Not completed during signup</p>
                    </div>
                  )}
                </div>

                <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-black">AUDIT</h3>
                    <span title="AUDIT screens for risky alcohol use (score 0-40)." className="text-blue-400 cursor-help">?</span>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">Alcohol use screening (10 questions)</p>
                  {assessmentResults.audit ? (
                    <div className="text-center">
                      <div className="bg-blue-50 rounded-lg p-3 mb-3">
                        <p className="text-lg font-bold text-blue-600">{assessmentResults.audit}</p>
                        <p className="text-sm text-gray-600">Score from signup</p>
                      </div>
                      <p className="text-xs text-gray-500">Assessment completed during registration</p>
                    </div>
                  ) : (
                    <div className="text-center py-4">
                      <p className="text-sm text-gray-500">Not completed during signup</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Note about assessments */}
              <div className="mt-6 p-4 bg-blue-50 rounded-xl border border-blue-200">
                <div className="flex items-center">
                  <svg className="w-5 h-5 text-blue-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-sm text-blue-700">
                    <strong>Note:</strong> Clinical assessment results are based on your responses during registration and cannot be updated here. 
                    These assessments provide baseline health information for your profile.
                  </p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

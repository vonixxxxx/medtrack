import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

const RegistrationForm = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState({
    // Core Demographics
    dateOfBirth: '',
    biologicalSex: '',
    ethnicity: '',
    
    // Female-specific fields
    hasMenses: null,
    ageAtMenarche: '',
    menstrualRegularity: '',
    lastMenstrualPeriod: '',
    cycleLength: '',
    periodDuration: '',
    usesContraception: null,
    contraceptionType: '',
    hasPreviousPregnancies: null,
    isPerimenopausal: null,
    isPostmenopausal: null,
    ageAtMenopause: '',
    menopauseType: '',
    onHRT: null,
    hrtType: '',
    
    // Male-specific fields
    iiefScore: '',
    lowTestosteroneSymptoms: [],
    redFlagQuestions: {
      gynecomastia: false,
      testicularAtrophy: false,
      infertility: false,
      pituitaryDisease: false,
      headTrauma: false,
      chemoRadiation: false
    },
    
    // Lifestyle
    auditScore: '',
    smokingStatus: '',
    smokingStartAge: '',
    cigarettesPerDay: '',
    vapingInfo: {
      deviceInfo: '',
      nicotineMg: '',
      pgVgRatio: '',
      usagePattern: '',
      psecdiScore: '',
      readinessToQuit: 5
    },
    ipaqScore: '',
    
    // Anthropometrics & Vitals
    weight: '',
    height: '',
    waistCircumference: '',
    hipCircumference: '',
    neckCircumference: '',
    bloodPressure: ''
  });

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // ONS Ethnicity Categories
  const ethnicityOptions = [
    'White - English, Welsh, Scottish, Northern Irish or British',
    'White - Irish',
    'White - Gypsy or Irish Traveller',
    'White - Any other White background',
    'Mixed or Multiple ethnic groups - White and Black Caribbean',
    'Mixed or Multiple ethnic groups - White and Black African',
    'Mixed or Multiple ethnic groups - White and Asian',
    'Mixed or Multiple ethnic groups - Any other Mixed or Multiple ethnic background',
    'Asian or Asian British - Indian',
    'Asian or Asian British - Pakistani',
    'Asian or Asian British - Bangladeshi',
    'Asian or Asian British - Chinese',
    'Asian or Asian British - Any other Asian background',
    'Black, Black British, Caribbean or African - Caribbean',
    'Black, Black British, Caribbean or African - African',
    'Black, Black British, Caribbean or African - Any other Black, Black British, Caribbean or African background',
    'Other ethnic group - Arab',
    'Other ethnic group - Any other ethnic group'
  ];

  // IIEF-5 Questions
  const iiefQuestions = [
    'How do you rate your confidence that you could get and keep an erection?',
    'When you had erections with sexual stimulation, how often were your erections hard enough for penetration?',
    'During sexual intercourse, how often were you able to maintain your erection after you had penetrated your partner?',
    'During sexual intercourse, how difficult was it to maintain your erection to completion of intercourse?',
    'When you attempted sexual intercourse, how often was it satisfactory for you?'
  ];

  // AUDIT Questions
  const auditQuestions = [
    'How often do you have a drink containing alcohol?',
    'How many drinks containing alcohol do you have on a typical day when you are drinking?',
    'How often do you have 6 or more drinks on one occasion?',
    'How often during the last year have you found that you were not able to stop drinking once you had started?',
    'How often during the last year have you failed to do what was normally expected from you because of drinking?',
    'How often during the last year have you needed a first drink in the morning to get yourself going after a heavy drinking session?',
    'How often during the last year have you had a feeling of guilt or remorse after drinking?',
    'How often during the last year have you been unable to remember what happened the night before because you had been drinking?',
    'Have you or someone else been injured as a result of your drinking?',
    'Has a relative or friend or a doctor or another health worker been concerned about your drinking or suggested you cut down?'
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear errors when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }));
    }
  };

  const handleArrayChange = (field, value, checked) => {
    setFormData(prev => ({
      ...prev,
      [field]: checked 
        ? [...prev[field], value]
        : prev[field].filter(item => item !== value)
    }));
  };

  const handleObjectChange = (field, subField, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: {
        ...prev[field],
        [subField]: value
      }
    }));
  };

  const validateStep = (step) => {
    const newErrors = {};
    
    switch (step) {
      case 1: // Core Demographics
        if (!formData.dateOfBirth) newErrors.dateOfBirth = 'Date of birth is required';
        if (!formData.biologicalSex) newErrors.biologicalSex = 'Biological sex is required';
        if (!formData.ethnicity) newErrors.ethnicity = 'Ethnicity is required';
        break;
        
      case 2: // Female-specific
        if (formData.biologicalSex === 'female') {
          if (formData.hasMenses === null) newErrors.hasMenses = 'Please indicate if you have menses';
          if (formData.hasMenses === true) {
            if (!formData.ageAtMenarche) newErrors.ageAtMenarche = 'Age at menarche is required';
            if (!formData.menstrualRegularity) newErrors.menstrualRegularity = 'Menstrual regularity is required';
            if (!formData.lastMenstrualPeriod) newErrors.lastMenstrualPeriod = 'Last menstrual period is required';
            if (!formData.cycleLength) newErrors.cycleLength = 'Cycle length is required';
            if (!formData.periodDuration) newErrors.periodDuration = 'Period duration is required';
            if (formData.usesContraception === null) newErrors.usesContraception = 'Please indicate if you use contraception';
            if (formData.usesContraception === true && !formData.contraceptionType) {
              newErrors.contraceptionType = 'Contraception type is required';
            }
          }
          if (formData.hasPreviousPregnancies === null) newErrors.hasPreviousPregnancies = 'Please indicate if you have had previous pregnancies';
          if (formData.isPerimenopausal === null) newErrors.isPerimenopausal = 'Please indicate if you are perimenopausal';
          if (formData.isPostmenopausal === null) newErrors.isPostmenopausal = 'Please indicate if you are postmenopausal';
          if (formData.isPostmenopausal === true) {
            if (!formData.ageAtMenopause) newErrors.ageAtMenopause = 'Age at menopause is required';
            if (!formData.menopauseType) newErrors.menopauseType = 'Menopause type is required';
            if (formData.onHRT === null) newErrors.onHRT = 'Please indicate if you are on HRT';
            if (formData.onHRT === true && !formData.hrtType) newErrors.hrtType = 'HRT type is required';
          }
        }
        break;
        
      case 3: // Male-specific
        if (formData.biologicalSex === 'male') {
          if (!formData.iiefScore) newErrors.iiefScore = 'IIEF-5 score is required';
          if (formData.lowTestosteroneSymptoms.length === 0) {
            newErrors.lowTestosteroneSymptoms = 'Please select at least one symptom';
          }
        }
        break;
        
      case 4: // Lifestyle
        if (!formData.auditScore) newErrors.auditScore = 'AUDIT score is required';
        if (!formData.smokingStatus) newErrors.smokingStatus = 'Smoking status is required';
        if (['current', 'ex'].includes(formData.smokingStatus)) {
          if (!formData.smokingStartAge) newErrors.smokingStartAge = 'Smoking start age is required';
          if (!formData.cigarettesPerDay) newErrors.cigarettesPerDay = 'Cigarettes per day is required';
        }
        if (!formData.ipaqScore) newErrors.ipaqScore = 'IPAQ score is required';
        break;
        
      case 5: // Anthropometrics
        if (!formData.weight) newErrors.weight = 'Weight is required';
        if (!formData.height) newErrors.height = 'Height is required';
        break;
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const nextStep = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const prevStep = () => {
    setCurrentStep(prev => prev - 1);
  };

  const handleSubmit = async () => {
    if (!validateStep(currentStep)) return;
    
    setIsSubmitting(true);
    try {
      // Submit form data to backend
      const response = await fetch('/api/registration/complete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(formData)
      });
      
      if (response.ok) {
        navigate('/dashboard');
      } else {
        const error = await response.json();
        setErrors({ submit: error.message });
      }
    } catch (error) {
      setErrors({ submit: 'Registration failed. Please try again.' });
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderStep1 = () => (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="space-y-6"
    >
      <h2 className="text-2xl font-bold text-gray-900">Core Demographics</h2>
      
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Date of Birth *
        </label>
        <input
          type="date"
          value={formData.dateOfBirth}
          onChange={(e) => handleInputChange('dateOfBirth', e.target.value)}
          className={`w-full px-3 py-2 border rounded-md ${
            errors.dateOfBirth ? 'border-red-500' : 'border-gray-300'
          }`}
        />
        {errors.dateOfBirth && (
          <p className="text-red-500 text-sm mt-1">{errors.dateOfBirth}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Biological Sex at Birth *
        </label>
        <select
          value={formData.biologicalSex}
          onChange={(e) => handleInputChange('biologicalSex', e.target.value)}
          className={`w-full px-3 py-2 border rounded-md ${
            errors.biologicalSex ? 'border-red-500' : 'border-gray-300'
          }`}
        >
          <option value="">Select biological sex</option>
          <option value="female">Female</option>
          <option value="male">Male</option>
          <option value="other">Other</option>
        </select>
        {errors.biologicalSex && (
          <p className="text-red-500 text-sm mt-1">{errors.biologicalSex}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Ethnicity *
        </label>
        <select
          value={formData.ethnicity}
          onChange={(e) => handleInputChange('ethnicity', e.target.value)}
          className={`w-full px-3 py-2 border rounded-md ${
            errors.ethnicity ? 'border-red-500' : 'border-gray-300'
          }`}
        >
          <option value="">Select ethnicity</option>
          {ethnicityOptions.map((option, index) => (
            <option key={index} value={option}>{option}</option>
          ))}
        </select>
        {errors.ethnicity && (
          <p className="text-red-500 text-sm mt-1">{errors.ethnicity}</p>
        )}
      </div>
    </motion.div>
  );

  const renderStep2 = () => (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="space-y-6"
    >
      <h2 className="text-2xl font-bold text-gray-900">Female-Specific Health</h2>
      
      {formData.biologicalSex === 'female' && (
        <>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Do you have menses? *
            </label>
            <div className="space-x-4">
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="hasMenses"
                  value="true"
                  checked={formData.hasMenses === true}
                  onChange={(e) => handleInputChange('hasMenses', e.target.value === 'true')}
                  className="mr-2"
                />
                Yes
              </label>
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="hasMenses"
                  value="false"
                  checked={formData.hasMenses === false}
                  onChange={(e) => handleInputChange('hasMenses', e.target.value === 'true')}
                  className="mr-2"
                />
                No
              </label>
            </div>
            {errors.hasMenses && (
              <p className="text-red-500 text-sm mt-1">{errors.hasMenses}</p>
            )}
          </div>

          {formData.hasMenses === true && (
            <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Age at menarche (years) *
                </label>
                <input
                  type="number"
                  min="8"
                  max="20"
                  value={formData.ageAtMenarche}
                  onChange={(e) => handleInputChange('ageAtMenarche', e.target.value)}
                  className={`w-full px-3 py-2 border rounded-md ${
                    errors.ageAtMenarche ? 'border-red-500' : 'border-gray-300'
                  }`}
                />
                {errors.ageAtMenarche && (
                  <p className="text-red-500 text-sm mt-1">{errors.ageAtMenarche}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Menstrual regularity *
                </label>
                <select
                  value={formData.menstrualRegularity}
                  onChange={(e) => handleInputChange('menstrualRegularity', e.target.value)}
                  className={`w-full px-3 py-2 border rounded-md ${
                    errors.menstrualRegularity ? 'border-red-500' : 'border-gray-300'
                  }`}
                >
                  <option value="">Select regularity</option>
                  <option value="regular">Regular</option>
                  <option value="irregular">Irregular</option>
                </select>
                {errors.menstrualRegularity && (
                  <p className="text-red-500 text-sm mt-1">{errors.menstrualRegularity}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Date of last menstrual period *
                </label>
                <input
                  type="date"
                  value={formData.lastMenstrualPeriod}
                  onChange={(e) => handleInputChange('lastMenstrualPeriod', e.target.value)}
                  className={`w-full px-3 py-2 border rounded-md ${
                    errors.lastMenstrualPeriod ? 'border-red-500' : 'border-gray-300'
                  }`}
                />
                {errors.lastMenstrualPeriod && (
                  <p className="text-red-500 text-sm mt-1">{errors.lastMenstrualPeriod}</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Cycle length (days) *
                  </label>
                  <input
                    type="number"
                    min="20"
                    max="45"
                    value={formData.cycleLength}
                    onChange={(e) => handleInputChange('cycleLength', e.target.value)}
                    className={`w-full px-3 py-2 border rounded-md ${
                      errors.cycleLength ? 'border-red-500' : 'border-gray-300'
                    }`}
                  />
                  {errors.cycleLength && (
                    <p className="text-red-500 text-sm mt-1">{errors.cycleLength}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Period duration (days) *
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="14"
                    value={formData.periodDuration}
                    onChange={(e) => handleInputChange('periodDuration', e.target.value)}
                    className={`w-full px-3 py-2 border rounded-md ${
                      errors.periodDuration ? 'border-red-500' : 'border-gray-300'
                    }`}
                  />
                  {errors.periodDuration && (
                    <p className="text-red-500 text-sm mt-1">{errors.periodDuration}</p>
                  )}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Do you use contraception? *
                </label>
                <div className="space-x-4">
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="usesContraception"
                      value="true"
                      checked={formData.usesContraception === true}
                      onChange={(e) => handleInputChange('usesContraception', e.target.value === 'true')}
                      className="mr-2"
                    />
                    Yes
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="usesContraception"
                      value="false"
                      checked={formData.usesContraception === false}
                      onChange={(e) => handleInputChange('usesContraception', e.target.value === 'true')}
                      className="mr-2"
                    />
                    No
                  </label>
                </div>
                {errors.usesContraception && (
                  <p className="text-red-500 text-sm mt-1">{errors.usesContraception}</p>
                )}
              </div>

              {formData.usesContraception === true && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Type of contraception *
                  </label>
                  <input
                    type="text"
                    value={formData.contraceptionType}
                    onChange={(e) => handleInputChange('contraceptionType', e.target.value)}
                    placeholder="e.g., Combined pill, IUD, condoms"
                    className={`w-full px-3 py-2 border rounded-md ${
                      errors.contraceptionType ? 'border-red-500' : 'border-gray-300'
                    }`}
                  />
                  {errors.contraceptionType && (
                    <p className="text-red-500 text-sm mt-1">{errors.contraceptionType}</p>
                  )}
                </div>
              )}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Have you had previous pregnancies? *
            </label>
            <div className="space-x-4">
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="hasPreviousPregnancies"
                  value="true"
                  checked={formData.hasPreviousPregnancies === true}
                  onChange={(e) => handleInputChange('hasPreviousPregnancies', e.target.value === 'true')}
                  className="mr-2"
                />
                Yes
              </label>
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="hasPreviousPregnancies"
                  value="false"
                  checked={formData.hasPreviousPregnancies === false}
                  onChange={(e) => handleInputChange('hasPreviousPregnancies', e.target.value === 'true')}
                  className="mr-2"
                />
                No
              </label>
            </div>
            {errors.hasPreviousPregnancies && (
              <p className="text-red-500 text-sm mt-1">{errors.hasPreviousPregnancies}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Are you perimenopausal? *
            </label>
            <div className="space-x-4">
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="isPerimenopausal"
                  value="true"
                  checked={formData.isPerimenopausal === true}
                  onChange={(e) => handleInputChange('isPerimenopausal', e.target.value === 'true')}
                  className="mr-2"
                />
                Yes
              </label>
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="isPerimenopausal"
                  value="false"
                  checked={formData.isPerimenopausal === false}
                  onChange={(e) => handleInputChange('isPerimenopausal', e.target.value === 'true')}
                  className="mr-2"
                />
                No
              </label>
            </div>
            {errors.isPerimenopausal && (
              <p className="text-red-500 text-sm mt-1">{errors.isPerimenopausal}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Are you postmenopausal? *
            </label>
            <div className="space-x-4">
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="isPostmenopausal"
                  value="true"
                  checked={formData.isPostmenopausal === true}
                  onChange={(e) => handleInputChange('isPostmenopausal', e.target.value === 'true')}
                  className="mr-2"
                />
                Yes
              </label>
              <label className="inline-flex items-center">
                <input
                  type="radio"
                  name="isPostmenopausal"
                  value="false"
                  checked={formData.isPostmenopausal === false}
                  onChange={(e) => handleInputChange('isPostmenopausal', e.target.value === 'true')}
                  className="mr-2"
                />
                No
              </label>
            </div>
            {errors.isPostmenopausal && (
              <p className="text-red-500 text-sm mt-1">{errors.isPostmenopausal}</p>
            )}
          </div>

          {formData.isPostmenopausal === true && (
            <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Age at menopause (years) *
                </label>
                <input
                  type="number"
                  min="30"
                  max="70"
                  value={formData.ageAtMenopause}
                  onChange={(e) => handleInputChange('ageAtMenopause', e.target.value)}
                  className={`w-full px-3 py-2 border rounded-md ${
                    errors.ageAtMenopause ? 'border-red-500' : 'border-gray-300'
                  }`}
                />
                {errors.ageAtMenopause && (
                  <p className="text-red-500 text-sm mt-1">{errors.ageAtMenopause}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Type of menopause *
                </label>
                <select
                  value={formData.menopauseType}
                  onChange={(e) => handleInputChange('menopauseType', e.target.value)}
                  className={`w-full px-3 py-2 border rounded-md ${
                    errors.menopauseType ? 'border-red-500' : 'border-gray-300'
                  }`}
                >
                  <option value="">Select menopause type</option>
                  <option value="natural">Natural</option>
                  <option value="early">Early</option>
                  <option value="premature_ovarian_insufficiency">Premature Ovarian Insufficiency</option>
                  <option value="surgical">Surgical</option>
                  <option value="induced">Induced</option>
                </select>
                {errors.menopauseType && (
                  <p className="text-red-500 text-sm mt-1">{errors.menopauseType}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Are you on HRT? *
                </label>
                <div className="space-x-4">
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="onHRT"
                      value="true"
                      checked={formData.onHRT === true}
                      onChange={(e) => handleInputChange('onHRT', e.target.value === 'true')}
                      className="mr-2"
                    />
                    Yes
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="onHRT"
                      value="false"
                      checked={formData.onHRT === false}
                      onChange={(e) => handleInputChange('onHRT', e.target.value === 'true')}
                      className="mr-2"
                    />
                    No
                  </label>
                </div>
                {errors.onHRT && (
                  <p className="text-red-500 text-sm mt-1">{errors.onHRT}</p>
                )}
              </div>

              {formData.onHRT === true && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Type of HRT *
                  </label>
                  <input
                    type="text"
                    value={formData.hrtType}
                    onChange={(e) => handleInputChange('hrtType', e.target.value)}
                    placeholder="e.g., Estrogen only, Combined HRT"
                    className={`w-full px-3 py-2 border rounded-md ${
                      errors.hrtType ? 'border-red-500' : 'border-gray-300'
                    }`}
                  />
                  {errors.hrtType && (
                    <p className="text-red-500 text-sm mt-1">{errors.hrtType}</p>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </motion.div>
  );

  const renderStep3 = () => (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="space-y-6"
    >
      <h2 className="text-2xl font-bold text-gray-900">Male-Specific Health</h2>
      
      {formData.biologicalSex === 'male' && (
        <>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              IIEF-5 Score *
            </label>
            <p className="text-sm text-gray-600 mb-4">
              Please complete the IIEF-5 questionnaire and enter your total score (5-25)
            </p>
            <input
              type="number"
              min="5"
              max="25"
              value={formData.iiefScore}
              onChange={(e) => handleInputChange('iiefScore', e.target.value)}
              className={`w-full px-3 py-2 border rounded-md ${
                errors.iiefScore ? 'border-red-500' : 'border-gray-300'
              }`}
            />
            {errors.iiefScore && (
              <p className="text-red-500 text-sm mt-1">{errors.iiefScore}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Low Testosterone Symptoms *
            </label>
            <p className="text-sm text-gray-600 mb-4">
              Select all symptoms that apply to you:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { value: 'low_libido', label: 'Low libido / reduced morning erections' },
                { value: 'fatigue', label: 'Fatigue / low energy' },
                { value: 'depressed_mood', label: 'Depressed mood / irritability' },
                { value: 'reduced_muscle_mass', label: 'Reduced muscle mass / strength' },
                { value: 'increased_body_fat', label: 'Increased body fat' },
                { value: 'reduced_shaving_frequency', label: 'Reduced shaving frequency / body hair' },
                { value: 'decreased_bone_strength', label: 'Decreased bone strength' }
              ].map((symptom) => (
                <label key={symptom.value} className="inline-flex items-center">
                  <input
                    type="checkbox"
                    value={symptom.value}
                    checked={formData.lowTestosteroneSymptoms.includes(symptom.value)}
                    onChange={(e) => handleArrayChange('lowTestosteroneSymptoms', symptom.value, e.target.checked)}
                    className="mr-2"
                  />
                  {symptom.label}
                </label>
              ))}
            </div>
            {errors.lowTestosteroneSymptoms && (
              <p className="text-red-500 text-sm mt-1">{errors.lowTestosteroneSymptoms}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Red Flag Questions *
            </label>
            <p className="text-sm text-gray-600 mb-4">
              Please answer the following questions:
            </p>
            <div className="space-y-3">
              {[
                { field: 'gynecomastia', label: 'Do you have gynecomastia (enlarged breast tissue)?' },
                { field: 'testicularAtrophy', label: 'Do you have testicular atrophy (smaller than normal testicles)?' },
                { field: 'infertility', label: 'Have you experienced infertility issues?' },
                { field: 'pituitaryDisease', label: 'Do you have a history of pituitary disease?' },
                { field: 'headTrauma', label: 'Have you experienced significant head trauma?' },
                { field: 'chemoRadiation', label: 'Have you received chemotherapy or radiation therapy?' }
              ].map((question) => (
                <div key={question.field} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-700">{question.label}</span>
                  <div className="space-x-4">
                    <label className="inline-flex items-center">
                      <input
                        type="radio"
                        name={question.field}
                        value="true"
                        checked={formData.redFlagQuestions[question.field] === true}
                        onChange={(e) => handleObjectChange('redFlagQuestions', question.field, e.target.value === 'true')}
                        className="mr-2"
                      />
                      Yes
                    </label>
                    <label className="inline-flex items-center">
                      <input
                        type="radio"
                        name={question.field}
                        value="false"
                        checked={formData.redFlagQuestions[question.field] === false}
                        onChange={(e) => handleObjectChange('redFlagQuestions', question.field, e.target.value === 'true')}
                        className="mr-2"
                      />
                      No
                    </label>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </motion.div>
  );

  const renderStep4 = () => (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="space-y-6"
    >
      <h2 className="text-2xl font-bold text-gray-900">Lifestyle Assessment</h2>
      
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          AUDIT Score *
        </label>
        <p className="text-sm text-gray-600 mb-4">
          Please complete the AUDIT questionnaire and enter your total score (0-40)
        </p>
        <input
          type="number"
          min="0"
          max="40"
          value={formData.auditScore}
          onChange={(e) => handleInputChange('auditScore', e.target.value)}
          className={`w-full px-3 py-2 border rounded-md ${
            errors.auditScore ? 'border-red-500' : 'border-gray-300'
          }`}
        />
        {errors.auditScore && (
          <p className="text-red-500 text-sm mt-1">{errors.auditScore}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Smoking Status *
        </label>
        <select
          value={formData.smokingStatus}
          onChange={(e) => handleInputChange('smokingStatus', e.target.value)}
          className={`w-full px-3 py-2 border rounded-md ${
            errors.smokingStatus ? 'border-red-500' : 'border-gray-300'
          }`}
        >
          <option value="">Select smoking status</option>
          <option value="never">Never</option>
          <option value="current">Current smoker</option>
          <option value="ex">Ex-smoker</option>
          <option value="vaping">Vaping</option>
        </select>
        {errors.smokingStatus && (
          <p className="text-red-500 text-sm mt-1">{errors.smokingStatus}</p>
        )}
      </div>

      {['current', 'ex'].includes(formData.smokingStatus) && (
        <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Age started smoking *
              </label>
              <input
                type="number"
                min="10"
                max="80"
                value={formData.smokingStartAge}
                onChange={(e) => handleInputChange('smokingStartAge', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md ${
                  errors.smokingStartAge ? 'border-red-500' : 'border-gray-300'
                }`}
              />
              {errors.smokingStartAge && (
                <p className="text-red-500 text-sm mt-1">{errors.smokingStartAge}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Cigarettes per day *
              </label>
              <input
                type="number"
                min="1"
                max="100"
                value={formData.cigarettesPerDay}
                onChange={(e) => handleInputChange('cigarettesPerDay', e.target.value)}
                className={`w-full px-3 py-2 border rounded-md ${
                  errors.cigarettesPerDay ? 'border-red-500' : 'border-gray-300'
                }`}
              />
              {errors.cigarettesPerDay && (
                <p className="text-red-500 text-sm mt-1">{errors.cigarettesPerDay}</p>
              )}
            </div>
          </div>

          {formData.dateOfBirth && formData.smokingStartAge && formData.cigarettesPerDay && (
            <div className="p-3 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Calculated Pack Years:</strong> {((formData.cigarettesPerDay / 20) * (new Date().getFullYear() - new Date(formData.dateOfBirth).getFullYear() - formData.smokingStartAge)).toFixed(1)}
              </p>
            </div>
          )}
        </div>
      )}

      {formData.smokingStatus === 'vaping' && (
        <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-medium text-gray-900">Vaping Information</h3>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Device Information
            </label>
            <input
              type="text"
              value={formData.vapingInfo.deviceInfo}
              onChange={(e) => handleObjectChange('vapingInfo', 'deviceInfo', e.target.value)}
              placeholder="e.g., Pod system, Mod, Disposable"
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Nicotine (mg/mL)
              </label>
              <input
                type="number"
                min="0"
                max="50"
                step="0.1"
                value={formData.vapingInfo.nicotineMg}
                onChange={(e) => handleObjectChange('vapingInfo', 'nicotineMg', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                PG/VG Ratio
              </label>
              <input
                type="text"
                value={formData.vapingInfo.pgVgRatio}
                onChange={(e) => handleObjectChange('vapingInfo', 'pgVgRatio', e.target.value)}
                placeholder="e.g., 70/30"
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Usage Pattern
            </label>
            <textarea
              value={formData.vapingInfo.usagePattern}
              onChange={(e) => handleObjectChange('vapingInfo', 'usagePattern', e.target.value)}
              placeholder="Describe your vaping pattern and frequency"
              rows="3"
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                PSECDI Score
              </label>
              <input
                type="number"
                min="0"
                max="20"
                value={formData.vapingInfo.psecdiScore}
                onChange={(e) => handleObjectChange('vapingInfo', 'psecdiScore', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Readiness to Quit (0-10)
              </label>
              <input
                type="range"
                min="0"
                max="10"
                value={formData.vapingInfo.readinessToQuit}
                onChange={(e) => handleObjectChange('vapingInfo', 'readinessToQuit', parseInt(e.target.value))}
                className="w-full"
              />
              <span className="text-sm text-gray-600">{formData.vapingInfo.readinessToQuit}</span>
            </div>
          </div>
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          IPAQ Score *
        </label>
        <p className="text-sm text-gray-600 mb-4">
          Please complete the IPAQ questionnaire and enter your total score (0-100)
        </p>
        <input
          type="number"
          min="0"
          max="100"
          value={formData.ipaqScore}
          onChange={(e) => handleInputChange('ipaqScore', e.target.value)}
          className={`w-full px-3 py-2 border rounded-md ${
            errors.ipaqScore ? 'border-red-500' : 'border-gray-300'
          }`}
        />
        {errors.ipaqScore && (
          <p className="text-red-500 text-sm mt-1">{errors.ipaqScore}</p>
        )}
      </div>
    </motion.div>
  );

  const renderStep5 = () => (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="space-y-6"
    >
      <h2 className="text-2xl font-bold text-gray-900">Anthropometrics & Vitals</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Weight (kg) *
          </label>
          <input
            type="number"
            min="20"
            max="500"
            step="0.1"
            value={formData.weight}
            onChange={(e) => handleInputChange('weight', e.target.value)}
            className={`w-full px-3 py-2 border rounded-md ${
              errors.weight ? 'border-red-500' : 'border-gray-300'
            }`}
          />
          {errors.weight && (
            <p className="text-red-500 text-sm mt-1">{errors.weight}</p>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Height (m) *
          </label>
          <input
            type="number"
            min="0.5"
            max="3"
            step="0.01"
            value={formData.height}
            onChange={(e) => handleInputChange('height', e.target.value)}
            className={`w-full px-3 py-2 border rounded-md ${
              errors.height ? 'border-red-500' : 'border-gray-300'
            }`}
          />
          {errors.height && (
            <p className="text-red-500 text-sm mt-1">{errors.height}</p>
          )}
        </div>
      </div>

      {formData.weight && formData.height && (
        <div className="p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>Calculated BMI:</strong> {(formData.weight / (formData.height * formData.height)).toFixed(1)}
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Waist Circumference (cm)
          </label>
          <input
            type="number"
            min="30"
            max="200"
            step="0.1"
            value={formData.waistCircumference}
            onChange={(e) => handleInputChange('waistCircumference', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Hip Circumference (cm)
          </label>
          <input
            type="number"
            min="30"
            max="200"
            step="0.1"
            value={formData.hipCircumference}
            onChange={(e) => handleInputChange('hipCircumference', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Neck Circumference (cm)
          </label>
          <input
            type="number"
            min="20"
            max="50"
            step="0.1"
            value={formData.neckCircumference}
            onChange={(e) => handleInputChange('neckCircumference', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          />
        </div>
      </div>

      {formData.waistCircumference && formData.hipCircumference && (
        <div className="p-4 bg-green-50 rounded-lg">
          <p className="text-sm text-green-800">
            <strong>Calculated WHR:</strong> {(formData.waistCircumference / formData.hipCircumference).toFixed(2)}
          </p>
        </div>
      )}

      {formData.waistCircumference && formData.height && (
        <div className="p-4 bg-green-50 rounded-lg">
          <p className="text-sm text-green-800">
            <strong>Calculated WHtR:</strong> {(formData.waistCircumference / (formData.height * 100)).toFixed(2)}
          </p>
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Blood Pressure
        </label>
        <input
          type="text"
          value={formData.bloodPressure}
          onChange={(e) => handleInputChange('bloodPressure', e.target.value)}
          placeholder="e.g., 120/80 mmHg"
          pattern="^\d{2,3}\/\d{2,3}\s*mmHg$"
          className="w-full px-3 py-2 border border-gray-300 rounded-md"
        />
        <p className="text-xs text-gray-500 mt-1">Format: systolic/diastolic mmHg (e.g., 120/80 mmHg)</p>
      </div>
    </motion.div>
  );

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900">MedTrack Registration</h1>
            <p className="text-gray-600 mt-2">Complete your health profile to get started</p>
          </div>

          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Step {currentStep} of 5</span>
              <span className="text-sm text-gray-500">{Math.round((currentStep / 5) * 100)}% Complete</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(currentStep / 5) * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Form Steps */}
          {currentStep === 1 && renderStep1()}
          {currentStep === 2 && renderStep2()}
          {currentStep === 3 && renderStep3()}
          {currentStep === 4 && renderStep4()}
          {currentStep === 5 && renderStep5()}

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-8">
            <button
              onClick={prevStep}
              disabled={currentStep === 1}
              className={`px-6 py-2 rounded-md ${
                currentStep === 1
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-gray-600 text-white hover:bg-gray-700'
              }`}
            >
              Previous
            </button>

            {currentStep < 5 ? (
              <button
                onClick={nextStep}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Next
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className={`px-6 py-2 rounded-md ${
                  isSubmitting
                    ? 'bg-gray-400 text-gray-600 cursor-not-allowed'
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isSubmitting ? 'Submitting...' : 'Complete Registration'}
              </button>
            )}
          </div>

          {errors.submit && (
            <p className="text-red-500 text-center mt-4">{errors.submit}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default RegistrationForm;

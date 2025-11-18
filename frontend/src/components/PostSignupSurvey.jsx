import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  User, 
  Calendar, 
  ChevronLeft, 
  ChevronRight,
  Check,
  Heart,
  Activity,
  ClipboardCheck
} from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent } from './ui/card';
import api from '../api';

const PostSignupSurvey = ({ isOpen, onComplete, userEmail }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Survey data state - keeping all the same fields
  const [formData, setFormData] = useState({
    // Step 1: Basic Demographics (Required)
    name: '',
    dateOfBirth: '',
    biologicalSex: '',
    ethnicity: '',
    location: '',
    postcode: '',
    nhsNumber: '',

    // Step 2: Medical History (Optional)
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
    isOnHRT: null,
    hrtType: '',

    // Step 3: Male-Specific Questions
    iiefScores: [null, null, null, null, null],
    lowTestosteroneSymptoms: [],
    redFlagQuestions: [],

    // Step 4: Lifestyle Assessment
    auditScores: [null, null, null, null, null, null, null, null, null, null],
    smokingStatus: '',
    smokingStartAge: '',
    cigarettesPerDay: '',
    vapingDevice: '',
    nicotineMg: '',
    pgVgRatio: '',
    usagePattern: '',
    psecdiScore: 0,
    readinessToQuit: 0,
    ipaqScore: 0,

    // Step 5: Physical Measurements & Medical Data (Optional)
    weight: '',
    height: '',
    waistCircumference: '',
    hipCircumference: '',
    neckCircumference: '',
    systolicBP: '',
    diastolicBP: '',
    
    // Additional medical fields
    baselineWeight: '',
    baselineWeightDate: '',
    baselineBMI: '',
    baselineHbA1c: '',
    baselineHbA1cDate: '',
    baselineFastingGlucose: '',
    randomGlucose: '',
    baselineTC: '',
    baselineHDL: '',
    baselineLDL: '',
    baselineTG: '',
    baselineLipidDate: '',
    
    // Medical conditions (optional checkboxes)
    ascvd: false,
    htn: false,
    hypertension: false,
    dyslipidaemia: false,
    ischaemicHeartDisease: false,
    heartFailure: false,
    cerebrovascularDisease: false,
    pulmonaryHypertension: false,
    dvt: false,
    pe: false,
    osa: false,
    sleepStudies: false,
    cpap: false,
    asthma: false,
    t2dm: false,
    prediabetes: false,
    diabetesType: '',
    gord: false,
    ckd: false,
    kidneyStones: false,
    masld: false,
    infertility: false,
    pcos: false,
    anxiety: false,
    depression: false,
    bipolarDisorder: false,
    emotionalEating: false,
    schizoaffectiveDisorder: false,
    oaKnee: false,
    oaHip: false,
    limitedMobility: false,
    lymphoedema: false,
    thyroidDisorder: false,
    iih: false,
    epilepsy: false,
    functionalNeurologicalDisorder: false,
    cancer: false,
    bariatricGastricBand: false,
    bariatricSleeve: false,
    bariatricBypass: false,
    bariatricBalloon: false,
    
    // Medications
    lipidLoweringTreatment: '',
    antihypertensiveMedications: '',
    allMedicationsFromSCR: '',
    
    // Clinical data
    diagnosesCodedInSCR: '',
    totalQualifyingComorbidities: '',
    mes: '',
    notes: '',
    criteriaForWegovy: ''
  });

  const steps = [
    { number: 1, title: 'Basic Demographics', description: 'Tell us about yourself', icon: User },
    { number: 2, title: 'Health Profile', description: 'Your health information', icon: Heart },
    { number: 3, title: 'Lifestyle Assessment', description: 'Your daily habits and activities', icon: Activity },
    { number: 4, title: 'Physical Measurements', description: 'Your current health metrics', icon: ClipboardCheck },
    { number: 5, title: 'Medical Conditions', description: 'Your medical history (optional)', icon: Heart },
    { number: 6, title: 'Review & Complete', description: 'Review your information', icon: Check }
  ];

  const totalSteps = steps.length;

  // Ethnicity options (ONS categories)
  const ethnicityOptions = [
    'White - English, Welsh, Scottish, Northern Irish or British',
    'White - Irish',
    'White - Gypsy or Irish Traveller',
    'White - Roma',
    'White - Other',
    'Mixed or Multiple ethnic groups - White and Black Caribbean',
    'Mixed or Multiple ethnic groups - White and Black African',
    'Mixed or Multiple ethnic groups - White and Asian',
    'Mixed or Multiple ethnic groups - Other',
    'Asian or Asian British - Indian',
    'Asian or Asian British - Pakistani',
    'Asian or Asian British - Bangladeshi',
    'Asian or Asian British - Chinese',
    'Asian or Asian British - Other',
    'Black, Black British, Caribbean or African - Caribbean',
    'Black, Black British, Caribbean or African - African',
    'Black, Black British, Caribbean or African - Other',
    'Other ethnic group - Arab',
    'Other ethnic group - Any other ethnic group',
    'Prefer not to say'
  ];

  // IIEF-5 Questions for male users
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
    'How often do you have six or more drinks on one occasion?',
    'How often during the last year have you found that you were not able to stop drinking once you had started?',
    'How often during the last year have you failed to do what was normally expected from you because of drinking?',
    'How often during the last year have you needed a first drink in the morning to get yourself going after a heavy drinking session?',
    'How often during the last year have you had a feeling of guilt or remorse after drinking?',
    'How often during the last year have you been unable to remember what happened the night before because you had been drinking?',
    'Have you or someone else been injured as a result of your drinking?',
    'Has a relative or friend, doctor or other health worker been concerned about your drinking or suggested you cut down?'
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (error) setError('');
  };

  const handleArrayChange = (field, value, checked) => {
    setFormData(prev => ({
      ...prev,
      [field]: checked 
        ? [...prev[field], value]
        : prev[field].filter(item => item !== value)
    }));
  };

  const validateStep = (step) => {
    switch (step) {
      case 1:
        if (!formData.name || !formData.dateOfBirth || !formData.biologicalSex || !formData.ethnicity) {
          setError('Name, date of birth, biological sex, and ethnicity are required');
          return false;
        }
        return true;
      
      case 2:
        if (formData.biologicalSex === 'Female') {
          if (formData.hasMenses === null) {
            setError('Please answer whether you have menses');
            return false;
          }
          if (formData.hasMenses === true) {
            if (!formData.ageAtMenarche || !formData.menstrualRegularity || !formData.lastMenstrualPeriod) {
              setError('Please complete all menstrual cycle questions');
              return false;
            }
          } else {
            if (formData.isPerimenopausal === null && formData.isPostmenopausal === null) {
              setError('Please indicate if you are perimenopausal or postmenopausal');
              return false;
            }
          }
        } else if (formData.biologicalSex === 'Male') {
          if (formData.iiefScores.some(score => score === null || score === undefined)) {
            setError('Please complete the IIEF-5 questionnaire');
            return false;
          }
        }
        return true;
      
      case 3:
        if (formData.auditScores.some(score => score === null || score === undefined)) {
          setError('Please complete the AUDIT questionnaire');
          return false;
        }
        if (!formData.smokingStatus) {
          setError('Please indicate your smoking status');
          return false;
        }
        return true;
      
      case 4:
        if (!formData.weight || !formData.height) {
          setError('Weight and height are required');
          return false;
        }
        return true;
      
      case 5:
        return true;
      
      case 6:
        return true;
      
      default:
        return true;
    }
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    setCurrentStep(prev => prev - 1);
  };

  const handleSubmit = async () => {
    if (!validateStep(currentStep)) return;
    
    setIsLoading(true);
    setError('');

    try {
      const surveyData = {
        name: formData.name,
        dateOfBirth: new Date(formData.dateOfBirth),
        biologicalSex: formData.biologicalSex,
        ethnicity: formData.ethnicity,
        location: formData.location || '',
        postcode: formData.postcode || '',
        nhsNumber: formData.nhsNumber || '',
        
        hasMenses: formData.hasMenses,
        ageAtMenarche: formData.ageAtMenarche ? parseInt(formData.ageAtMenarche) : null,
        menstrualRegularity: formData.menstrualRegularity || '',
        lastMenstrualPeriod: formData.lastMenstrualPeriod ? new Date(formData.lastMenstrualPeriod) : null,
        cycleLength: formData.cycleLength ? parseInt(formData.cycleLength) : null,
        periodDuration: formData.periodDuration ? parseInt(formData.periodDuration) : null,
        usesContraception: formData.usesContraception,
        contraceptionType: formData.contraceptionType || '',
        hasPreviousPregnancies: formData.hasPreviousPregnancies,
        isPerimenopausal: formData.isPerimenopausal,
        isPostmenopausal: formData.isPostmenopausal,
        ageAtMenopause: formData.ageAtMenopause ? parseInt(formData.ageAtMenopause) : null,
        menopauseType: formData.menopauseType || '',
        isOnHRT: formData.isOnHRT,
        hrtType: formData.hrtType || '',
        iiefScore: formData.iiefScores.reduce((sum, score) => sum + (score || 0), 0),
        lowTestosteroneSymptoms: JSON.stringify(formData.lowTestosteroneSymptoms),
        redFlagQuestions: JSON.stringify(formData.redFlagQuestions),
        auditScore: formData.auditScores.reduce((sum, score) => sum + (score || 0), 0),
        smokingStatus: formData.smokingStatus,
        smokingStartAge: formData.smokingStartAge ? parseInt(formData.smokingStartAge) : null,
        cigarettesPerDay: formData.cigarettesPerDay ? parseInt(formData.cigarettesPerDay) : null,
        vapingDevice: formData.vapingDevice || '',
        nicotineMg: formData.nicotineMg ? parseFloat(formData.nicotineMg) : null,
        pgVgRatio: formData.pgVgRatio || '',
        usagePattern: formData.usagePattern || '',
        psecdiScore: formData.psecdiScore || 0,
        readinessToQuit: formData.readinessToQuit || 0,
        ipaqScore: formData.ipaqScore || 0,
        
        weight: parseFloat(formData.weight),
        height: parseFloat(formData.height),
        waistCircumference: formData.waistCircumference ? parseFloat(formData.waistCircumference) : null,
        hipCircumference: formData.hipCircumference ? parseFloat(formData.hipCircumference) : null,
        neckCircumference: formData.neckCircumference ? parseFloat(formData.neckCircumference) : null,
        systolicBP: formData.systolicBP ? parseInt(formData.systolicBP) : null,
        diastolicBP: formData.diastolicBP ? parseInt(formData.diastolicBP) : null,
        
        baselineWeight: formData.baselineWeight ? parseFloat(formData.baselineWeight) : null,
        baselineWeightDate: formData.baselineWeightDate ? new Date(formData.baselineWeightDate) : null,
        baselineBMI: formData.baselineBMI ? parseFloat(formData.baselineBMI) : null,
        baselineHbA1c: formData.baselineHbA1c ? parseFloat(formData.baselineHbA1c) : null,
        baselineHbA1cDate: formData.baselineHbA1cDate ? new Date(formData.baselineHbA1cDate) : null,
        baselineFastingGlucose: formData.baselineFastingGlucose ? parseFloat(formData.baselineFastingGlucose) : null,
        randomGlucose: formData.randomGlucose ? parseFloat(formData.randomGlucose) : null,
        baselineTC: formData.baselineTC ? parseFloat(formData.baselineTC) : null,
        baselineHDL: formData.baselineHDL ? parseFloat(formData.baselineHDL) : null,
        baselineLDL: formData.baselineLDL ? parseFloat(formData.baselineLDL) : null,
        baselineTG: formData.baselineTG ? parseFloat(formData.baselineTG) : null,
        baselineLipidDate: formData.baselineLipidDate ? new Date(formData.baselineLipidDate) : null,
        
        ascvd: formData.ascvd,
        htn: formData.htn,
        hypertension: formData.hypertension,
        dyslipidaemia: formData.dyslipidaemia,
        ischaemicHeartDisease: formData.ischaemicHeartDisease,
        heartFailure: formData.heartFailure,
        cerebrovascularDisease: formData.cerebrovascularDisease,
        pulmonaryHypertension: formData.pulmonaryHypertension,
        dvt: formData.dvt,
        pe: formData.pe,
        osa: formData.osa,
        sleepStudies: formData.sleepStudies,
        cpap: formData.cpap,
        asthma: formData.asthma,
        t2dm: formData.t2dm,
        prediabetes: formData.prediabetes,
        diabetesType: formData.diabetesType || '',
        gord: formData.gord,
        ckd: formData.ckd,
        kidneyStones: formData.kidneyStones,
        masld: formData.masld,
        infertility: formData.infertility,
        pcos: formData.pcos,
        anxiety: formData.anxiety,
        depression: formData.depression,
        bipolarDisorder: formData.bipolarDisorder,
        emotionalEating: formData.emotionalEating,
        schizoaffectiveDisorder: formData.schizoaffectiveDisorder,
        oaKnee: formData.oaKnee,
        oaHip: formData.oaHip,
        limitedMobility: formData.limitedMobility,
        lymphoedema: formData.lymphoedema,
        thyroidDisorder: formData.thyroidDisorder,
        iih: formData.iih,
        epilepsy: formData.epilepsy,
        functionalNeurologicalDisorder: formData.functionalNeurologicalDisorder,
        cancer: formData.cancer,
        bariatricGastricBand: formData.bariatricGastricBand,
        bariatricSleeve: formData.bariatricSleeve,
        bariatricBypass: formData.bariatricBypass,
        bariatricBalloon: formData.bariatricBalloon,
        
        lipidLoweringTreatment: formData.lipidLoweringTreatment || '',
        antihypertensiveMedications: formData.antihypertensiveMedications || '',
        allMedicationsFromSCR: formData.allMedicationsFromSCR || '',
        diagnosesCodedInSCR: formData.diagnosesCodedInSCR || '',
        totalQualifyingComorbidities: formData.totalQualifyingComorbidities ? parseInt(formData.totalQualifyingComorbidities) : null,
        mes: formData.mes ? parseFloat(formData.mes) : null,
        notes: formData.notes || '',
        criteriaForWegovy: formData.criteriaForWegovy || ''
      };

      await api.post('auth/survey-data', surveyData);
      
      // Get user ID from localStorage to mark survey as complete
      const userStr = localStorage.getItem('user');
      const user = userStr ? JSON.parse(userStr) : null;
      const userId = user?.id;
      
      await api.put('auth/complete-survey', { userId });
      
      console.log('Survey completed successfully for user:', userId);
      onComplete();
    } catch (err) {
      console.error('Survey submission error:', err);
      setError(err.response?.data?.error || err.response?.data?.message || 'Failed to save survey data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Render step content - keeping all the same logic but updating styles
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="inline-flex p-4 bg-primary-50 rounded-3xl mb-4"
              >
                <User className="w-8 h-8 text-primary-600" />
              </motion.div>
              <h3 className="text-2xl font-bold text-neutral-900 mb-2">Welcome to MedTrack!</h3>
              <p className="text-neutral-600">
                Let&apos;s complete your health profile to provide you with personalized medication tracking and insights.
              </p>
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-900 mb-2">
                Full Name <span className="text-error-500">*</span>
              </label>
              <Input
                type="text"
                value={formData.name}
                onChange={(e) => handleInputChange('name', e.target.value)}
                placeholder="Enter your full name"
                className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-900 mb-2">
                Date of Birth <span className="text-error-500">*</span>
              </label>
              <div className="relative">
                <Calendar className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-neutral-400" />
                <Input
                  type="date"
                  value={formData.dateOfBirth}
                  onChange={(e) => handleInputChange('dateOfBirth', e.target.value)}
                  className="pl-12 bg-white border-neutral-200 text-neutral-900 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-900 mb-2">
                Biological Sex <span className="text-error-500">*</span>
              </label>
              <select
                value={formData.biologicalSex}
                onChange={(e) => handleInputChange('biologicalSex', e.target.value)}
                className="w-full px-4 py-3 bg-white border border-neutral-200 rounded-xl text-neutral-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-500 focus:outline-none h-12"
              >
                <option value="">Select biological sex</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-900 mb-2">
                Ethnicity <span className="text-error-500">*</span>
              </label>
              <select
                value={formData.ethnicity}
                onChange={(e) => handleInputChange('ethnicity', e.target.value)}
                className="w-full px-4 py-3 bg-white border border-neutral-200 rounded-xl text-neutral-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-500 focus:outline-none h-12"
              >
                <option value="">Select ethnicity</option>
                {ethnicityOptions.map((option, index) => (
                  <option key={index} value={option}>{option}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-900 mb-2">
                Location <span className="text-neutral-400 text-xs">(optional)</span>
              </label>
              <Input
                type="text"
                value={formData.location}
                onChange={(e) => handleInputChange('location', e.target.value)}
                placeholder="e.g., London, Manchester, Birmingham"
                className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-semibold text-neutral-900 mb-2">
                  Postcode <span className="text-neutral-400 text-xs">(optional)</span>
                </label>
                <Input
                  type="text"
                  value={formData.postcode}
                  onChange={(e) => handleInputChange('postcode', e.target.value)}
                  placeholder="e.g., SW1A 1AA"
                  className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-neutral-900 mb-2">
                  NHS Number <span className="text-neutral-400 text-xs">(optional)</span>
                </label>
                <Input
                  type="text"
                  value={formData.nhsNumber}
                  onChange={(e) => handleInputChange('nhsNumber', e.target.value)}
                  placeholder="10-digit NHS number"
                  className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            {formData.biologicalSex === 'Female' && (
              <>
                <div className="text-center mb-6">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="inline-flex p-4 bg-primary-50 rounded-3xl mb-4"
                  >
                    <Heart className="w-8 h-8 text-primary-600" />
                  </motion.div>
                  <h3 className="text-xl font-bold text-neutral-900 mb-2">Female Health Information</h3>
                  <p className="text-neutral-600 text-sm">Help us understand your reproductive health</p>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-neutral-900 mb-3">
                    Do you have menses? <span className="text-error-500">*</span>
                  </label>
                  <div className="flex gap-3">
                    <button
                      type="button"
                      onClick={() => handleInputChange('hasMenses', true)}
                      className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                        formData.hasMenses === true
                          ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                          : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                      }`}
                    >
                      Yes
                    </button>
                    <button
                      type="button"
                      onClick={() => handleInputChange('hasMenses', false)}
                      className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                        formData.hasMenses === false
                          ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                          : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                      }`}
                    >
                      No
                    </button>
                  </div>
                </div>

                {formData.hasMenses === true && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-6 pt-4 border-t border-neutral-100"
                  >
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-semibold text-neutral-900 mb-2">
                          Age at menarche (years) <span className="text-error-500">*</span>
                        </label>
                        <Input
                          type="number"
                          placeholder="Age at first period"
                          value={formData.ageAtMenarche}
                          onChange={(e) => handleInputChange('ageAtMenarche', e.target.value)}
                          className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-semibold text-neutral-900 mb-2">
                          Menstrual regularity <span className="text-error-500">*</span>
                        </label>
                        <select
                          value={formData.menstrualRegularity}
                          onChange={(e) => handleInputChange('menstrualRegularity', e.target.value)}
                          className="w-full px-4 py-3 bg-white border border-neutral-200 rounded-xl text-neutral-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-500 focus:outline-none h-12"
                        >
                          <option value="">Select regularity</option>
                          <option value="Regular">Regular (21-35 days)</option>
                          <option value="Irregular">Irregular</option>
                          <option value="Very irregular">Very irregular</option>
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-semibold text-neutral-900 mb-2">
                        Date of last menstrual period <span className="text-error-500">*</span>
                      </label>
                      <Input
                        type="date"
                        value={formData.lastMenstrualPeriod}
                        onChange={(e) => handleInputChange('lastMenstrualPeriod', e.target.value)}
                        className="bg-white border-neutral-200 text-neutral-900 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-semibold text-neutral-900 mb-2">
                          Typical cycle length (days)
                        </label>
                        <Input
                          type="number"
                          placeholder="Days"
                          value={formData.cycleLength}
                          onChange={(e) => handleInputChange('cycleLength', e.target.value)}
                          className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-semibold text-neutral-900 mb-2">
                          Typical period duration (days)
                        </label>
                        <Input
                          type="number"
                          placeholder="Days"
                          value={formData.periodDuration}
                          onChange={(e) => handleInputChange('periodDuration', e.target.value)}
                          className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-semibold text-neutral-900 mb-3">
                        Do you use contraception?
                      </label>
                      <div className="flex gap-3">
                        <button
                          type="button"
                          onClick={() => handleInputChange('usesContraception', true)}
                          className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                            formData.usesContraception === true
                              ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                              : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                          }`}
                        >
                          Yes
                        </button>
                        <button
                          type="button"
                          onClick={() => handleInputChange('usesContraception', false)}
                          className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                            formData.usesContraception === false
                              ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                              : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                          }`}
                        >
                          No
                        </button>
                      </div>
                    </div>

                    {formData.usesContraception === true && (
                      <div>
                        <label className="block text-sm font-semibold text-neutral-900 mb-2">
                          Type of contraception
                        </label>
                        <Input
                          type="text"
                          placeholder="e.g., Oral contraceptive pill, IUD, etc."
                          value={formData.contraceptionType}
                          onChange={(e) => handleInputChange('contraceptionType', e.target.value)}
                          className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                        />
                      </div>
                    )}

                    <div>
                      <label className="block text-sm font-semibold text-neutral-900 mb-3">
                        Previous pregnancies?
                      </label>
                      <div className="flex gap-3">
                        <button
                          type="button"
                          onClick={() => handleInputChange('hasPreviousPregnancies', true)}
                          className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                            formData.hasPreviousPregnancies === true
                              ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                              : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                          }`}
                        >
                          Yes
                        </button>
                        <button
                          type="button"
                          onClick={() => handleInputChange('hasPreviousPregnancies', false)}
                          className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                            formData.hasPreviousPregnancies === false
                              ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                              : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                          }`}
                        >
                          No
                        </button>
                      </div>
                    </div>
                  </motion.div>
                )}

                {formData.hasMenses === false && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-6 pt-4 border-t border-neutral-100"
                  >
                    <div>
                      <label className="block text-sm font-semibold text-neutral-900 mb-3">
                        Are you perimenopausal?
                      </label>
                      <div className="flex gap-3">
                        <button
                          type="button"
                          onClick={() => handleInputChange('isPerimenopausal', true)}
                          className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                            formData.isPerimenopausal === true
                              ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                              : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                          }`}
                        >
                          Yes
                        </button>
                        <button
                          type="button"
                          onClick={() => handleInputChange('isPerimenopausal', false)}
                          className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                            formData.isPerimenopausal === false
                              ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                              : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                          }`}
                        >
                          No
                        </button>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-semibold text-neutral-900 mb-3">
                        Are you postmenopausal?
                      </label>
                      <div className="flex gap-3">
                        <button
                          type="button"
                          onClick={() => handleInputChange('isPostmenopausal', true)}
                          className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                            formData.isPostmenopausal === true
                              ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                              : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                          }`}
                        >
                          Yes
                        </button>
                        <button
                          type="button"
                          onClick={() => handleInputChange('isPostmenopausal', false)}
                          className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                            formData.isPostmenopausal === false
                              ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                              : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                          }`}
                        >
                          No
                        </button>
                      </div>
                    </div>

                    {formData.isPostmenopausal === true && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-6 pt-4 border-t border-neutral-100"
                      >
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="block text-sm font-semibold text-neutral-900 mb-2">
                              Age at menopause
                            </label>
                            <Input
                              type="number"
                              placeholder="Age at menopause"
                              value={formData.ageAtMenopause}
                              onChange={(e) => handleInputChange('ageAtMenopause', e.target.value)}
                              className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-semibold text-neutral-900 mb-2">
                              Type of menopause
                            </label>
                            <select
                              value={formData.menopauseType}
                              onChange={(e) => handleInputChange('menopauseType', e.target.value)}
                              className="w-full px-4 py-3 bg-white border border-neutral-200 rounded-xl text-neutral-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-500 focus:outline-none h-12"
                            >
                              <option value="">Select type</option>
                              <option value="natural">Natural</option>
                              <option value="surgical">Surgical</option>
                            </select>
                          </div>
                        </div>

                        <div>
                          <label className="block text-sm font-semibold text-neutral-900 mb-3">
                            Are you on HRT?
                          </label>
                          <div className="flex gap-3">
                            <button
                              type="button"
                              onClick={() => handleInputChange('isOnHRT', true)}
                              className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                                formData.isOnHRT === true
                                  ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                                  : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                              }`}
                            >
                              Yes
                            </button>
                            <button
                              type="button"
                              onClick={() => handleInputChange('isOnHRT', false)}
                              className={`flex-1 px-6 py-4 rounded-2xl font-semibold transition-all duration-200 ${
                                formData.isOnHRT === false
                                  ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                                  : 'bg-neutral-50 text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                              }`}
                            >
                              No
                            </button>
                          </div>
                        </div>

                        {formData.isOnHRT === true && (
                          <div>
                            <label className="block text-sm font-semibold text-neutral-900 mb-2">
                              Type of HRT
                            </label>
                            <Input
                              type="text"
                              placeholder="e.g., Estrogen only, Combined HRT, etc."
                              value={formData.hrtType}
                              onChange={(e) => handleInputChange('hrtType', e.target.value)}
                              className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                            />
                          </div>
                        )}
                      </motion.div>
                    )}
                  </motion.div>
                )}
              </>
            )}

            {formData.biologicalSex === 'Male' && (
              <>
                <div className="text-center mb-6">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="inline-flex p-4 bg-primary-50 rounded-3xl mb-4"
                  >
                    <Activity className="w-8 h-8 text-primary-600" />
                  </motion.div>
                  <h3 className="text-xl font-bold text-neutral-900 mb-2">Male Health Information</h3>
                  <p className="text-neutral-600 text-sm">IIEF-5 Questionnaire</p>
                </div>

                <div className="space-y-6">
                  <p className="text-sm text-neutral-600 bg-primary-50 p-4 rounded-2xl border border-primary-100">
                    Please rate your sexual function over the past 4 weeks (0 = No sexual activity, 1 = Almost never/Never, 2 = A few times, 3 = Sometimes, 4 = Most times, 5 = Almost always/Always)
                  </p>
                  
                  {iiefQuestions.map((question, index) => (
                    <div key={index} className="bg-neutral-50 p-6 rounded-2xl border border-neutral-100">
                      <label className="block text-sm font-semibold text-neutral-900 mb-4">
                        {index + 1}. {question}
                      </label>
                      <div className="flex gap-2">
                        {[0, 1, 2, 3, 4, 5].map((score) => (
                          <button
                            key={score}
                            type="button"
                            onClick={() => {
                              const newScores = [...formData.iiefScores];
                              newScores[index] = score;
                              handleInputChange('iiefScores', newScores);
                            }}
                            className={`flex-1 py-3 rounded-xl font-semibold transition-all duration-200 ${
                              formData.iiefScores[index] === score
                                ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                                : 'bg-white text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                            }`}
                          >
                            {score}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-8 space-y-4">
                  <label className="block text-sm font-semibold text-neutral-900 mb-3">
                    Symptoms of low testosterone (select all that apply)
                  </label>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      'Reduced libido',
                      'Erectile dysfunction',
                      'Fatigue / low energy',
                      'Depressed mood / irritability',
                      'Reduced muscle mass / strength',
                      'Increased fat mass',
                      'Decreased bone strength',
                      'N/A'
                    ].map((symptom) => (
                      <label key={symptom} className="flex items-center space-x-3 p-4 bg-neutral-50 rounded-2xl border-2 border-neutral-200 hover:border-primary-300 cursor-pointer transition-all">
                        <input
                          type="checkbox"
                          checked={formData.lowTestosteroneSymptoms.includes(symptom)}
                          onChange={(e) => handleArrayChange('lowTestosteroneSymptoms', symptom, e.target.checked)}
                          className="w-5 h-5 text-primary-600 bg-white border-neutral-300 rounded focus:ring-primary-500 focus:ring-2"
                        />
                        <span className="text-neutral-900 text-sm font-medium">{symptom}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="mt-6 space-y-4">
                  <label className="block text-sm font-semibold text-neutral-900 mb-3">
                    Red flag questions (select all that apply)
                  </label>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      'Gynecomastia',
                      'Testicular atrophy',
                      'Infertility',
                      'Pituitary disease',
                      'Head trauma',
                      'Chemo/radiation'
                    ].map((flag) => (
                      <label key={flag} className="flex items-center space-x-3 p-4 bg-neutral-50 rounded-2xl border-2 border-neutral-200 hover:border-primary-300 cursor-pointer transition-all">
                        <input
                          type="checkbox"
                          checked={formData.redFlagQuestions.includes(flag)}
                          onChange={(e) => handleArrayChange('redFlagQuestions', flag, e.target.checked)}
                          className="w-5 h-5 text-primary-600 bg-white border-neutral-300 rounded focus:ring-primary-500 focus:ring-2"
                        />
                        <span className="text-neutral-900 text-sm font-medium">{flag}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="inline-flex p-4 bg-primary-50 rounded-3xl mb-4"
              >
                <Activity className="w-8 h-8 text-primary-600" />
              </motion.div>
              <h3 className="text-xl font-bold text-neutral-900 mb-2">Lifestyle Assessment</h3>
              <p className="text-neutral-600 text-sm">AUDIT Questionnaire</p>
            </div>

            <div className="space-y-6">
              <p className="text-sm text-neutral-600 bg-primary-50 p-4 rounded-2xl border border-primary-100">
                Please answer these questions about your alcohol consumption (0 = Never, 1 = Monthly or less, 2 = 2-4 times a month, 3 = 2-3 times a week, 4 = 4+ times a week)
              </p>
              
              {auditQuestions.map((question, index) => (
                <div key={index} className="bg-neutral-50 p-6 rounded-2xl border border-neutral-100">
                  <label className="block text-sm font-semibold text-neutral-900 mb-4">
                    {index + 1}. {question}
                  </label>
                  <div className="flex gap-2">
                    {[0, 1, 2, 3, 4].map((score) => (
                      <button
                        key={score}
                        type="button"
                        onClick={() => {
                          const newScores = [...formData.auditScores];
                          newScores[index] = score;
                          handleInputChange('auditScores', newScores);
                        }}
                        className={`flex-1 py-3 rounded-xl font-semibold transition-all duration-200 ${
                          formData.auditScores[index] === score
                            ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/25'
                            : 'bg-white text-neutral-700 border-2 border-neutral-200 hover:border-primary-300'
                        }`}
                      >
                        {score}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="pt-6 border-t border-neutral-100">
              <label className="block text-sm font-semibold text-neutral-900 mb-2">
                Smoking status <span className="text-error-500">*</span>
              </label>
              <select
                value={formData.smokingStatus}
                onChange={(e) => handleInputChange('smokingStatus', e.target.value)}
                className="w-full px-4 py-3 bg-white border border-neutral-200 rounded-xl text-neutral-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-500 focus:outline-none h-12"
              >
                <option value="">Select smoking status</option>
                <option value="Never smoked">Never smoked</option>
                <option value="Former smoker">Former smoker</option>
                <option value="Current smoker">Current smoker</option>
                <option value="Vaping only">Vaping only</option>
                <option value="Both smoking and vaping">Both smoking and vaping</option>
              </select>
            </div>

            {formData.smokingStatus && formData.smokingStatus !== 'Never smoked' && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-4 pt-4 border-t border-neutral-100"
              >
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-neutral-900 mb-2">
                      Age started smoking
                    </label>
                    <Input
                      type="number"
                      placeholder="Age"
                      value={formData.smokingStartAge}
                      onChange={(e) => handleInputChange('smokingStartAge', e.target.value)}
                      className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-neutral-900 mb-2">
                      Cigarettes per day
                    </label>
                    <Input
                      type="number"
                      placeholder="Number"
                      value={formData.cigarettesPerDay}
                      onChange={(e) => handleInputChange('cigarettesPerDay', e.target.value)}
                      className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                    />
                  </div>
                </div>

                {(formData.smokingStatus === 'Vaping only' || formData.smokingStatus === 'Both smoking and vaping') && (
                  <div className="space-y-4 pt-4 border-t border-neutral-100">
                    <div>
                      <label className="block text-sm font-semibold text-neutral-900 mb-2">
                        Vaping device
                      </label>
                      <Input
                        type="text"
                        placeholder="e.g., Pod system, Mod, Disposable"
                        value={formData.vapingDevice}
                        onChange={(e) => handleInputChange('vapingDevice', e.target.value)}
                        className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-semibold text-neutral-900 mb-2">
                          Nicotine strength (mg)
                        </label>
                        <Input
                          type="number"
                          step="0.1"
                          placeholder="mg"
                          value={formData.nicotineMg}
                          onChange={(e) => handleInputChange('nicotineMg', e.target.value)}
                          className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-semibold text-neutral-900 mb-2">
                          PG/VG ratio
                        </label>
                        <Input
                          type="text"
                          placeholder="e.g., 70/30"
                          value={formData.pgVgRatio}
                          onChange={(e) => handleInputChange('pgVgRatio', e.target.value)}
                          className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                        />
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="inline-flex p-4 bg-primary-50 rounded-3xl mb-4"
              >
                <ClipboardCheck className="w-8 h-8 text-primary-600" />
              </motion.div>
              <h3 className="text-xl font-bold text-neutral-900 mb-2">Physical Measurements</h3>
              <p className="text-neutral-600 text-sm">Your current health metrics</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-semibold text-neutral-900 mb-2">
                  Weight (kg) <span className="text-error-500">*</span>
                </label>
                <Input
                  type="number"
                  step="0.1"
                  placeholder="Weight"
                  value={formData.weight}
                  onChange={(e) => handleInputChange('weight', e.target.value)}
                  className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-neutral-900 mb-2">
                  Height (cm) <span className="text-error-500">*</span>
                </label>
                <Input
                  type="number"
                  placeholder="Height"
                  value={formData.height}
                  onChange={(e) => handleInputChange('height', e.target.value)}
                  className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-semibold text-neutral-900 mb-2">
                  Waist (cm)
                </label>
                <Input
                  type="number"
                  step="0.1"
                  placeholder="Waist"
                  value={formData.waistCircumference}
                  onChange={(e) => handleInputChange('waistCircumference', e.target.value)}
                  className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-neutral-900 mb-2">
                  Hip (cm)
                </label>
                <Input
                  type="number"
                  step="0.1"
                  placeholder="Hip"
                  value={formData.hipCircumference}
                  onChange={(e) => handleInputChange('hipCircumference', e.target.value)}
                  className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-neutral-900 mb-2">
                  Neck (cm)
                </label>
                <Input
                  type="number"
                  step="0.1"
                  placeholder="Neck"
                  value={formData.neckCircumference}
                  onChange={(e) => handleInputChange('neckCircumference', e.target.value)}
                  className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-semibold text-neutral-900 mb-2">
                  Systolic BP (mmHg)
                </label>
                <Input
                  type="number"
                  placeholder="Systolic"
                  value={formData.systolicBP}
                  onChange={(e) => handleInputChange('systolicBP', e.target.value)}
                  className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-neutral-900 mb-2">
                  Diastolic BP (mmHg)
                </label>
                <Input
                  type="number"
                  placeholder="Diastolic"
                  value={formData.diastolicBP}
                  onChange={(e) => handleInputChange('diastolicBP', e.target.value)}
                  className="bg-white border-neutral-200 text-neutral-900 placeholder-neutral-400 focus:border-primary-500 focus:ring-primary-500 rounded-xl h-12"
                />
              </div>
            </div>
          </div>
        );

      case 5:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="inline-flex p-4 bg-primary-50 rounded-3xl mb-4"
              >
                <Heart className="w-8 h-8 text-primary-600" />
              </motion.div>
              <h3 className="text-xl font-bold text-neutral-900 mb-2">Medical Conditions</h3>
              <p className="text-neutral-600 text-sm">All fields are optional</p>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="text-neutral-900 font-bold mb-3 text-sm uppercase tracking-wide">Cardiovascular</h4>
                {[
                  { key: 'ascvd', label: 'ASCVD' },
                  { key: 'htn', label: 'Hypertension' },
                  { key: 'dyslipidaemia', label: 'Dyslipidaemia' },
                  { key: 'ischaemicHeartDisease', label: 'Ischaemic Heart Disease' },
                  { key: 'heartFailure', label: 'Heart Failure' },
                  { key: 'cerebrovascularDisease', label: 'Cerebrovascular Disease' },
                  { key: 'pulmonaryHypertension', label: 'Pulmonary Hypertension' },
                  { key: 'dvt', label: 'DVT' },
                  { key: 'pe', label: 'PE' }
                ].map((condition) => (
                  <label key={condition.key} className="flex items-center space-x-3 p-4 bg-neutral-50 rounded-2xl border-2 border-neutral-200 hover:border-primary-300 cursor-pointer transition-all">
                    <input
                      type="checkbox"
                      checked={formData[condition.key]}
                      onChange={(e) => handleInputChange(condition.key, e.target.checked)}
                      className="w-5 h-5 text-primary-600 bg-white border-neutral-300 rounded focus:ring-primary-500 focus:ring-2"
                    />
                    <span className="text-neutral-900 text-sm font-medium">{condition.label}</span>
                  </label>
                ))}
              </div>

              <div className="space-y-4">
                <h4 className="text-neutral-900 font-bold mb-3 text-sm uppercase tracking-wide">Metabolic & Endocrine</h4>
                {[
                  { key: 't2dm', label: 'Type 2 Diabetes' },
                  { key: 'prediabetes', label: 'Prediabetes' },
                  { key: 'thyroidDisorder', label: 'Thyroid Disorder' },
                  { key: 'masld', label: 'MASLD' },
                  { key: 'pcos', label: 'PCOS' },
                  { key: 'infertility', label: 'Infertility' }
                ].map((condition) => (
                  <label key={condition.key} className="flex items-center space-x-3 p-4 bg-neutral-50 rounded-2xl border-2 border-neutral-200 hover:border-primary-300 cursor-pointer transition-all">
                    <input
                      type="checkbox"
                      checked={formData[condition.key]}
                      onChange={(e) => handleInputChange(condition.key, e.target.checked)}
                      className="w-5 h-5 text-primary-600 bg-white border-neutral-300 rounded focus:ring-primary-500 focus:ring-2"
                    />
                    <span className="text-neutral-900 text-sm font-medium">{condition.label}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="text-neutral-900 font-bold mb-3 text-sm uppercase tracking-wide">Respiratory & Sleep</h4>
                {[
                  { key: 'asthma', label: 'Asthma' },
                  { key: 'osa', label: 'OSA' },
                  { key: 'sleepStudies', label: 'Sleep Studies' },
                  { key: 'cpap', label: 'CPAP' }
                ].map((condition) => (
                  <label key={condition.key} className="flex items-center space-x-3 p-4 bg-neutral-50 rounded-2xl border-2 border-neutral-200 hover:border-primary-300 cursor-pointer transition-all">
                    <input
                      type="checkbox"
                      checked={formData[condition.key]}
                      onChange={(e) => handleInputChange(condition.key, e.target.checked)}
                      className="w-5 h-5 text-primary-600 bg-white border-neutral-300 rounded focus:ring-primary-500 focus:ring-2"
                    />
                    <span className="text-neutral-900 text-sm font-medium">{condition.label}</span>
                  </label>
                ))}
              </div>

              <div className="space-y-4">
                <h4 className="text-neutral-900 font-bold mb-3 text-sm uppercase tracking-wide">Mental Health</h4>
                {[
                  { key: 'anxiety', label: 'Anxiety' },
                  { key: 'depression', label: 'Depression' },
                  { key: 'bipolarDisorder', label: 'Bipolar Disorder' },
                  { key: 'emotionalEating', label: 'Emotional Eating' },
                  { key: 'schizoaffectiveDisorder', label: 'Schizoaffective Disorder' }
                ].map((condition) => (
                  <label key={condition.key} className="flex items-center space-x-3 p-4 bg-neutral-50 rounded-2xl border-2 border-neutral-200 hover:border-primary-300 cursor-pointer transition-all">
                    <input
                      type="checkbox"
                      checked={formData[condition.key]}
                      onChange={(e) => handleInputChange(condition.key, e.target.checked)}
                      className="w-5 h-5 text-primary-600 bg-white border-neutral-300 rounded focus:ring-primary-500 focus:ring-2"
                    />
                    <span className="text-neutral-900 text-sm font-medium">{condition.label}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="text-neutral-900 font-bold mb-3 text-sm uppercase tracking-wide">Musculoskeletal</h4>
                {[
                  { key: 'oaKnee', label: 'OA Knee' },
                  { key: 'oaHip', label: 'OA Hip' },
                  { key: 'limitedMobility', label: 'Limited Mobility' },
                  { key: 'lymphoedema', label: 'Lymphoedema' }
                ].map((condition) => (
                  <label key={condition.key} className="flex items-center space-x-3 p-4 bg-neutral-50 rounded-2xl border-2 border-neutral-200 hover:border-primary-300 cursor-pointer transition-all">
                    <input
                      type="checkbox"
                      checked={formData[condition.key]}
                      onChange={(e) => handleInputChange(condition.key, e.target.checked)}
                      className="w-5 h-5 text-primary-600 bg-white border-neutral-300 rounded focus:ring-primary-500 focus:ring-2"
                    />
                    <span className="text-neutral-900 text-sm font-medium">{condition.label}</span>
                  </label>
                ))}
              </div>

              <div className="space-y-4">
                <h4 className="text-neutral-900 font-bold mb-3 text-sm uppercase tracking-wide">Other Conditions</h4>
                {[
                  { key: 'gord', label: 'GORD' },
                  { key: 'ckd', label: 'CKD' },
                  { key: 'kidneyStones', label: 'Kidney Stones' },
                  { key: 'iih', label: 'IIH' },
                  { key: 'epilepsy', label: 'Epilepsy' },
                  { key: 'functionalNeurologicalDisorder', label: 'Functional Neurological Disorder' },
                  { key: 'cancer', label: 'Cancer' }
                ].map((condition) => (
                  <label key={condition.key} className="flex items-center space-x-3 p-4 bg-neutral-50 rounded-2xl border-2 border-neutral-200 hover:border-primary-300 cursor-pointer transition-all">
                    <input
                      type="checkbox"
                      checked={formData[condition.key]}
                      onChange={(e) => handleInputChange(condition.key, e.target.checked)}
                      className="w-5 h-5 text-primary-600 bg-white border-neutral-300 rounded focus:ring-primary-500 focus:ring-2"
                    />
                    <span className="text-neutral-900 text-sm font-medium">{condition.label}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="space-y-4 pt-4 border-t border-neutral-100">
              <h4 className="text-neutral-900 font-bold mb-3 text-sm uppercase tracking-wide">Bariatric Surgery (if applicable)</h4>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { key: 'bariatricGastricBand', label: 'Gastric Band' },
                  { key: 'bariatricSleeve', label: 'Sleeve' },
                  { key: 'bariatricBypass', label: 'Bypass' },
                  { key: 'bariatricBalloon', label: 'Balloon' }
                ].map((condition) => (
                  <label key={condition.key} className="flex items-center space-x-3 p-4 bg-neutral-50 rounded-2xl border-2 border-neutral-200 hover:border-primary-300 cursor-pointer transition-all">
                    <input
                      type="checkbox"
                      checked={formData[condition.key]}
                      onChange={(e) => handleInputChange(condition.key, e.target.checked)}
                      className="w-5 h-5 text-primary-600 bg-white border-neutral-300 rounded focus:ring-primary-500 focus:ring-2"
                    />
                    <span className="text-neutral-900 text-sm font-medium">{condition.label}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );

      case 6:
        return (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="inline-flex p-4 bg-medical-50 rounded-3xl mb-4"
              >
                <Check className="w-8 h-8 text-medical-600" />
              </motion.div>
              <h3 className="text-2xl font-bold text-neutral-900 mb-2">Review Your Information</h3>
              <p className="text-neutral-600">Please review all your responses before submitting</p>
            </div>

            <div className="space-y-4 max-h-96 overflow-y-auto">
              <div className="bg-neutral-50 rounded-2xl p-6 border border-neutral-100">
                <h4 className="text-neutral-900 font-bold mb-3">Basic Information</h4>
                <div className="text-sm text-neutral-700 space-y-2">
                  <p><span className="text-gray-500 font-medium">Name:</span> {formData.name}</p>
                  <p><span className="text-gray-500 font-medium">Date of Birth:</span> {formData.dateOfBirth}</p>
                  <p><span className="text-gray-500 font-medium">Biological Sex:</span> {formData.biologicalSex}</p>
                  <p><span className="text-gray-500 font-medium">Ethnicity:</span> {formData.ethnicity}</p>
                </div>
              </div>

              {formData.biologicalSex === 'Female' && formData.hasMenses !== null && (
                <div className="bg-neutral-50 rounded-2xl p-6 border border-neutral-100">
                  <h4 className="text-neutral-900 font-bold mb-3">Female Health Information</h4>
                  <div className="text-sm text-neutral-700 space-y-2">
                    <p><span className="text-gray-500 font-medium">Has Menses:</span> {formData.hasMenses ? 'Yes' : 'No'}</p>
                    {formData.hasMenses && (
                      <>
                        <p><span className="text-gray-500 font-medium">Age at Menarche:</span> {formData.ageAtMenarche} years</p>
                        <p><span className="text-gray-500 font-medium">Menstrual Regularity:</span> {formData.menstrualRegularity}</p>
                        <p><span className="text-gray-500 font-medium">Last Period:</span> {formData.lastMenstrualPeriod}</p>
                      </>
                    )}
                  </div>
                </div>
              )}

              {formData.biologicalSex === 'Male' && (
                <div className="bg-neutral-50 rounded-2xl p-6 border border-neutral-100">
                  <h4 className="text-neutral-900 font-bold mb-3">Male Health Information</h4>
                  <div className="text-sm text-neutral-700 space-y-2">
                    <p><span className="text-gray-500 font-medium">IIEF-5 Score:</span> {formData.iiefScores.reduce((sum, score) => sum + (score || 0), 0)}</p>
                    <p><span className="text-gray-500 font-medium">Low T Symptoms:</span> {formData.lowTestosteroneSymptoms.join(', ') || 'None'}</p>
                  </div>
                </div>
              )}

              <div className="bg-neutral-50 rounded-2xl p-6 border border-neutral-100">
                <h4 className="text-neutral-900 font-bold mb-3">Lifestyle</h4>
                <div className="text-sm text-neutral-700 space-y-2">
                  <p><span className="text-gray-500 font-medium">AUDIT Score:</span> {formData.auditScores.reduce((sum, score) => sum + (score || 0), 0)}</p>
                  <p><span className="text-gray-500 font-medium">Smoking Status:</span> {formData.smokingStatus}</p>
                </div>
              </div>

              <div className="bg-neutral-50 rounded-2xl p-6 border border-neutral-100">
                <h4 className="text-neutral-900 font-bold mb-3">Physical Measurements</h4>
                <div className="text-sm text-neutral-700 space-y-2">
                  <p><span className="text-gray-500 font-medium">Weight:</span> {formData.weight} kg</p>
                  <p><span className="text-gray-500 font-medium">Height:</span> {formData.height} cm</p>
                  {formData.waistCircumference && <p><span className="text-gray-500 font-medium">Waist:</span> {formData.waistCircumference} cm</p>}
                  {formData.systolicBP && <p><span className="text-gray-500 font-medium">Blood Pressure:</span> {formData.systolicBP}/{formData.diastolicBP} mmHg</p>}
                </div>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4"
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="bg-white rounded-3xl shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col"
          >
            {/* Header - Landing Page Style */}
            <div className="bg-gradient-to-r from-primary-600 to-primary-500 px-8 py-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold text-white mb-1">Complete Your Health Profile</h2>
                  <p className="text-primary-100 text-sm">
                    Step {currentStep} of {totalSteps}: {steps[currentStep - 1].title}
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-white text-sm font-semibold mb-2">
                    {Math.round((currentStep / totalSteps) * 100)}% Complete
                  </div>
                  <div className="w-40 bg-white/20 rounded-full h-2.5">
                    <motion.div 
                      className="bg-white rounded-full h-2.5 transition-all duration-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${(currentStep / totalSteps) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Progress Steps - Modern Design */}
            <div className="px-8 py-6 bg-neutral-50 border-b border-neutral-100">
              <div className="flex items-center justify-between">
                {steps.map((step, index) => {
                  const Icon = step.icon;
                  const isActive = currentStep === step.number;
                  const isCompleted = currentStep > step.number;
                  
                  return (
                    <React.Fragment key={step.number}>
                      <div className="flex flex-col items-center flex-1">
                        <motion.div
                          className={`
                            w-12 h-12 rounded-2xl flex items-center justify-center text-sm font-semibold transition-all duration-300
                            ${isCompleted 
                              ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/30' 
                              : isActive
                              ? 'bg-primary-100 text-primary-700 border-2 border-primary-500'
                              : 'bg-white text-neutral-400 border-2 border-neutral-200'
                            }
                          `}
                          whileHover={{ scale: 1.05 }}
                        >
                          {isCompleted ? <Check className="w-5 h-5" /> : <Icon className="w-5 h-5" />}
                        </motion.div>
                        <div className="mt-2 text-center">
                          <p className={`text-xs font-medium ${isActive ? 'text-primary-700' : 'text-gray-500'}`}>
                            Step {step.number}
                          </p>
                          <p className={`text-xs mt-0.5 ${isActive ? 'text-neutral-900 font-semibold' : 'text-neutral-400'}`}>
                            {step.title}
                          </p>
                        </div>
                      </div>
                      {index < steps.length - 1 && (
                        <div className={`
                          flex-1 h-0.5 mx-4 transition-all duration-300
                          ${isCompleted ? 'bg-primary-600' : 'bg-gray-200'}
                        `} />
                      )}
                    </React.Fragment>
                  );
                })}
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-8 bg-white">
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-6 p-4 bg-error-50 border border-error-200 rounded-2xl text-error-700 text-sm"
                >
                  {error}
                </motion.div>
              )}

              <AnimatePresence mode="wait">
                <motion.div
                  key={currentStep}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  {renderStepContent()}
                </motion.div>
              </AnimatePresence>
            </div>

            {/* Navigation */}
            <div className="px-8 py-6 bg-neutral-50 border-t border-neutral-100">
              <div className="flex justify-between items-center">
                <Button
                  onClick={handlePrevious}
                  disabled={currentStep === 1}
                  variant="outline"
                  className="px-6 py-3 rounded-xl font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 border-neutral-200 hover:bg-gray-100"
                >
                  <ChevronLeft className="w-4 h-4" />
                  Previous
                </Button>

                {currentStep < totalSteps ? (
                  <Button
                    onClick={handleNext}
                    className="px-8 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg shadow-primary-600/25 hover:shadow-xl flex items-center gap-2"
                  >
                    Next
                    <ChevronRight className="w-4 h-4" />
                  </Button>
                ) : (
                  <Button
                    onClick={handleSubmit}
                    disabled={isLoading}
                    className="px-8 py-3 bg-medical-600 hover:bg-medical-700 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg shadow-medical-600/25 hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        Completing...
                      </>
                    ) : (
                      <>
                        Complete Profile
                        <Check className="w-4 h-4" />
                      </>
                    )}
                  </Button>
                )}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default PostSignupSurvey;

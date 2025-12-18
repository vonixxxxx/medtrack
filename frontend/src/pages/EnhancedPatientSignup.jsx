import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import api from '../api';

const EnhancedPatientSignup = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const [formData, setFormData] = useState({
    // Basic account info
    email: '',
    password: '',
    hospitalCode: '',
    role: 'patient',
    
    // Basic demographics
    name: '',
    dob: '',
    sex: '',
    ethnic_group: '',
    location: '',
    postcode: '',
    nhs_number: '',
    
    // Clinical measurements
    height: '',
    baseline_weight: '',
    baseline_weight_date: '',
    
    // Diabetes screening
    has_diabetes: false,
    diabetes_type: '',
    baseline_hba1c: '',
    baseline_hba1c_date: '',
    
    // Cardiovascular risk factors
    ascvd: false,
    htn: false,
    dyslipidaemia: false,
    osa: false,
    sleep_studies: false,
    cpap: false,
    t2dm: false,
    prediabetes: false,
    
    // Medical conditions
    asthma: false,
    hypertension: false,
    ischaemic_heart_disease: false,
    heart_failure: false,
    cerebrovascular_disease: false,
    pulmonary_hypertension: false,
    dvt: false,
    pe: false,
    gord: false,
    ckd: false,
    kidney_stones: false,
    masld: false,
    infertility: false,
    pcos: false,
    anxiety: false,
    depression: false,
    bipolar_disorder: false,
    emotional_eating: false,
    schizoaffective_disorder: false,
    oa_knee: false,
    oa_hip: false,
    limited_mobility: false,
    lymphoedema: false,
    thyroid_disorder: false,
    iih: false,
    epilepsy: false,
    functional_neurological_disorder: false,
    cancer: false,
    bariatric_gastric_band: false,
    bariatric_sleeve: false,
    bariatric_bypass: false,
    bariatric_balloon: false,
    
    // Notes
    notes: '',
    criteria_for_wegovy: ''
  });

  const steps = [
    { id: 1, title: 'Account Information', fields: ['email', 'password', 'hospitalCode'] },
    { id: 2, title: 'Basic Demographics', fields: ['name', 'dob', 'sex', 'ethnic_group', 'location', 'postcode', 'nhs_number'] },
    { id: 3, title: 'Clinical Measurements', fields: ['height', 'baseline_weight', 'baseline_weight_date'] },
    { id: 4, title: 'Diabetes Screening', fields: ['has_diabetes', 'diabetes_type', 'baseline_hba1c', 'baseline_hba1c_date'] },
    { id: 5, title: 'Medical Conditions', fields: ['ascvd', 'htn', 'dyslipidaemia', 'osa', 'sleep_studies', 'cpap', 't2dm', 'prediabetes'] },
    { id: 6, title: 'Additional Conditions', fields: ['asthma', 'hypertension', 'ischaemic_heart_disease', 'heart_failure', 'cerebrovascular_disease', 'pulmonary_hypertension', 'dvt', 'pe', 'gord', 'ckd', 'kidney_stones', 'masld', 'infertility', 'pcos', 'anxiety', 'depression', 'bipolar_disorder', 'emotional_eating', 'schizoaffective_disorder', 'oa_knee', 'oa_hip', 'limited_mobility', 'lymphoedema', 'thyroid_disorder', 'iih', 'epilepsy', 'functional_neurological_disorder', 'cancer', 'bariatric_gastric_band', 'bariatric_sleeve', 'bariatric_bypass', 'bariatric_balloon'] },
    { id: 7, title: 'Notes & Criteria', fields: ['notes', 'criteria_for_wegovy'] }
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleCheckboxChange = (field, checked) => {
    setFormData(prev => ({
      ...prev,
      [field]: checked
    }));
  };

  const validateStep = (step) => {
    const requiredFields = {
      1: ['email', 'password', 'hospitalCode'],
      2: ['name', 'dob', 'sex'],
      3: [],
      4: [],
      5: [],
      6: [],
      7: []
    };

    const fields = requiredFields[step] || [];
    return fields.every(field => formData[field] && formData[field].toString().trim() !== '');
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => Math.min(prev + 1, steps.length));
    } else {
      setError('Please fill in all required fields');
    }
  };

  const handlePrev = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1));
  };

  const handleSubmit = async () => {
    // Validate hospital code FIRST - before any other validation
    if (!formData.hospitalCode || formData.hospitalCode.trim() === '') {
      setError('Hospital code is required');
      return;
    }

    if (formData.hospitalCode.trim() !== '123456789') {
      setError('Invalid hospital code');
      setIsLoading(false);
      return;
    }

    if (!validateStep(currentStep)) {
      setError('Please fill in all required fields');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      // Calculate BMI if height and weight are provided
      let calculatedBMI = null;
      if (formData.height && formData.baseline_weight) {
        const heightInMeters = parseFloat(formData.height) / 100;
        const weightInKg = parseFloat(formData.baseline_weight);
        calculatedBMI = (weightInKg / (heightInMeters * heightInMeters)).toFixed(1);
      }

      // Prepare patient data
      const patientData = {
        // Basic demographics
        dob: formData.dob ? new Date(formData.dob) : null,
        sex: formData.sex,
        ethnic_group: formData.ethnic_group,
        location: formData.location,
        postcode: formData.postcode,
        nhs_number: formData.nhs_number,
        
        // Clinical measurements
        height: formData.height ? parseFloat(formData.height) : null,
        baseline_weight: formData.baseline_weight ? parseFloat(formData.baseline_weight) : null,
        baseline_bmi: calculatedBMI ? parseFloat(calculatedBMI) : null,
        baseline_weight_date: formData.baseline_weight_date ? new Date(formData.baseline_weight_date) : null,
        
        // Diabetes
        diabetes_type: formData.diabetes_type,
        baseline_hba1c: formData.baseline_hba1c ? parseFloat(formData.baseline_hba1c) : null,
        baseline_hba1c_date: formData.baseline_hba1c_date ? new Date(formData.baseline_hba1c_date) : null,
        
        // Boolean fields
        ascvd: formData.ascvd,
        htn: formData.htn,
        dyslipidaemia: formData.dyslipidaemia,
        osa: formData.osa,
        sleep_studies: formData.sleep_studies,
        cpap: formData.cpap,
        t2dm: formData.t2dm,
        prediabetes: formData.prediabetes,
        asthma: formData.asthma,
        hypertension: formData.hypertension,
        ischaemic_heart_disease: formData.ischaemic_heart_disease,
        heart_failure: formData.heart_failure,
        cerebrovascular_disease: formData.cerebrovascular_disease,
        pulmonary_hypertension: formData.pulmonary_hypertension,
        dvt: formData.dvt,
        pe: formData.pe,
        gord: formData.gord,
        ckd: formData.ckd,
        kidney_stones: formData.kidney_stones,
        masld: formData.masld,
        infertility: formData.infertility,
        pcos: formData.pcos,
        anxiety: formData.anxiety,
        depression: formData.depression,
        bipolar_disorder: formData.bipolar_disorder,
        emotional_eating: formData.emotional_eating,
        schizoaffective_disorder: formData.schizoaffective_disorder,
        oa_knee: formData.oa_knee,
        oa_hip: formData.oa_hip,
        limited_mobility: formData.limited_mobility,
        lymphoedema: formData.lymphoedema,
        thyroid_disorder: formData.thyroid_disorder,
        iih: formData.iih,
        epilepsy: formData.epilepsy,
        functional_neurological_disorder: formData.functional_neurological_disorder,
        cancer: formData.cancer,
        bariatric_gastric_band: formData.bariatric_gastric_band,
        bariatric_sleeve: formData.bariatric_sleeve,
        bariatric_bypass: formData.bariatric_bypass,
        bariatric_balloon: formData.bariatric_balloon,
        
        // Notes
        notes: formData.notes,
        criteria_for_wegovy: formData.criteria_for_wegovy
      };

      // Create user and patient
      const { data } = await api.post('auth/signup', {
        email: formData.email,
        password: formData.password,
        role: formData.role,
        hospitalCode: formData.hospitalCode.trim(),
        patientData: patientData
      });

      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
      
      navigate('/dashboard/patient');
    } catch (err) {
      setError(err.response?.data?.error || 'Signup failed');
    } finally {
      setIsLoading(false);
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-white mb-4">Account Information</h3>
            
            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Email Address *
              </label>
              <input
                type="email"
                value={formData.email}
                onChange={(e) => handleInputChange('email', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                placeholder="Enter your email"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Password *
              </label>
              <input
                type="password"
                value={formData.password}
                onChange={(e) => handleInputChange('password', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                placeholder="Create a password"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Hospital Code *
              </label>
              <input
                type="text"
                value={formData.hospitalCode}
                onChange={(e) => handleInputChange('hospitalCode', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                placeholder="Enter hospital code"
                required
              />
              <p className="text-xs text-gray-400 mt-1">
                Required for all accounts. Contact your institution if you don't have a code.
              </p>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-white mb-4">Basic Demographics</h3>
            
            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Full Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => handleInputChange('name', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                placeholder="Enter your full name"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Date of Birth *
              </label>
              <input
                type="date"
                value={formData.dob}
                onChange={(e) => handleInputChange('dob', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-2">
                Sex *
              </label>
              <select
                value={formData.sex}
                onChange={(e) => handleInputChange('sex', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
                required
              >
                <option value="">Select sex</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
                <option value="Prefer not to say">Prefer not to say</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Ethnic Group <span className="text-gray-500">(optional)</span>
              </label>
              <select
                value={formData.ethnic_group}
                onChange={(e) => handleInputChange('ethnic_group', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
              >
                <option value="">Select ethnic group</option>
                <option value="White British">White British</option>
                <option value="White Irish">White Irish</option>
                <option value="White Other">White Other</option>
                <option value="Mixed White and Black Caribbean">Mixed White and Black Caribbean</option>
                <option value="Mixed White and Black African">Mixed White and Black African</option>
                <option value="Mixed White and Asian">Mixed White and Asian</option>
                <option value="Mixed Other">Mixed Other</option>
                <option value="Asian or Asian British Indian">Asian or Asian British Indian</option>
                <option value="Asian or Asian British Pakistani">Asian or Asian British Pakistani</option>
                <option value="Asian or Asian British Bangladeshi">Asian or Asian British Bangladeshi</option>
                <option value="Asian or Asian British Other">Asian or Asian British Other</option>
                <option value="Black or Black British Caribbean">Black or Black British Caribbean</option>
                <option value="Black or Black British African">Black or Black British African</option>
                <option value="Black or Black British Other">Black or Black British Other</option>
                <option value="Chinese">Chinese</option>
                <option value="Other">Other</option>
                <option value="Prefer not to say">Prefer not to say</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Location <span className="text-gray-500">(optional)</span>
              </label>
              <input
                type="text"
                value={formData.location}
                onChange={(e) => handleInputChange('location', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                placeholder="Enter your location"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Postcode <span className="text-gray-500">(optional)</span>
              </label>
              <input
                type="text"
                value={formData.postcode}
                onChange={(e) => handleInputChange('postcode', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                placeholder="Enter your postcode"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                NHS Number <span className="text-gray-500">(optional)</span>
              </label>
              <input
                type="text"
                value={formData.nhs_number}
                onChange={(e) => handleInputChange('nhs_number', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                placeholder="Enter your NHS number if known"
              />
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-white mb-4">Clinical Measurements</h3>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Height (cm) <span className="text-gray-500">(optional)</span>
              </label>
              <input
                type="number"
                value={formData.height}
                onChange={(e) => handleInputChange('height', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                placeholder="Enter your height in cm"
                min="100"
                max="250"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Baseline Weight (kg) <span className="text-gray-500">(optional)</span>
              </label>
              <input
                type="number"
                value={formData.baseline_weight}
                onChange={(e) => handleInputChange('baseline_weight', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                placeholder="Enter your baseline weight in kg"
                min="20"
                max="300"
                step="0.1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Baseline Weight Date <span className="text-gray-500">(optional)</span>
              </label>
              <input
                type="date"
                value={formData.baseline_weight_date}
                onChange={(e) => handleInputChange('baseline_weight_date', e.target.value)}
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
              />
            </div>

            {formData.height && formData.baseline_weight && (
              <div className="p-4 bg-blue-900/20 border border-blue-800 rounded-xl">
                <p className="text-blue-300 text-sm">
                  <strong>Calculated BMI:</strong> {((parseFloat(formData.baseline_weight) / Math.pow(parseFloat(formData.height) / 100, 2)).toFixed(1))}
                </p>
              </div>
            )}
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-white mb-4">Diabetes Screening</h3>
            
            <div>
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={formData.has_diabetes}
                  onChange={(e) => handleCheckboxChange('has_diabetes', e.target.checked)}
                  className="w-4 h-4 text-white bg-gray-800 border-gray-600 rounded focus:ring-white focus:ring-2"
                />
                <span className="text-white">Do you have diabetes?</span>
              </label>
            </div>

            {formData.has_diabetes && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Diabetes Type <span className="text-gray-500">(optional)</span>
                  </label>
                  <select
                    value={formData.diabetes_type}
                    onChange={(e) => handleInputChange('diabetes_type', e.target.value)}
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
                  >
                    <option value="">Select diabetes type</option>
                    <option value="Type 1">Type 1</option>
                    <option value="Type 2">Type 2</option>
                    <option value="Gestational">Gestational</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Baseline HbA1c (%) <span className="text-gray-500">(optional)</span>
                  </label>
                  <input
                    type="number"
                    value={formData.baseline_hba1c}
                    onChange={(e) => handleInputChange('baseline_hba1c', e.target.value)}
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                    placeholder="Enter your baseline HbA1c"
                    min="0"
                    max="20"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Baseline HbA1c Date <span className="text-gray-500">(optional)</span>
                  </label>
                  <input
                    type="date"
                    value={formData.baseline_hba1c_date}
                    onChange={(e) => handleInputChange('baseline_hba1c_date', e.target.value)}
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white"
                  />
                </div>
              </>
            )}
          </div>
        );

      case 5:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-white mb-4">Key Medical Conditions</h3>
            <p className="text-gray-400 text-sm mb-6">Please select any conditions that apply to you:</p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { key: 'ascvd', label: 'ASCVD (Atherosclerotic Cardiovascular Disease)' },
                { key: 'htn', label: 'Hypertension (HTN)' },
                { key: 'dyslipidaemia', label: 'Dyslipidaemia' },
                { key: 'osa', label: 'Obstructive Sleep Apnoea (OSA)' },
                { key: 'sleep_studies', label: 'Sleep Studies' },
                { key: 'cpap', label: 'CPAP' },
                { key: 't2dm', label: 'Type 2 Diabetes Mellitus (T2DM)' },
                { key: 'prediabetes', label: 'Prediabetes' }
              ].map((condition) => (
                <label key={condition.key} className="flex items-center space-x-3 p-3 bg-gray-800 rounded-xl hover:bg-gray-700 transition-colors">
                  <input
                    type="checkbox"
                    checked={formData[condition.key]}
                    onChange={(e) => handleCheckboxChange(condition.key, e.target.checked)}
                    className="w-4 h-4 text-white bg-gray-800 border-gray-600 rounded focus:ring-white focus:ring-2"
                  />
                  <span className="text-white text-sm">{condition.label}</span>
                </label>
              ))}
            </div>
          </div>
        );

      case 6:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-white mb-4">Additional Medical Conditions</h3>
            <p className="text-gray-400 text-sm mb-6">Please select any additional conditions that apply to you:</p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { key: 'asthma', label: 'Asthma' },
                { key: 'hypertension', label: 'Hypertension' },
                { key: 'ischaemic_heart_disease', label: 'Ischaemic Heart Disease' },
                { key: 'heart_failure', label: 'Heart Failure' },
                { key: 'cerebrovascular_disease', label: 'Cerebrovascular Disease' },
                { key: 'pulmonary_hypertension', label: 'Pulmonary Hypertension' },
                { key: 'dvt', label: 'Deep Vein Thrombosis (DVT)' },
                { key: 'pe', label: 'Pulmonary Embolism (PE)' },
                { key: 'gord', label: 'Gastro-oesophageal Reflux Disease (GORD)' },
                { key: 'ckd', label: 'Chronic Kidney Disease (CKD)' },
                { key: 'kidney_stones', label: 'Kidney Stones' },
                { key: 'masld', label: 'Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD)' },
                { key: 'infertility', label: 'Infertility' },
                { key: 'pcos', label: 'Polycystic Ovary Syndrome (PCOS)' },
                { key: 'anxiety', label: 'Anxiety' },
                { key: 'depression', label: 'Depression' },
                { key: 'bipolar_disorder', label: 'Bipolar Disorder' },
                { key: 'emotional_eating', label: 'Emotional Eating' },
                { key: 'schizoaffective_disorder', label: 'Schizoaffective Disorder' },
                { key: 'oa_knee', label: 'Osteoarthritis Knee' },
                { key: 'oa_hip', label: 'Osteoarthritis Hip' },
                { key: 'limited_mobility', label: 'Limited Mobility (due to obesity)' },
                { key: 'lymphoedema', label: 'Lymphoedema' },
                { key: 'thyroid_disorder', label: 'Thyroid Disorder' },
                { key: 'iih', label: 'Idiopathic Intracranial Hypertension (IIH)' },
                { key: 'epilepsy', label: 'Epilepsy' },
                { key: 'functional_neurological_disorder', label: 'Functional Neurological Disorder' },
                { key: 'cancer', label: 'Cancer' },
                { key: 'bariatric_gastric_band', label: 'Bariatric - Gastric Band' },
                { key: 'bariatric_sleeve', label: 'Bariatric - Sleeve' },
                { key: 'bariatric_bypass', label: 'Bariatric - Bypass' },
                { key: 'bariatric_balloon', label: 'Bariatric - Balloon' }
              ].map((condition) => (
                <label key={condition.key} className="flex items-center space-x-3 p-3 bg-gray-800 rounded-xl hover:bg-gray-700 transition-colors">
                  <input
                    type="checkbox"
                    checked={formData[condition.key]}
                    onChange={(e) => handleCheckboxChange(condition.key, e.target.checked)}
                    className="w-4 h-4 text-white bg-gray-800 border-gray-600 rounded focus:ring-white focus:ring-2"
                  />
                  <span className="text-white text-sm">{condition.label}</span>
                </label>
              ))}
            </div>
          </div>
        );

      case 7:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-white mb-4">Notes & Criteria</h3>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Additional Notes <span className="text-gray-500">(optional)</span>
              </label>
              <textarea
                value={formData.notes}
                onChange={(e) => handleInputChange('notes', e.target.value)}
                className="w-full h-32 px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white resize-none"
                placeholder="Enter any additional notes about your health..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Criteria for Wegovy <span className="text-gray-500">(optional)</span>
              </label>
              <textarea
                value={formData.criteria_for_wegovy}
                onChange={(e) => handleInputChange('criteria_for_wegovy', e.target.value)}
                className="w-full h-32 px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white resize-none"
                placeholder="Enter any criteria for Wegovy treatment..."
              />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center py-12">
      <div className="w-full max-w-4xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-900 rounded-3xl border border-gray-800 p-8"
        >
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Enhanced Patient Registration</h1>
            <p className="text-gray-400">Complete your medical profile for comprehensive care</p>
          </div>

          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-4">
              {steps.map((step) => (
                <div key={step.id} className="flex items-center">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                      currentStep >= step.id
                        ? 'bg-white text-gray-900'
                        : 'bg-gray-700 text-gray-400'
                    }`}
                  >
                    {step.id}
                  </div>
                  {step.id < steps.length && (
                    <div
                      className={`w-16 h-1 mx-2 ${
                        currentStep > step.id ? 'bg-white' : 'bg-gray-700'
                      }`}
                    />
                  )}
                </div>
              ))}
            </div>
            <div className="text-center">
              <p className="text-white font-medium">{steps[currentStep - 1].title}</p>
              <p className="text-gray-400 text-sm">Step {currentStep} of {steps.length}</p>
            </div>
          </div>

          {/* Form Content */}
          <div className="mb-8">
            {renderStepContent()}
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-xl text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="flex justify-between">
            <button
              onClick={handlePrev}
              disabled={currentStep === 1}
              className="px-6 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>

            {currentStep < steps.length ? (
              <button
                onClick={handleNext}
                className="px-6 py-3 bg-white text-gray-900 rounded-xl hover:bg-gray-100 transition-colors font-medium"
              >
                Next
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={isLoading}
                className="px-6 py-3 bg-white text-gray-900 rounded-xl hover:bg-gray-100 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Creating Account...' : 'Complete Registration'}
              </button>
            )}
          </div>

          {/* Back to Login */}
          <div className="text-center mt-6">
            <p className="text-gray-400">
              Already have an account?{' '}
              <button
                onClick={() => navigate('/login')}
                className="text-white hover:underline"
              >
                Sign in
              </button>
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default EnhancedPatientSignup;

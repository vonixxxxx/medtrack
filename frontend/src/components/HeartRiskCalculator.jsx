import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Heart, AlertTriangle, CheckCircle, Info, TrendingUp, Calculator } from 'lucide-react';

const HeartRiskCalculator = ({ onSubmit, initialData = null }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState(initialData || {
    age: '',
    gender: '',
    race: 'white',
    totalCholesterol: '',
    hdlCholesterol: '',
    systolicBP: '',
    isOnBPMedication: false,
    isSmoker: false,
    hasDiabetes: false,
    calculationMethod: 'ascvd' // 'ascvd' or 'framingham'
  });
  const [isCompleted, setIsCompleted] = useState(false);
  const [riskResult, setRiskResult] = useState(null);

  const steps = [
    {
      id: 'demographics',
      title: 'Demographics',
      description: 'Basic information for risk calculation'
    },
    {
      id: 'cholesterol',
      title: 'Cholesterol Profile',
      description: 'Your cholesterol levels'
    },
    {
      id: 'bloodPressure',
      title: 'Blood Pressure',
      description: 'Blood pressure and medication status'
    },
    {
      id: 'lifestyle',
      title: 'Lifestyle & Health',
      description: 'Smoking and diabetes status'
    },
    {
      id: 'calculation',
      title: 'Risk Calculation',
      description: 'Choose calculation method and view results'
    }
  ];

  const getRiskColor = (riskCategory) => {
    if (riskCategory?.includes('Low')) return 'text-green-600';
    if (riskCategory?.includes('Borderline') || riskCategory?.includes('Intermediate')) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getRiskBgColor = (riskCategory) => {
    if (riskCategory?.includes('Low')) return 'bg-green-50';
    if (riskCategory?.includes('Borderline') || riskCategory?.includes('Intermediate')) return 'bg-yellow-50';
    return 'bg-red-50';
  };

  const getRiskIcon = (riskCategory) => {
    if (riskCategory?.includes('Low')) return CheckCircle;
    if (riskCategory?.includes('Borderline') || riskCategory?.includes('Intermediate')) return AlertTriangle;
    return AlertTriangle;
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const calculateRisk = async () => {
    try {
      const response = await fetch('/api/screening/heart-risk', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          ...formData,
          age: parseInt(formData.age),
          totalCholesterol: parseInt(formData.totalCholesterol),
          hdlCholesterol: parseInt(formData.hdlCholesterol),
          systolicBP: parseInt(formData.systolicBP)
        })
      });

      if (response.ok) {
        const result = await response.json();
        setRiskResult(result);
        setIsCompleted(true);
        
        if (onSubmit) {
          onSubmit({
            ...formData,
            result: result,
            completedAt: new Date().toISOString()
          });
        }
      } else {
        console.error('Failed to calculate risk');
      }
    } catch (error) {
      console.error('Error calculating risk:', error);
    }
  };

  const canProceed = () => {
    switch (currentStep) {
      case 0: // Demographics
        return formData.age && formData.gender;
      case 1: // Cholesterol
        return formData.totalCholesterol && formData.hdlCholesterol;
      case 2: // Blood Pressure
        return formData.systolicBP !== '';
      case 3: // Lifestyle
        return true; // All fields are optional
      default:
        return true;
    }
  };

  const progress = ((currentStep + 1) / steps.length) * 100;

  if (isCompleted) {
    const IconComponent = getRiskIcon(riskResult.riskCategory);
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-2xl mx-auto p-6 bg-white rounded-xl shadow-lg"
      >
        <div className="text-center mb-6">
          <IconComponent className={`w-16 h-16 mx-auto mb-4 ${getRiskColor(riskResult.riskCategory)}`} />
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Heart Risk Assessment Complete</h2>
          <p className="text-gray-600">Your cardiovascular risk assessment results</p>
        </div>

        <div className={`p-6 rounded-xl ${getRiskBgColor(riskResult.riskCategory)} mb-6`}>
          <div className="text-center">
            <div className="text-4xl font-bold mb-2">{riskResult.riskPercentage}%</div>
            <div className={`text-xl font-semibold ${getRiskColor(riskResult.riskCategory)}`}>
              {riskResult.riskCategory}
            </div>
            <p className="text-sm text-gray-600 mt-2">
              {riskResult.calculationMethod === 'ascvd' ? 'ASCVD' : 'Framingham'} 10-year risk
            </p>
          </div>
        </div>

        {/* Risk Level Explanation */}
        <div className="bg-gray-50 p-4 rounded-lg mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">Risk Level Explanation:</h3>
          <div className="space-y-2 text-sm text-gray-600">
            {riskResult.riskCategory === 'Low Risk' && (
              <p>Your cardiovascular risk is low. Continue maintaining a healthy lifestyle with regular check-ups.</p>
            )}
            {riskResult.riskCategory === 'Borderline Risk' && (
              <p>Your risk is borderline. Consider lifestyle modifications and discuss with your healthcare provider.</p>
            )}
            {riskResult.riskCategory === 'Intermediate Risk' && (
              <p>Your risk is intermediate. Focus on lifestyle changes and consider additional testing.</p>
            )}
            {riskResult.riskCategory === 'High Risk' && (
              <p>Your risk is high. Immediate lifestyle changes and medical intervention are recommended.</p>
            )}
          </div>
        </div>

        {/* Recommendations */}
        <div className="bg-blue-50 p-4 rounded-lg mb-6">
          <h3 className="font-semibold text-blue-800 mb-2">Recommendations:</h3>
          <ul className="text-sm text-blue-700 space-y-1">
            <li>• Maintain healthy blood pressure and cholesterol levels</li>
            <li>• Exercise regularly and maintain a healthy weight</li>
            <li>• Avoid smoking and limit alcohol consumption</li>
            <li>• Follow a heart-healthy diet</li>
            <li>• Regular check-ups with your healthcare provider</li>
          </ul>
        </div>

        <div className="flex space-x-3">
          <button
            onClick={() => {
              setIsCompleted(false);
              setCurrentStep(0);
              setRiskResult(null);
            }}
            className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            Retake Assessment
          </button>
          <button
            onClick={() => window.print()}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Print Results
          </button>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-2xl mx-auto p-6 bg-white rounded-xl shadow-lg"
    >
      <div className="text-center mb-6">
        <Heart className="w-12 h-12 mx-auto mb-4 text-red-500" />
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Heart Risk Calculator</h2>
        <p className="text-gray-600">Cardiovascular risk assessment using validated clinical algorithms</p>
        <p className="text-sm text-gray-500 mt-2">
          Step {currentStep + 1} of {steps.length}: {steps[currentStep].title}
        </p>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-red-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Step Content */}
      {currentStep === 0 && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Age (years) *
            </label>
            <input
              type="number"
              min="20"
              max="79"
              value={formData.age}
              onChange={(e) => handleInputChange('age', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
              placeholder="Enter your age"
            />
            <p className="text-xs text-gray-500 mt-1">Age 20-79 years required for accurate calculation</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Gender *
            </label>
            <div className="space-y-2">
              {['male', 'female'].map((gender) => (
                <label key={gender} className="flex items-center">
                  <input
                    type="radio"
                    name="gender"
                    value={gender}
                    checked={formData.gender === gender}
                    onChange={(e) => handleInputChange('gender', e.target.value)}
                    className="mr-2 text-red-600"
                  />
                  <span className="capitalize">{gender}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Race
            </label>
            <select
              value={formData.race}
              onChange={(e) => handleInputChange('race', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
            >
              <option value="white">White</option>
              <option value="other">Other</option>
            </select>
          </div>
        </div>
      )}

      {currentStep === 1 && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Total Cholesterol (mg/dL) *
            </label>
            <input
              type="number"
              min="100"
              max="600"
              value={formData.totalCholesterol}
              onChange={(e) => handleInputChange('totalCholesterol', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
              placeholder="e.g., 200"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              HDL Cholesterol (mg/dL) *
            </label>
            <input
              type="number"
              min="20"
              max="100"
              value={formData.hdlCholesterol}
              onChange={(e) => handleInputChange('hdlCholesterol', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
              placeholder="e.g., 50"
            />
          </div>
        </div>
      )}

      {currentStep === 2 && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Systolic Blood Pressure (mmHg) *
            </label>
            <input
              type="number"
              min="80"
              max="250"
              value={formData.systolicBP}
              onChange={(e) => handleInputChange('systolicBP', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
              placeholder="e.g., 120"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Are you currently taking blood pressure medication?
            </label>
            <div className="space-y-2">
              {[true, false].map((value) => (
                <label key={value} className="flex items-center">
                  <input
                    type="radio"
                    name="bpMedication"
                    value={value}
                    checked={formData.isOnBPMedication === value}
                    onChange={(e) => handleInputChange('isOnBPMedication', e.target.value === 'true')}
                    className="mr-2 text-red-600"
                  />
                  <span>{value ? 'Yes' : 'No'}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {currentStep === 3 && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Do you currently smoke?
            </label>
            <div className="space-y-2">
              {[true, false].map((value) => (
                <label key={value} className="flex items-center">
                  <input
                    type="radio"
                    name="smoking"
                    value={value}
                    checked={formData.isSmoker === value}
                    onChange={(e) => handleInputChange('isSmoker', e.target.value === 'true')}
                    className="mr-2 text-red-600"
                  />
                  <span>{value ? 'Yes' : 'No'}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Do you have diabetes?
            </label>
            <div className="space-y-2">
              {[true, false].map((value) => (
                <label key={value} className="flex items-center">
                  <input
                    type="radio"
                    name="diabetes"
                    value={value}
                    checked={formData.hasDiabetes === value}
                    onChange={(e) => handleInputChange('hasDiabetes', e.target.value === 'true')}
                    className="mr-2 text-red-600"
                  />
                  <span>{value ? 'Yes' : 'No'}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {currentStep === 4 && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Calculation Method
            </label>
            <div className="space-y-2">
              {[
                { value: 'ascvd', label: 'ASCVD Risk Score (AHA/ACC 2013)', description: 'Recommended for ages 40-79' },
                { value: 'framingham', label: 'Framingham Risk Score', description: 'Alternative method for all ages' }
              ].map((method) => (
                <label key={method.value} className="flex items-start">
                  <input
                    type="radio"
                    name="calculationMethod"
                    value={method.value}
                    checked={formData.calculationMethod === method.value}
                    onChange={(e) => handleInputChange('calculationMethod', e.target.value)}
                    className="mr-2 mt-1 text-red-600"
                  />
                  <div>
                    <span className="font-medium">{method.label}</span>
                    <p className="text-sm text-gray-500">{method.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="flex items-start">
              <Info className="w-5 h-5 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
              <div className="text-sm text-blue-800">
                <p className="font-medium mb-1">About Heart Risk Calculation</p>
                <p>
                  This calculator uses validated clinical algorithms to estimate your 10-year risk of 
                  cardiovascular disease. Results are estimates and should be discussed with your healthcare provider.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between mt-6">
        <button
          onClick={prevStep}
          disabled={currentStep === 0}
          className={`px-4 py-2 rounded-lg transition-colors ${
            currentStep === 0
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-gray-600 text-white hover:bg-gray-700'
          }`}
        >
          Previous
        </button>

        {currentStep < steps.length - 1 ? (
          <button
            onClick={nextStep}
            disabled={!canProceed()}
            className={`px-4 py-2 rounded-lg transition-colors ${
              canProceed()
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            Next
          </button>
        ) : (
          <button
            onClick={calculateRisk}
            disabled={!canProceed()}
            className={`px-4 py-2 rounded-lg transition-colors ${
              canProceed()
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            Calculate Risk
          </button>
        )}
      </div>
    </motion.div>
  );
};

export default HeartRiskCalculator;

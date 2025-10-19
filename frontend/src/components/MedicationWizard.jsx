import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  ChevronRight, 
  ChevronLeft, 
  Search, 
  CheckCircle, 
  AlertCircle, 
  Pill, 
  Clock, 
  Activity,
  Loader2,
  ArrowRight
} from 'lucide-react';
import { cn } from '../lib/utils';
import api from '../api';

const MedicationWizard = ({ isOpen, onClose, onSuccess }) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  
  // Form data state
  const [formData, setFormData] = useState({
    medicationName: '',
    strength: '',
    unit: 'mg',
    frequency: 'daily',
    customFrequency: '',
    selectedMetrics: [],
    startDate: new Date().toISOString().split('T')[0],
    instructions: ''
  });
  
  // Validation state
  const [validation, setValidation] = useState({
    medicationName: { isValid: false, error: '', suggestions: [] },
    dosage: { isValid: false, error: '' },
    frequency: { isValid: false, error: '' },
    metrics: { isValid: false, error: '' }
  });
  
  // UI state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [selectedMedication, setSelectedMedication] = useState(null);
  
  // Available options
  const [availableMetrics, setAvailableMetrics] = useState([]);
  
  const inputRef = useRef(null);
  
  // Focus management
  useEffect(() => {
    if (isOpen && currentStep === 1) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen, currentStep]);
  
  // Load available metrics on mount
  useEffect(() => {
    if (isOpen) {
      loadAvailableMetrics();
    }
  }, [isOpen]);

  
  const loadAvailableMetrics = async () => {
    try {
      const response = await api.get('ai/metrics/options');
      setAvailableMetrics(response.data || [
        'Heart Rate', 'Blood Pressure', 'Temperature', 'Pain Level', 
        'Sleep Quality', 'Energy Level', 'Mood', 'Weight', 'Blood Sugar'
      ]);
    } catch (error) {
      console.error('Error loading metrics:', error);
      setAvailableMetrics([
        'Heart Rate', 'Blood Pressure', 'Temperature', 'Pain Level', 
        'Sleep Quality', 'Energy Level', 'Mood', 'Weight', 'Blood Sugar'
      ]);
    }
  };
  
  // Step validation
  const validateStep = async (step) => {
    switch (step) {
      case 1:
        return await validateMedicationName();
      case 2:
        return validateDosage();
      case 3:
        return validateFrequency();
      case 4:
        return validateMetrics();
      default:
        return true;
    }
  };
  
  // Step 1: Medication Name Validation
  const validateMedicationName = async () => {
    if (!formData.medicationName.trim()) {
      setValidation(prev => ({
        ...prev,
        medicationName: { isValid: false, error: 'Medication name is required', suggestions: [] }
      }));
      return false;
    }
    
    setIsValidating(true);
    try {
      const response = await api.post('ai/validate', {
        medication: formData.medicationName,
        dosage: formData.strength + formData.unit,
        frequency: formData.frequency
      });
      
      const isValid = response.data.isValid;
      const suggestions = response.data.suggestions || [];
      const warnings = response.data.warnings || [];
      
      setValidation(prev => ({
        ...prev,
        medicationName: { 
          isValid, 
          error: isValid ? '' : 'Medication not found in database', 
          suggestions: suggestions.slice(0, 3)
        }
      }));
      
      if (isValid) {
        setSelectedMedication(response.data.extracted_entities);
      }
      
      return isValid;
    } catch (error) {
      console.error('Validation error:', error);
      setValidation(prev => ({
        ...prev,
        medicationName: { isValid: false, error: 'Validation failed', suggestions: [] }
      }));
      return false;
    } finally {
      setIsValidating(false);
    }
  };
  
  // Step 2: Dosage Validation
  const validateDosage = () => {
    const { strength, unit } = formData;
    
    if (!strength || isNaN(parseFloat(strength)) || parseFloat(strength) <= 0) {
      setValidation(prev => ({
        ...prev,
        dosage: { isValid: false, error: 'Please enter a valid dosage amount' }
      }));
      return false;
    }
    
    if (!unit) {
      setValidation(prev => ({
        ...prev,
        dosage: { isValid: false, error: 'Please select a unit' }
      }));
      return false;
    }
    
    setValidation(prev => ({
      ...prev,
      dosage: { isValid: true, error: '' }
    }));
    return true;
  };
  
  // Step 3: Frequency Validation
  const validateFrequency = () => {
    const { frequency, customFrequency } = formData;
    
    if (frequency === 'custom' && !customFrequency.trim()) {
      setValidation(prev => ({
        ...prev,
        frequency: { isValid: false, error: 'Please specify custom frequency' }
      }));
      return false;
    }
    
    setValidation(prev => ({
      ...prev,
      frequency: { isValid: true, error: '' }
    }));
    return true;
  };
  
  // Step 4: Metrics Validation
  const validateMetrics = () => {
    const { selectedMetrics } = formData;
    
    if (selectedMetrics.length === 0) {
      setValidation(prev => ({
        ...prev,
        metrics: { isValid: false, error: 'Please select at least one metric to track' }
      }));
      return false;
    }
    
    setValidation(prev => ({
      ...prev,
      metrics: { isValid: true, error: '' }
    }));
    return true;
  };
  
  // Handle medication search
  const handleMedicationSearch = async (query) => {
    setSearchQuery(query);
    setFormData(prev => ({ ...prev, medicationName: query }));
    
    if (query.length < 2) {
      setSearchResults([]);
      setShowSearchResults(false);
      return;
    }
    
    try {
      const response = await api.post('ai/search-med', {
        query,
        limit: 5,
        min_confidence: 0.5
      });
      
      setSearchResults(response.data.results || []);
      setShowSearchResults(true);
      
      // Auto-fill form if parsed input contains dosage and frequency
      if (response.data.parsed_input) {
        const parsed = response.data.parsed_input;
        
        // Auto-fill dosage if detected
        if (parsed.dosage) {
          const dosageMatch = parsed.dosage.match(/(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units)/i);
          if (dosageMatch) {
            setFormData(prev => ({
              ...prev,
              strength: dosageMatch[1],
              unit: dosageMatch[2]
            }));
          }
        }
        
        // Auto-fill frequency if detected
        if (parsed.frequency) {
          let frequency = '';
          if (parsed.frequency.includes('twice') || parsed.frequency.includes('two times')) {
            frequency = 'Twice daily';
          } else if (parsed.frequency.includes('once') || parsed.frequency.includes('one time')) {
            frequency = 'Once daily';
          } else if (parsed.frequency.includes('three times') || parsed.frequency.includes('thrice')) {
            frequency = 'Three times daily';
          } else if (parsed.frequency.includes('weekly')) {
            frequency = 'Weekly';
          } else if (parsed.frequency.includes('monthly')) {
            frequency = 'Monthly';
          } else if (parsed.frequency.includes('as needed') || parsed.frequency.includes('prn')) {
            frequency = 'As needed';
          }
          
          if (frequency) {
            setFormData(prev => ({
              ...prev,
              frequency: frequency
            }));
          }
        }
      }
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
    }
  };
  
  // Handle medication selection
  const handleMedicationSelect = (medication) => {
    setFormData(prev => ({ 
      ...prev, 
      medicationName: medication.name 
    }));
    setSearchQuery(medication.name);
    setSelectedMedication(medication);
    setShowSearchResults(false);
    
    // Mark medication as valid when selected from search results
    setValidation(prev => ({
      ...prev,
      medicationName: { 
        isValid: true, 
        error: '', 
        suggestions: medication.suggestions || [],
        warnings: []
      }
    }));
  };
  
  // Handle metric selection
  const handleMetricToggle = (metric) => {
    setFormData(prev => ({
      ...prev,
      selectedMetrics: prev.selectedMetrics.includes(metric)
        ? prev.selectedMetrics.filter(m => m !== metric)
        : [...prev.selectedMetrics, metric]
    }));
  };
  
  // Navigation
  const handleNext = async () => {
    const isValid = await validateStep(currentStep);
    if (isValid && currentStep < 4) {
      setCurrentStep(prev => prev + 1);
    }
  };
  
  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
    }
  };
  
  // Submit medication
  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      const medicationData = {
        name: formData.medicationName,
        strength: `${formData.strength}${formData.unit}`,
        frequency: formData.frequency === 'custom' ? formData.customFrequency : formData.frequency,
        startDate: formData.startDate,
        instructions: formData.instructions,
        selectedMetrics: formData.selectedMetrics,
        validationStatus: 'validated'
      };
      
      const response = await api.post('meds/user', medicationData);
      
      onSuccess?.(response.data);
      onClose();
      
      // Reset form
      setFormData({
        medicationName: '',
        strength: '',
        unit: 'mg',
        frequency: 'daily',
        customFrequency: '',
        selectedMetrics: [],
        startDate: new Date().toISOString().split('T')[0],
        instructions: ''
      });
      setCurrentStep(1);
      setValidation({
        medicationName: { isValid: false, error: '', suggestions: [] },
        dosage: { isValid: false, error: '' },
        frequency: { isValid: false, error: '' },
        metrics: { isValid: false, error: '' }
      });
      
    } catch (error) {
      console.error('Error adding medication:', error);
      // Handle error (show toast, etc.)
    } finally {
      setIsSubmitting(false);
    }
  };
  
  const steps = [
    { number: 1, title: 'Medication Name', icon: Pill },
    { number: 2, title: 'Dosage & Strength', icon: Activity },
    { number: 3, title: 'Frequency', icon: Clock },
    { number: 4, title: 'Health Metrics', icon: Activity }
  ];
  
  const canProceed = () => {
    switch (currentStep) {
      case 1:
        // Allow proceeding if medication is selected OR validation is valid
        return validation.medicationName.isValid || selectedMedication !== null;
      case 2:
        return validation.dosage.isValid;
      case 3:
        return validation.frequency.isValid;
      case 4:
        return validation.metrics.isValid;
      default:
        return false;
    }
  };
  
  if (!isOpen) return null;
  
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="bg-background rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden border border-border"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-foreground to-foreground/90 text-background p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-2xl font-bold">Add New Medication</h2>
                <p className="text-background/80">Step {currentStep} of 4</p>
              </div>
              <button
                onClick={onClose}
                className="text-background/80 hover:text-background transition-colors p-2 rounded-lg hover:bg-background/10"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            
            {/* Progress Steps */}
            <div className="flex items-center space-x-2">
              {steps.map((step, index) => {
                const Icon = step.icon;
                const isActive = currentStep === step.number;
                const isCompleted = currentStep > step.number;
                
                return (
                  <React.Fragment key={step.number}>
                    <div className={cn(
                      "flex items-center space-x-2 px-3 py-2 rounded-lg transition-all",
                      isActive ? "bg-background/20" : isCompleted ? "bg-background/10" : "bg-background/5"
                    )}>
                      <div className={cn(
                        "w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold",
                        isCompleted ? "bg-green-500" : isActive ? "bg-background text-foreground" : "bg-background/20"
                      )}>
                        {isCompleted ? <CheckCircle className="w-5 h-5" /> : step.number}
                      </div>
                      <span className="text-sm font-medium">{step.title}</span>
                    </div>
                    {index < steps.length - 1 && (
                      <ChevronRight className="w-4 h-4 text-background/60" />
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
          
          {/* Content */}
          <div className="p-6">
            <AnimatePresence mode="wait">
              {/* Step 1: Medication Name */}
              {currentStep === 1 && (
                <motion.div
                  key="step1"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      What medication would you like to add?
                    </h3>
                    <p className="text-gray-600">
                      Search our database of medications to find the right one.
                    </p>
                  </div>
                  
                  <div className="relative">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                      <input
                        ref={inputRef}
                        type="text"
                        value={searchQuery}
                        onChange={(e) => handleMedicationSearch(e.target.value)}
                        placeholder="Search medications (e.g., Metformin, Aspirin)..."
                        className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                      />
                      {isValidating && (
                        <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 text-blue-500 w-5 h-5 animate-spin" />
                      )}
                    </div>
                    
                    {/* Search Results */}
                    {showSearchResults && searchResults.length > 0 && (
                      <div className="absolute z-10 w-full mt-2 bg-background border border-border rounded-xl shadow-lg max-h-60 overflow-y-auto">
                        {searchResults.map((med, index) => (
                          <div
                            key={index}
                            onClick={() => handleMedicationSelect(med)}
                            className="p-4 hover:bg-secondary/50 cursor-pointer border-b border-border last:border-b-0 transition-colors"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                  <h4 className="font-medium text-foreground">{med.name}</h4>
                                  <span className={cn(
                                    "px-2 py-1 rounded-full text-xs font-medium",
                                    med.type === 'brand' ? "bg-blue-100 text-blue-700" : "bg-green-100 text-green-700"
                                  )}>
                                    {med.type}
                                  </span>
                                </div>
                                <div className="text-sm text-muted-foreground space-y-1">
                                  {med.generic_name && med.generic_name !== med.name && (
                                    <p>Generic: {med.generic_name}</p>
                                  )}
                                  {med.brand_name && med.brand_name !== med.name && (
                                    <p>Brand: {med.brand_name}</p>
                                  )}
                                  {med.dosage_forms && med.dosage_forms.length > 0 && (
                                    <p>Form: {med.dosage_forms.join(', ')}</p>
                                  )}
                                  {med.strengths && med.strengths.length > 0 && (
                                    <p>Strengths: {med.strengths.join(', ')}</p>
                                  )}
                                  {med.manufacturer && med.manufacturer !== 'Unknown' && (
                                    <p>Manufacturer: {med.manufacturer}</p>
                                  )}
                                </div>
                                {med.suggestions && med.suggestions.length > 0 && (
                                  <div className="mt-2">
                                    <p className="text-xs text-muted-foreground">Suggestions:</p>
                                    <div className="flex flex-wrap gap-1 mt-1">
                                      {med.suggestions.slice(0, 3).map((suggestion, idx) => (
                                        <span
                                          key={idx}
                                          className="px-2 py-1 bg-muted text-muted-foreground text-xs rounded"
                                        >
                                          {suggestion}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                              <div className="text-right ml-4">
                                <div className="flex items-center gap-1">
                                  <div className={cn(
                                    "w-2 h-2 rounded-full",
                                    med.confidence >= 0.8 ? "bg-green-500" : 
                                    med.confidence >= 0.6 ? "bg-yellow-500" : "bg-orange-500"
                                  )} />
                                  <span className="text-sm font-medium text-muted-foreground">
                                    {Math.round(med.confidence * 100)}%
                                  </span>
                                </div>
                                <p className="text-xs text-muted-foreground mt-1">match</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  {/* Validation Error */}
                  {validation.medicationName.error && (
                    <div className="flex items-center space-x-2 text-red-600">
                      <AlertCircle className="w-5 h-5" />
                      <span className="text-sm">{validation.medicationName.error}</span>
                    </div>
                  )}
                  
                  {/* Suggestions */}
                  {validation.medicationName.suggestions.length > 0 && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <h4 className="font-medium text-blue-900 mb-2">Did you mean:</h4>
                      <div className="space-y-1">
                        {validation.medicationName.suggestions.map((suggestion, index) => (
                          <button
                            key={index}
                            onClick={() => handleMedicationSelect({ name: suggestion })}
                            className="block text-sm text-blue-700 hover:text-blue-900 hover:underline"
                          >
                            {suggestion}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
              
              {/* Step 2: Dosage & Strength */}
              {currentStep === 2 && (
                <motion.div
                  key="step2"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      What's the dosage and strength?
                    </h3>
                    <p className="text-gray-600">
                      Enter the amount and unit of your medication.
                    </p>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Amount
                      </label>
                      <input
                        type="number"
                        value={formData.strength}
                        onChange={(e) => setFormData(prev => ({ ...prev, strength: e.target.value }))}
                        placeholder="500"
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        min="0"
                        step="0.1"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Unit
                      </label>
                      <select
                        value={formData.unit}
                        onChange={(e) => setFormData(prev => ({ ...prev, unit: e.target.value }))}
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      >
                        <option value="mg">mg</option>
                        <option value="g">g</option>
                        <option value="ml">ml</option>
                        <option value="mcg">mcg</option>
                        <option value="units">units</option>
                        <option value="tablets">tablets</option>
                        <option value="capsules">capsules</option>
                      </select>
                    </div>
                  </div>
                  
                  {/* Validation Error */}
                  {validation.dosage.error && (
                    <div className="flex items-center space-x-2 text-red-600">
                      <AlertCircle className="w-5 h-5" />
                      <span className="text-sm">{validation.dosage.error}</span>
                    </div>
                  )}
                </motion.div>
              )}
              
              {/* Step 3: Frequency */}
              {currentStep === 3 && (
                <motion.div
                  key="step3"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      How often do you take this medication?
                    </h3>
                    <p className="text-gray-600">
                      Select the frequency that matches your prescription.
                    </p>
                  </div>
                  
                  <div className="space-y-3">
                    {[
                      { value: 'daily', label: 'Once daily' },
                      { value: 'twice-daily', label: 'Twice daily' },
                      { value: 'three-times-daily', label: 'Three times daily' },
                      { value: 'weekly', label: 'Weekly' },
                      { value: 'as-needed', label: 'As needed' },
                      { value: 'custom', label: 'Custom schedule' }
                    ].map((option) => (
                      <label
                        key={option.value}
                        className={cn(
                          "flex items-center p-4 border rounded-xl cursor-pointer transition-all",
                          formData.frequency === option.value
                            ? "border-blue-500 bg-blue-50"
                            : "border-gray-200 hover:border-gray-300"
                        )}
                      >
                        <input
                          type="radio"
                          name="frequency"
                          value={option.value}
                          checked={formData.frequency === option.value}
                          onChange={(e) => setFormData(prev => ({ ...prev, frequency: e.target.value }))}
                          className="sr-only"
                        />
                        <div className={cn(
                          "w-5 h-5 rounded-full border-2 mr-3 flex items-center justify-center",
                          formData.frequency === option.value
                            ? "border-blue-500 bg-blue-500"
                            : "border-gray-300"
                        )}>
                          {formData.frequency === option.value && (
                            <div className="w-2 h-2 bg-white rounded-full" />
                          )}
                        </div>
                        <span className="font-medium">{option.label}</span>
                      </label>
                    ))}
                  </div>
                  
                  {/* Custom Frequency Input */}
                  {formData.frequency === 'custom' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Custom Schedule
                      </label>
                      <input
                        type="text"
                        value={formData.customFrequency}
                        onChange={(e) => setFormData(prev => ({ ...prev, customFrequency: e.target.value }))}
                        placeholder="e.g., Every 8 hours, Morning and evening"
                        className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  )}
                  
                  {/* Validation Error */}
                  {validation.frequency.error && (
                    <div className="flex items-center space-x-2 text-red-600">
                      <AlertCircle className="w-5 h-5" />
                      <span className="text-sm">{validation.frequency.error}</span>
                    </div>
                  )}
                </motion.div>
              )}
              
              {/* Step 4: Health Metrics */}
              {currentStep === 4 && (
                <motion.div
                  key="step4"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      Which health metrics would you like to track?
                    </h3>
                    <p className="text-gray-600">
                      Select metrics to monitor while taking this medication.
                    </p>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    {availableMetrics.map((metric) => (
                      <button
                        key={metric}
                        onClick={() => handleMetricToggle(metric)}
                        className={cn(
                          "p-4 border rounded-xl text-left transition-all",
                          formData.selectedMetrics.includes(metric)
                            ? "border-blue-500 bg-blue-50 text-blue-900"
                            : "border-gray-200 hover:border-gray-300"
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium">{metric}</span>
                          {formData.selectedMetrics.includes(metric) && (
                            <CheckCircle className="w-5 h-5 text-blue-500" />
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                  
                  {/* Additional Instructions */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Additional Instructions (Optional)
                    </label>
                    <textarea
                      value={formData.instructions}
                      onChange={(e) => setFormData(prev => ({ ...prev, instructions: e.target.value }))}
                      placeholder="Take with food, avoid alcohol, etc."
                      rows={3}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    />
                  </div>
                  
                  {/* Validation Error */}
                  {validation.metrics.error && (
                    <div className="flex items-center space-x-2 text-red-600">
                      <AlertCircle className="w-5 h-5" />
                      <span className="text-sm">{validation.metrics.error}</span>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          
          {/* Footer */}
          <div className="bg-muted/30 px-6 py-4 flex items-center justify-between">
            <button
              onClick={handlePrevious}
              disabled={currentStep === 1}
              className={cn(
                "flex items-center space-x-2 px-4 py-2 rounded-lg transition-all",
                currentStep === 1
                  ? "text-gray-400 cursor-not-allowed"
                  : "text-gray-600 hover:bg-gray-200"
              )}
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Previous</span>
            </button>
            
            <div className="flex items-center space-x-3">
              {currentStep < 4 ? (
                <button
                  onClick={handleNext}
                  disabled={!canProceed() || isValidating}
                  className={cn(
                    "flex items-center space-x-2 px-6 py-2 rounded-lg font-medium transition-all",
                    canProceed() && !isValidating
                      ? "bg-foreground text-background hover:bg-foreground/90"
                      : "bg-muted text-muted-foreground cursor-not-allowed"
                  )}
                >
                  <span>Next</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              ) : (
                <button
                  onClick={handleSubmit}
                  disabled={!canProceed() || isSubmitting}
                  className={cn(
                    "flex items-center space-x-2 px-6 py-2 rounded-lg font-medium transition-all",
                    canProceed() && !isSubmitting
                      ? "bg-green-600 text-white hover:bg-green-700"
                      : "bg-muted text-muted-foreground cursor-not-allowed"
                  )}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Adding...</span>
                    </>
                  ) : (
                    <>
                      <span>Add Medication</span>
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default MedicationWizard;
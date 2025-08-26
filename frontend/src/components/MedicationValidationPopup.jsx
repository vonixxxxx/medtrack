import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, Search, Pill, Syringe, Activity, CheckCircle, Info, AlertTriangle,
  Shield, Database, Clock, MapPin, Calendar, FileText, Target, 
  Settings, Zap, Heart
} from 'lucide-react';

const MedicationValidationPopup = ({ isOpen, onClose, onMedicationSelected }) => {
  // State for search and results
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  
  // State for selected medication and product
  const [selectedMedication, setSelectedMedication] = useState(null);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [productOptions, setProductOptions] = useState(null);
  
  // State for form data
  const [medicationForm, setMedicationForm] = useState({
    dosage: '',
    frequency: '',
    startDate: '',
    endDate: '',
    notes: '',
    monitoringMetrics: [],
    metricUpdateFrequency: 'daily',
    intakeType: '',
    intakePlace: '',
    customValues: {
      dose: false,
      frequency: false,
      intakeType: false,
      intakePlace: false
    }
  });

  // UI state
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showCompleteForm, setShowCompleteForm] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});

  // Available monitoring metrics
  const availableMetrics = [
    'Blood Pressure', 'Blood Sugar', 'Weight', 'Heart Rate', 'Temperature',
    'Cholesterol', 'Liver Function', 'Kidney Function', 'Side Effects',
    'Mood Changes', 'Sleep Quality', 'Energy Levels', 'Appetite Changes'
  ];

  // Frequency options for metric monitoring
  const metricFrequencyOptions = [
    'daily',
    'twice daily',
    'every 2 days',
    'weekly',
    'every 2 weeks',
    'monthly',
    'as needed',
    'before medication',
    'after medication',
    'with medication'
  ];

  // API base URL
  const API_BASE_URL = 'http://localhost:8000/api';
  const searchRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      setSearchTerm('');
      setSearchResults([]);
      setSuggestions([]);
      setSelectedMedication(null);
      setSelectedProduct(null);
      setProductOptions(null);
      setError('');
      setSuccess('');
      setShowCompleteForm(false);
      setValidationErrors({});
      setMedicationForm({
        dosage: '',
        frequency: '',
        startDate: '',
        endDate: '',
        notes: '',
        monitoringMetrics: [],
        metricUpdateFrequency: 'daily',
        intakeType: '',
        intakePlace: '',
        customValues: {
          dose: false,
          frequency: false,
          intakeType: false,
          intakePlace: false
        }
      });
    }
  }, [isOpen]);

  // Hospital-grade medication search
  const searchMedications = async (query) => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    setError('');
    
    try {
      console.log(`🏥 Hospital-Grade Search: "${query}"`);
      
      const response = await fetch(`${API_BASE_URL}/meds/search?q=${encodeURIComponent(query.trim())}&limit=10`);
      const data = await response.json();
      
      if (response.ok) {
        setSearchResults(data.matches || []);
        setSuggestions(data.suggestions || []);
        
        // Show provenance information
        const source = data.hospitalGrade ? 
          '🏥 NHS Hospital-Grade System' : 
          '📊 Fallback Database';
        
        console.log(`✅ Search completed via: ${source}`);
        
        if (data.matches.length === 0) {
          const suggestionText = data.suggestions?.length > 0 ? 
            `Did you mean: ${data.suggestions.slice(0, 3).join(', ')}?` : 
            'Try searching for common medications like "paracetamol", "ibuprofen", or "aspirin".';
          
          setError(`No medications found for "${query}". ${suggestionText}`);
        } else {
          setError(''); // Clear any previous errors
        }
      } else {
        setError(data.error || 'Search failed. Please try again.');
        setSearchResults([]);
        setSuggestions([]);
      }
    } catch (error) {
      console.error('🚨 Hospital-Grade Search error:', error);
      setError('Search service temporarily unavailable. Please check your connection and try again.');
      setSearchResults([]);
      setSuggestions([]);
    } finally {
      setIsSearching(false);
    }
  };

  // Get hospital-grade product options for UI rendering
  const getProductOptions = async (productId) => {
    try {
      console.log(`🏥 Fetching hospital-grade options for product: ${productId}`);
      
      const response = await fetch(`${API_BASE_URL}/meds/product/${productId}/options`);
      const data = await response.json();
      
      if (response.ok) {
        console.log(`✅ Product options received:`, {
          brand: data.brand_name,
          generic: data.generic_name,
          intakeType: data.allowed_intake_type,
          strengths: data.strengths?.length || 0,
          frequencies: data.allowed_frequencies?.length || 0,
          source: data.nhsValidated ? 'NHS Hospital-Grade' : 'Database Fallback'
        });
        
        setProductOptions(data);
        
        // Auto-populate fields with server-approved options
        setMedicationForm(prev => ({
          ...prev,
          intakeType: data.allowed_intake_type || '',
          intakePlace: data.default_places?.[0] || 'at home'
        }));
        
        setError(''); // Clear any errors
        
      } else {
        console.error('🚨 Failed to get product options:', data);
        setError(data.error || 'Failed to get approved options for this medication.');
      }
    } catch (error) {
      console.error('🚨 Product options error:', error);
      setError('Failed to connect to validation service. Please try again.');
    }
  };

  // Hospital-grade medication validation
  const validateMedicationConfig = async () => {
    if (!selectedMedication || !selectedProduct || !productOptions) {
      setError('Please select a medication and product first.');
      return;
    }

    setError('');
    setValidationErrors({});

    try {
      // Find the strength details from server-approved options
      const selectedStrength = productOptions.strengths?.find(s => 
        s.value === parseFloat(medicationForm.dosage)
      );
      
      if (!selectedStrength && !medicationForm.customValues.dose) {
        setError('Please select a valid dosage from the approved options.');
        return;
      }

      const payload = {
        medication_id: selectedMedication.id,
        product_id: selectedProduct.id,
        intake_type: medicationForm.intakeType,
        intake_place: medicationForm.intakePlace,
        strength_value: parseFloat(medicationForm.dosage),
        strength_unit: selectedStrength?.unit || 'mg',
        frequency: medicationForm.frequency,
        custom_flags: medicationForm.customValues
      };

      console.log(`🏥 Hospital-Grade Validation:`, {
        medication: selectedMedication.genericName,
        product: selectedProduct.brandName,
        dosage: `${payload.strength_value} ${payload.strength_unit}`,
        frequency: payload.frequency,
        intakeType: payload.intake_type,
        customFlags: payload.custom_flags
      });

      const response = await fetch(`${API_BASE_URL}/meds/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (response.ok && data.valid) {
        console.log(`✅ Hospital-Grade Validation Passed:`, data.normalized);
        setSuccess(`✅ Configuration validated by ${data.source || 'Hospital-Grade System'}!`);
        setShowCompleteForm(true);
        setValidationErrors({});
      } else if (response.status === 422) {
        console.warn(`🚨 Hospital-Grade Validation Failed:`, data.errors);
        
        // Map server validation errors to fields
        const fieldErrors = {};
        data.errors?.forEach(err => {
          if (err.field === 'general') {
            setError(`❌ ${err.message}`);
          } else {
            fieldErrors[err.field] = err.message;
          }
        });
        setValidationErrors(fieldErrors);
        
        // Auto-refetch approved options to reset UI
        if (selectedProduct && data.suggested_options_endpoint) {
          console.log(`🔄 Refetching approved options...`);
          await getProductOptions(selectedProduct.id);
        }
        
        setError('Please correct the highlighted fields and try again.');
        
      } else {
        console.error(`🚨 Validation service error:`, data);
        setError(data.error || 'Validation service temporarily unavailable. Please try again.');
      }
    } catch (error) {
      console.error('🚨 Hospital-Grade Validation error:', error);
      setError('Failed to connect to validation service. Please check your connection and try again.');
    }
  };

  // Handle search input changes
  const handleSearchInputChange = (e) => {
    const value = e.target.value;
    setSearchTerm(value);
    setSearchResults([]);
    setSuggestions([]);
    setSelectedMedication(null);
    setSelectedProduct(null);
    setProductOptions(null);
    setError('');
    setSuccess('');
    setShowCompleteForm(false);
    setValidationErrors({});

    if (value.trim().length >= 2) {
      // Debounce search
      const timeoutId = setTimeout(() => {
        searchMedications(value);
      }, 300);
      return () => clearTimeout(timeoutId);
    }
  };

  // Handle search submission
  const handleSearch = (e) => {
    e.preventDefault();
    if (searchTerm.trim().length >= 2) {
      searchMedications(searchTerm);
    }
  };

  // Handle medication selection
  const handleMedicationSelect = (medication) => {
    setSelectedMedication(medication);
    setSelectedProduct(null);
    setProductOptions(null);
    setError('');
    setSuccess('');
    setShowCompleteForm(false);
    setValidationErrors({});

    // If only one product, auto-select it
    if (medication.products.length === 1) {
      handleProductSelect(medication.products[0]);
    }
  };

  // Handle product selection
  const handleProductSelect = (product) => {
    setSelectedProduct(product);
    setError('');
    setSuccess('');
    setShowCompleteForm(false);
    setValidationErrors({});
    
    // Fetch product options
    getProductOptions(product.id);
  };

  // Handle dosage selection
  const handleDosageSelect = (strength) => {
    setMedicationForm(prev => ({
      ...prev,
      dosage: strength.value.toString(),
      frequency: strength.frequency
    }));
    setValidationErrors(prev => ({ ...prev, dosage: undefined }));
  };

  // Handle form submission
  const handleCompleteFormSubmit = (e) => {
    e.preventDefault();
    
    // Basic validation
    const errors = {};
    
    if (!medicationForm.startDate) {
      errors.startDate = 'Start date is required.';
    }
    
    if (medicationForm.endDate && new Date(medicationForm.endDate) <= new Date(medicationForm.startDate)) {
      errors.endDate = 'End date must be after start date.';
    }
    
    if (medicationForm.monitoringMetrics.length === 0) {
      errors.monitoringMetrics = 'Please select at least one metric to monitor.';
    }
    
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      return;
    }
    
    // Create the medication object
    const completeMedication = {
      name: selectedMedication.genericName,
      generic: selectedMedication.genericName,
      brand: selectedProduct.brandName,
      class: selectedMedication.classHuman,
      dosage: medicationForm.dosage,
      form: medicationForm.intakeType,
      frequency: medicationForm.frequency,
      startDate: medicationForm.startDate,
      endDate: medicationForm.endDate,
      notes: medicationForm.notes,
      monitoringMetrics: medicationForm.monitoringMetrics,
      metricUpdateFrequency: medicationForm.metricUpdateFrequency,
      intakeType: medicationForm.intakeType,
      intakePlace: medicationForm.intakePlace,
      customValues: medicationForm.customValues,
      source: 'API Database',
      productId: selectedProduct.id,
      medicationId: selectedMedication.id
    };
    
    onMedicationSelected(completeMedication);
  };

  // Handle metric toggles
  const handleMetricToggle = (metric) => {
    setMedicationForm(prev => ({
      ...prev,
      monitoringMetrics: prev.monitoringMetrics.includes(metric)
        ? prev.monitoringMetrics.filter(m => m !== metric)
        : [...prev.monitoringMetrics, metric]
    }));
    setValidationErrors(prev => ({ ...prev, monitoringMetrics: undefined }));
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className="bg-white rounded-2xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-xl"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
                <Pill className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-800">Add Medication</h2>
                <p className="text-gray-600 text-sm">Search and validate your medication</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 bg-gray-100 hover:bg-gray-200 rounded-full flex items-center justify-center transition-colors"
            >
              <X className="w-4 h-4 text-gray-600" />
            </button>
          </div>

          {/* Main Content */}
          <div className="space-y-6">
            {/* Search Section */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Medication Name
              </label>
              <div className="relative">
                <input
                  ref={searchRef}
                  type="text"
                  value={searchTerm}
                  onChange={handleSearchInputChange}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch(e)}
                  placeholder="Search by generic name, brand, or class (e.g., ibuprofen, GLP1, Ozempic)"
                  className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
                <button
                  onClick={handleSearch}
                  disabled={isSearching || searchTerm.trim().length < 2}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <Search className="w-4 h-4" />
                </button>
              </div>
              
              {isSearching && (
                <div className="flex items-center space-x-2 text-blue-600 text-sm mt-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <span>Searching...</span>
                </div>
              )}
            </div>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div>
                <h3 className="font-medium text-gray-800 mb-3">Search Results</h3>
                <div className="space-y-3 max-h-60 overflow-y-auto">
                  {searchResults.map((medication) => (
                    <div
                      key={medication.id}
                      className="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:border-blue-300 transition-colors cursor-pointer"
                      onClick={() => handleMedicationSelect(medication)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold text-gray-800">{medication.genericName}</h4>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          medication.reason === 'exact' ? 'bg-green-100 text-green-800' :
                          medication.reason === 'synonym' ? 'bg-blue-100 text-blue-800' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>
                          {medication.reason}
                        </span>
                      </div>
                      
                      {medication.classHuman && (
                        <p className="text-sm text-gray-600 mb-3">{medication.classHuman}</p>
                      )}
                      
                      <div className="space-y-2">
                        {medication.products.map((product) => (
                          <div
                            key={product.id}
                            className="flex items-center space-x-3 p-2 bg-white rounded-lg hover:bg-blue-50 transition-colors"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleProductSelect(product);
                            }}
                          >
                            <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center">
                              {product.allowedIntakeType === 'Injection' ? (
                                <Syringe className="w-3 h-3 text-blue-600" />
                              ) : product.allowedIntakeType === 'Pill/Tablet' ? (
                                <Pill className="w-3 h-3 text-blue-600" />
                              ) : (
                                <Activity className="w-3 h-3 text-blue-600" />
                              )}
                            </div>
                            <div className="flex-1">
                              <p className="font-medium text-gray-800">{product.brandName}</p>
                              <p className="text-xs text-gray-600">{product.form} • {product.route}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Suggestions for no results */}
            {suggestions.length > 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="font-medium text-yellow-800 mb-2 flex items-center">
                  <Info className="w-4 h-4 mr-2" />
                  Did you mean?
                </h4>
                <div className="space-y-1">
                  {suggestions.map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => setSearchTerm(suggestion.split(' ')[0])}
                      className="block w-full text-left text-sm text-yellow-700 hover:text-yellow-900 hover:bg-yellow-100 rounded px-2 py-1 transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Product Configuration */}
            {selectedProduct && productOptions && (
              <div>
                <h3 className="font-medium text-gray-800 mb-3 flex items-center">
                  <Shield className="w-4 h-4 mr-2 text-green-600" />
                  Hospital-Grade Configuration
                </h3>
                
                <div className="bg-blue-50 rounded-lg p-4 mb-4 border border-blue-200">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h4 className="font-medium text-blue-800 mb-1">{selectedProduct.brandName}</h4>
                      <p className="text-sm text-blue-600">{productOptions.generic_name} • {productOptions.metadata?.form || selectedProduct.form} • {productOptions.metadata?.route || selectedProduct.route}</p>
                    </div>
                    <div className="flex items-center text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                      <Database className="w-3 h-3 mr-1" />
                      {productOptions.nhsValidated ? 'NHS' : 'DB'}
                    </div>
                  </div>
                  
                  {/* Provenance Badge */}
                  <div className="text-xs text-blue-500 mt-2 flex items-center">
                    <Info className="w-3 h-3 mr-1" />
                    Source: {productOptions.source || 'NHS Hospital-Grade System'}
                  </div>
                </div>

                {/* Dosage Selection */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <Zap className="w-4 h-4 mr-1 text-blue-600" />
                    Server-Approved Dosages Only
                  </label>
                  {validationErrors.strength_value && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-2 mb-2">
                      <p className="text-sm text-red-600 flex items-center">
                        <AlertTriangle className="w-4 h-4 mr-1" />
                        {validationErrors.strength_value}
                      </p>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    {productOptions.strengths && productOptions.strengths.length > 0 ? (
                      productOptions.strengths.map((strength) => (
                        <button
                          key={`${strength.value}-${strength.unit}-${strength.frequency || 'default'}`}
                          onClick={() => handleDosageSelect(strength)}
                          className={`p-3 text-sm rounded-lg border transition-all duration-200 ${
                            medicationForm.dosage === strength.value.toString()
                              ? 'bg-blue-600 text-white border-blue-600 shadow-md'
                              : 'bg-white text-gray-700 border-gray-200 hover:border-blue-300 hover:shadow-sm'
                          }`}
                        >
                          <div className="font-medium">{strength.value} {strength.unit}</div>
                          {strength.frequency && (
                            <div className="text-xs opacity-75">{strength.frequency}</div>
                          )}
                          {strength.label && (
                            <div className="text-xs opacity-60 mt-1">{strength.label}</div>
                          )}
                        </button>
                      ))
                    ) : (
                      <div className="col-span-2 text-center py-4 text-gray-500 bg-gray-50 rounded-lg border border-gray-200">
                        <AlertTriangle className="w-6 h-6 mx-auto mb-2 text-orange-500" />
                        <p className="text-sm">No approved dosages available</p>
                        <p className="text-xs">Consult dosage guidelines</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Place of Intake */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <MapPin className="w-4 h-4 mr-1 text-blue-600" />
                    Place of Intake
                  </label>
                  {validationErrors.intake_place && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-2 mb-2">
                      <p className="text-sm text-red-600 flex items-center">
                        <AlertTriangle className="w-4 h-4 mr-1" />
                        {validationErrors.intake_place}
                      </p>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    {productOptions.default_places && productOptions.default_places.length > 0 ? (
                      productOptions.default_places.map((place) => (
                        <button
                          key={place}
                          onClick={() => setMedicationForm(prev => ({ ...prev, intakePlace: place }))}
                          className={`p-2 text-sm rounded-lg border transition-all duration-200 ${
                            medicationForm.intakePlace === place
                              ? 'bg-blue-600 text-white border-blue-600 shadow-md'
                              : 'bg-white text-gray-700 border-gray-200 hover:border-blue-300 hover:shadow-sm'
                          }`}
                        >
                          {place}
                        </button>
                      ))
                    ) : (
                      <div className="col-span-2 text-center py-2 text-gray-500 bg-gray-50 rounded-lg border border-gray-200">
                        <p className="text-sm">at home</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Frequency */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <Clock className="w-4 h-4 mr-1 text-blue-600" />
                    Server-Approved Frequencies
                  </label>
                  {validationErrors.frequency && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-2 mb-2">
                      <p className="text-sm text-red-600 flex items-center">
                        <AlertTriangle className="w-4 h-4 mr-1" />
                        {validationErrors.frequency}
                      </p>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    {productOptions.allowed_frequencies && productOptions.allowed_frequencies.length > 0 ? (
                      productOptions.allowed_frequencies.map((freq) => (
                        <button
                          key={freq}
                          onClick={() => setMedicationForm(prev => ({ ...prev, frequency: freq }))}
                          className={`p-2 text-sm rounded-lg border transition-all duration-200 ${
                            medicationForm.frequency === freq
                              ? 'bg-blue-600 text-white border-blue-600 shadow-md'
                              : 'bg-white text-gray-700 border-gray-200 hover:border-blue-300 hover:shadow-sm'
                          }`}
                        >
                          {freq}
                        </button>
                      ))
                    ) : (
                      <div className="col-span-2 text-center py-4 text-gray-500 bg-gray-50 rounded-lg border border-gray-200">
                        <AlertTriangle className="w-6 h-6 mx-auto mb-2 text-orange-500" />
                        <p className="text-sm">No approved frequencies available</p>
                        <p className="text-xs">Consult dosage guidelines</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Validation Button */}
                <button
                  onClick={validateMedicationConfig}
                  disabled={!medicationForm.dosage || !medicationForm.frequency || !medicationForm.intakePlace}
                  className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-md hover:shadow-lg flex items-center justify-center"
                >
                  <Shield className="w-4 h-4 mr-2" />
                  Hospital-Grade Validation
                </button>
                
                {/* Validation Info */}
                <div className="mt-2 text-xs text-gray-500 text-center">
                  <Info className="w-3 h-3 inline mr-1" />
                  Zero tolerance for incorrect medication configurations
                </div>
              </div>
            )}

            {/* Complete Form */}
            {showCompleteForm && (
              <div>
                <h3 className="font-medium text-gray-800 mb-4">Complete Medication Details</h3>
                
                <form onSubmit={handleCompleteFormSubmit} className="space-y-4">
                  {/* Selected Medication Info */}
                  <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                    <h4 className="font-medium text-blue-800 mb-1">Selected Medication</h4>
                    <p className="text-blue-700 font-semibold">{selectedMedication.genericName}</p>
                    {medicationForm.dosage && (
                      <p className="text-sm text-blue-600">
                        Dosage: {medicationForm.dosage} {medicationForm.intakeType}
                      </p>
                    )}
                  </div>

                  {/* Dates */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Start Date
                      </label>
                      <input
                        type="date"
                        value={medicationForm.startDate}
                        onChange={(e) => setMedicationForm(prev => ({ ...prev, startDate: e.target.value }))}
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        required
                      />
                      {validationErrors.startDate && (
                        <p className="text-sm text-red-600 mt-1">{validationErrors.startDate}</p>
                      )}
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        End Date (Optional)
                      </label>
                      <input
                        type="date"
                        value={medicationForm.endDate}
                        onChange={(e) => setMedicationForm(prev => ({ ...prev, endDate: e.target.value }))}
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                      {validationErrors.endDate && (
                        <p className="text-sm text-red-600 mt-1">{validationErrors.endDate}</p>
                      )}
                    </div>
                  </div>

                  {/* Monitoring Metrics */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Metrics to Monitor
                    </label>
                    <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
                      {availableMetrics.map((metric) => (
                        <button
                          key={metric}
                          type="button"
                          onClick={() => handleMetricToggle(metric)}
                          className={`p-2 text-sm rounded-lg border transition-colors ${
                            medicationForm.monitoringMetrics.includes(metric)
                              ? 'bg-blue-600 text-white border-blue-600'
                              : 'bg-white text-gray-700 border-gray-200 hover:border-blue-300'
                          }`}
                        >
                          {metric}
                        </button>
                      ))}
                    </div>
                    {validationErrors.monitoringMetrics && (
                      <p className="text-sm text-red-600 mt-1">{validationErrors.monitoringMetrics}</p>
                    )}
                  </div>

                  {/* Metric Update Frequency */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      How often should metrics be updated?
                    </label>
                    <select
                      value={medicationForm.metricUpdateFrequency}
                      onChange={(e) => setMedicationForm(prev => ({ ...prev, metricUpdateFrequency: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      {metricFrequencyOptions.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Notes */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Additional Notes (Optional)
                    </label>
                    <textarea
                      value={medicationForm.notes}
                      onChange={(e) => setMedicationForm(prev => ({ ...prev, notes: e.target.value }))}
                      placeholder="Any additional information about your medication..."
                      rows={3}
                      className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  {/* Submit Button */}
                  <button
                    type="submit"
                    className="w-full bg-green-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-green-700 transition-colors"
                  >
                    Add Medication to Dashboard
                  </button>
                </form>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 text-red-800">
                  <AlertTriangle className="w-4 h-4" />
                  <span className="font-medium">{error}</span>
                </div>
              </div>
            )}

            {/* Success Display */}
            {success && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 text-green-800">
                  <CheckCircle className="w-4 h-4" />
                  <span className="font-medium">{success}</span>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default MedicationValidationPopup;

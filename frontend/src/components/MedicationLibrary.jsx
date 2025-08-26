import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Search, 
  Pill, 
  Info, 
  Calendar, 
  AlertTriangle,
  ExternalLink,
  Filter,
  BookOpen,
  Database,
  Download,
  Globe,
  Shield,
  CheckCircle,
  Clock,
  AlertCircle,
  Heart,
  Baby,
  Users,
  Brain,
  Zap
} from 'lucide-react';

export default function MedicationLibrary() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedMedication, setSelectedMedication] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeFilter, setActiveFilter] = useState('ema');
  const [showModal, setShowModal] = useState(false);
  const [dataSources, setDataSources] = useState([]);
  const [error, setError] = useState(null);
  const [lastSearch, setLastSearch] = useState('');

  // API base URL - use Vite's import.meta.env instead of process.env
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  useEffect(() => {
    loadDataSources();
  }, []);

  const loadDataSources = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/medications/sources`);
      if (response.ok) {
        const data = await response.json();
        setDataSources(data.sources);
      }
    } catch (error) {
      console.error('Failed to load data sources:', error);
    }
  };

  const handleSearch = async (query) => {
    setSearchQuery(query);
    
    if (query.trim().length < 2) {
      setSearchResults([]);
      setError(null);
      return;
    }

    setIsLoading(true);
    setError(null);
    setLastSearch(query);
    
    try {
      const params = new URLSearchParams({
        query: query.trim(),
        limit: 20,
        source: activeFilter
      });

      const response = await fetch(`${API_BASE}/api/medications/search?${params}`);
      
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json();
      setSearchResults(data.data || []);
      
    } catch (error) {
      console.error('Search error:', error);
      setError(`Search failed: ${error.message}`);
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFilterChange = (filter) => {
    setActiveFilter(filter);
    if (searchQuery.trim().length >= 2) {
      handleSearch(searchQuery);
    }
  };

  const openMedicationDetails = async (medication) => {
    try {
      const response = await fetch(`${API_BASE}/api/medications/details/${medication.source}/${medication.id}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch details: ${response.statusText}`);
      }

      const data = await response.json();
      setSelectedMedication(data.data);
      setShowModal(true);
      
    } catch (error) {
      console.error('Failed to fetch medication details:', error);
      // Fallback to basic info if details fetch fails
      setSelectedMedication(medication);
      setShowModal(true);
    }
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedMedication(null);
  };

  const getSourceIcon = (source) => {
    switch (source) {
      case 'rxnorm':
        return <Database className="w-4 h-4" />;
      case 'openfda':
        return <Shield className="w-4 h-4" />;
      case 'nhs_dmd':
        return <BookOpen className="w-4 h-4" />;
      case 'ema':
        return <Globe className="w-4 h-4" />;
      default:
        return <Pill className="w-4 h-4" />;
    }
  };

  const getSourceColor = (source) => {
    switch (source) {
      case 'rxnorm':
        return 'bg-blue-100 text-blue-800';
      case 'openfda':
        return 'bg-green-100 text-green-800';
      case 'nhs_dmd':
        return 'bg-purple-100 text-purple-800';
      case 'ema':
        return 'bg-orange-100 text-orange-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getSourceName = (source) => {
    const sourceInfo = dataSources.find(s => s.id === source);
    return sourceInfo ? sourceInfo.name : source;
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return 'Unknown';
    }
  };

  const renderSafetySection = (medication) => {
    if (!medication.warnings && !medication.contraindications) return null;

    return (
      <div className="bg-red-50 rounded-xl p-4 border border-red-200">
        <h3 className="font-semibold text-red-900 mb-3 flex items-center">
          <AlertCircle className="w-5 h-5 mr-2" />
          Safety Information
        </h3>
        <div className="space-y-3">
          {medication.warnings && medication.warnings.length > 0 && (
            <div>
              <h4 className="font-medium text-red-800 mb-2">⚠️ Warnings & Precautions</h4>
              <ul className="space-y-1">
                {medication.warnings.map((warning, index) => (
                  <li key={index} className="text-red-700 text-sm flex items-start">
                    <span className="text-red-500 mr-2">•</span>
                    <span>{warning}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {medication.contraindications && medication.contraindications.length > 0 && (
            <div>
              <h4 className="font-medium text-red-800 mb-2">🚫 Contraindications</h4>
              <ul className="space-y-1">
                {medication.contraindications.map((contraindication, index) => (
                  <li key={index} className="text-red-700 text-sm flex items-start">
                    <span className="text-red-500 mr-2">•</span>
                    <span>{contraindication}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderSideEffectsSection = (medication) => {
    if (!medication.sideEffects || medication.sideEffects.length === 0) return null;

    return (
      <div className="bg-yellow-50 rounded-xl p-4 border border-yellow-200">
        <h3 className="font-semibold text-yellow-900 mb-3 flex items-center">
          <Zap className="w-5 h-5 mr-2" />
          Side Effects
        </h3>
        <ul className="space-y-1">
          {medication.sideEffects.map((effect, index) => (
            <li key={index} className="text-yellow-800 text-sm flex items-start">
              <span className="text-yellow-600 mr-2">•</span>
              <span>{effect}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderSpecialPopulationsSection = (medication) => {
    if (!medication.pregnancyCategory && !medication.breastfeedingCategory && 
        !medication.pediatricUse && !medication.geriatricUse) return null;

    return (
      <div className="bg-indigo-50 rounded-xl p-4 border border-indigo-200">
        <h3 className="font-semibold text-indigo-900 mb-3 flex items-center">
          <Users className="w-5 h-5 mr-2" />
          Special Populations
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          {medication.pregnancyCategory && (
            <div className="text-indigo-800">
              <span className="font-medium">Pregnancy:</span> {medication.pregnancyCategory}
            </div>
          )}
          {medication.breastfeedingCategory && (
            <div className="text-indigo-800">
              <span className="font-medium">Breastfeeding:</span> {medication.breastfeedingCategory}
            </div>
          )}
          {medication.pediatricUse && (
            <div className="text-indigo-800">
              <span className="font-medium">Pediatric:</span> {medication.pediatricUse}
            </div>
          )}
          {medication.geriatricUse && (
            <div className="text-indigo-800">
              <span className="font-medium">Geriatric:</span> {medication.geriatricUse}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderLLMSummary = (medication) => {
    if (!medication.llmSummary) return null;

    return (
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-3 flex items-center">
          <Brain className="w-5 h-5 mr-2" />
          AI-Powered Summary
        </h3>
        <div className="text-sm text-blue-800 prose prose-sm max-w-none">
          {typeof medication.llmSummary === 'string' ? (
            <div className="whitespace-pre-wrap">{medication.llmSummary}</div>
          ) : (
            <div className="space-y-4">
              {medication.llmSummary.overview && (
                <div className="bg-white rounded-lg p-3 border border-blue-200">
                  <h4 className="font-medium text-blue-900 mb-2">📋 Overview</h4>
                  <p className="text-blue-800">{medication.llmSummary.overview}</p>
                </div>
              )}
              
              {medication.llmSummary.warnings && (
                <div className="bg-red-50 rounded-lg p-3 border border-red-200">
                  <h4 className="font-medium text-red-900 mb-2">⚠️ Warnings</h4>
                  <p className="text-red-800">{medication.llmSummary.warnings}</p>
                </div>
              )}
              
              {medication.llmSummary.sideEffects && (
                <div className="bg-yellow-50 rounded-lg p-3 border border-yellow-200">
                  <h4 className="font-medium text-yellow-900 mb-2">🔄 Side Effects</h4>
                  <p className="text-yellow-800">{medication.llmSummary.sideEffects}</p>
                </div>
              )}
              
              {medication.llmSummary.dosage && (
                <div className="bg-green-50 rounded-lg p-3 border border-green-200">
                  <h4 className="font-medium text-green-900 mb-2">💊 Dosage Information</h4>
                  <p className="text-green-800">{medication.llmSummary.dosage}</p>
                </div>
              )}
              
              {medication.llmSummary.interactions && (
                <div className="bg-purple-50 rounded-lg p-3 border border-purple-200">
                  <h4 className="font-medium text-purple-900 mb-2">🚫 Drug Interactions</h4>
                  <p className="text-purple-800">{medication.llmSummary.interactions}</p>
                </div>
              )}
              
              {medication.llmSummary.specialPopulations && (
                <div className="bg-indigo-50 rounded-lg p-3 border border-indigo-200">
                  <h4 className="font-medium text-indigo-900 mb-2">👥 Special Populations</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                    {medication.llmSummary.specialPopulations.pregnancy && (
                      <div className="text-indigo-800">{medication.llmSummary.specialPopulations.pregnancy}</div>
                    )}
                    {medication.llmSummary.specialPopulations.breastfeeding && (
                      <div className="text-indigo-800">{medication.llmSummary.specialPopulations.breastfeeding}</div>
                    )}
                    {medication.llmSummary.specialPopulations.pediatric && (
                      <div className="text-indigo-800">{medication.llmSummary.specialPopulations.pediatric}</div>
                    )}
                    {medication.llmSummary.specialPopulations.geriatric && (
                      <div className="text-indigo-800">{medication.llmSummary.specialPopulations.geriatric}</div>
                    )}
                  </div>
                </div>
              )}
              
              {medication.llmSummary.safetyTips && (
                <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
                  <h4 className="font-medium text-blue-900 mb-2">🔒 Safety Tips</h4>
                  <ul className="space-y-1 text-sm text-blue-800">
                    {medication.llmSummary.safetyTips.map((tip, index) => (
                      <li key={index} className="flex items-start">
                        <span className="text-blue-600 mr-2">•</span>
                        <span>{tip}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {medication.llmSummary.disclaimer && (
                <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                  <p className="text-xs text-gray-600 italic">{medication.llmSummary.disclaimer}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderTherapeuticInfo = (medication) => {
    if (!medication.atcClass && !medication.therapeuticIndications) return null;

    return (
      <div className="bg-green-50 rounded-xl p-4 border border-green-200">
        <h3 className="font-semibold text-green-900 mb-3 flex items-center">
          <Heart className="w-5 h-5 mr-2" />
          Therapeutic Information
        </h3>
        <div className="space-y-3">
          {medication.atcClass && medication.atcClass.length > 0 && (
            <div>
              <h4 className="font-medium text-green-800 mb-2">🏷️ ATC Classification</h4>
              <div className="text-green-700 text-sm">
                {medication.atcClass.join(', ')}
              </div>
            </div>
          )}
          
          {medication.therapeuticIndications && medication.therapeuticIndications.length > 0 && (
            <div>
              <h4 className="font-medium text-green-800 mb-2">💊 Therapeutic Indications</h4>
              <ul className="space-y-1">
                {medication.therapeuticIndications.map((indication, index) => (
                  <li key={index} className="text-green-700 text-sm flex items-start">
                    <span className="text-green-600 mr-2">•</span>
                    <span>{indication}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-white">
      <motion.div 
        className="max-w-7xl mx-auto px-6 py-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-blue-400 mb-2">Medication Compendium</h1>
          <p className="text-lg text-gray-600 mb-6">
            Comprehensive medication information from trusted medical sources
          </p>
          
          {/* Search Bar */}
          <div className="max-w-2xl mx-auto relative">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-blue-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search medications, generic names, or conditions..."
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                className="w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-2xl focus:border-blue-400 focus:outline-none transition-all duration-300 text-lg shadow-sm hover:shadow-md"
              />
            </div>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex justify-center mb-8">
          <div className="bg-gray-100 rounded-2xl p-1">
            {['ema', 'all', 'rxnorm', 'openfda', 'nhs_dmd'].map((filter) => (
              <button
                key={filter}
                onClick={() => handleFilterChange(filter)}
                className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                  activeFilter === filter
                    ? 'bg-blue-400 text-white shadow-lg'
                    : 'text-gray-600 hover:text-blue-400 hover:bg-white'
                }`}
              >
                {filter === 'all' ? 'All Sources' : filter === 'ema' ? 'EMA (Enhanced)' : filter.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="text-center mb-6">
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 inline-block">
              <AlertTriangle className="w-5 h-5 text-red-500 inline mr-2" />
              <span className="text-red-700">{error}</span>
            </div>
          </div>
        )}

        {/* Search Results */}
        <div className="mb-8">
          {isLoading ? (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
              <p className="text-gray-600">Searching global medication databases...</p>
              <p className="text-sm text-gray-500 mt-2">
                {activeFilter === 'ema' ? 'Querying enhanced EMA database with AI summarization' : 
                 `Querying ${activeFilter === 'all' ? 'all sources' : activeFilter} databases`}
              </p>
            </div>
          ) : searchResults.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {searchResults.map((medication, index) => (
                  <motion.div
                    key={`${medication.source}-${medication.id}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-white rounded-xl p-4 border border-gray-200 hover:border-blue-300 hover:shadow-md transition-all duration-200 cursor-pointer"
                    onClick={() => handleMedicationClick(medication)}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900 text-lg mb-1">
                          {medication.name || 'Unknown Medication'}
                        </h3>
                        {medication.genericName && medication.genericName !== medication.name && (
                          <p className="text-gray-600 text-sm mb-2">
                            Generic: {medication.genericName}
                          </p>
                        )}
                        {medication.activeIngredients && medication.activeIngredients.length > 0 && (
                          <p className="text-blue-600 text-sm mb-2">
                            Active Ingredients: {medication.activeIngredients.join(', ')}
                          </p>
                        )}
                      </div>
                      <div className="text-right">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          {medication.source === 'rxnorm' ? 'RxNorm (US NLM)' : 
                           medication.source === 'openfda' ? 'OpenFDA (US FDA)' : 
                           medication.source === 'ema' ? 'EMA (Enhanced)' : medication.source}
                        </span>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      {medication.therapeuticIndications && medication.therapeuticIndications.length > 0 && (
                        <div className="text-sm text-gray-700">
                          <span className="font-medium">Indications:</span> {medication.therapeuticIndications.slice(0, 2).join(', ')}
                          {medication.therapeuticIndications.length > 2 && '...'}
                        </div>
                      )}
                      
                      {medication.warnings && medication.warnings.length > 0 && (
                        <div className="text-sm text-red-600">
                          <span className="font-medium">⚠️ Warnings:</span> {medication.warnings[0]}
                        </div>
                      )}
                    </div>
                    
                    <div className="mt-3 pt-3 border-t border-gray-100 flex justify-between items-center text-xs text-gray-500">
                      <span>Updated: {new Date(medication.lastUpdated).toLocaleDateString()}</span>
                      <span className="text-blue-600 hover:text-blue-800">Click for details</span>
                    </div>
                  </motion.div>
                ))}
            </div>
          ) : searchQuery && !isLoading ? (
            <div className="text-center py-12">
              <AlertTriangle className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-xl font-medium text-gray-600 mb-2">No medications found</h3>
              <p className="text-gray-500">Try adjusting your search terms or browse different sources</p>
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Search className="w-8 h-8 text-gray-400" />
              </div>
              <p className="text-lg font-medium text-gray-500">Search for medications</p>
              <p className="text-sm text-gray-400">
                {activeFilter === 'ema' ? 
                  'Enter a drug name to search the enhanced EMA database with AI-powered summaries' :
                  'Enter a drug name, generic name, or condition to search global databases'
                }
              </p>
            </div>
          )}
        </div>

        {/* Data Sources Information */}
        {dataSources.length > 0 && (
          <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200">
            <h2 className="text-xl font-bold text-blue-400 mb-4 text-center">
              Trusted Global Data Sources
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {dataSources.map((source) => (
                <div key={source.id} className="bg-white rounded-xl p-4 border border-gray-200">
                  <div className="flex items-center mb-3">
                    {getSourceIcon(source.id)}
                    <h3 className="font-semibold text-gray-900 ml-2 text-sm">{source.name}</h3>
                  </div>
                  <p className="text-gray-600 text-xs mb-2">{source.description}</p>
                  <div className="text-xs text-gray-500">
                    {source.features.slice(0, 2).join(', ')}
                    {source.features.length > 2 && '...'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>

      {/* Enhanced Medication Details Modal */}
      {showModal && selectedMedication && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <motion.div 
            className="bg-white rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.3 }}
          >
            <div className="p-6">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-blue-400 mb-2">
                    {selectedMedication.name}
                  </h2>
                  {selectedMedication.genericName && selectedMedication.genericName !== selectedMedication.name && (
                    <p className="text-xl text-gray-600">{selectedMedication.genericName}</p>
                  )}
                  <div className="flex items-center space-x-2 mt-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${getSourceColor(selectedMedication.source)}`}>
                      {getSourceIcon(selectedMedication.source)}
                      <span className="ml-1">{getSourceName(selectedMedication.source)}</span>
                    </span>
                    <span className="text-sm text-gray-500">
                      Last updated: {formatDate(selectedMedication.lastUpdated)}
                    </span>
                  </div>
                </div>
                <button
                  onClick={closeModal}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* AI Summary Section */}
              {renderLLMSummary(selectedMedication)}

              {/* Safety Information Section */}
              {renderSafetySection(selectedMedication)}

              {/* Side Effects Section */}
              {renderSideEffectsSection(selectedMedication)}

              {/* Therapeutic Information Section */}
              {renderTherapeuticInfo(selectedMedication)}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                {/* Left Column */}
                <div className="space-y-6">
                  {selectedMedication.brandNames && selectedMedication.brandNames.length > 0 && (
                    <div className="bg-gray-50 rounded-xl p-4">
                      <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                        <Info className="w-5 h-5 mr-2 text-blue-400" />
                        Brand Names
                      </h3>
                      <div className="space-y-1">
                        {selectedMedication.brandNames.map((brand, index) => (
                          <div key={index} className="text-gray-700 text-sm">• {brand}</div>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedMedication.dosageForms && selectedMedication.dosageForms.length > 0 && (
                    <div className="bg-gray-50 rounded-xl p-4">
                      <h3 className="font-semibold text-gray-900 mb-3">Available Forms</h3>
                      <div className="space-y-2">
                        <div>
                          <span className="font-medium text-gray-700">Dosage Forms:</span>
                          <p className="text-gray-600 text-sm">{selectedMedication.dosageForms.join(', ')}</p>
                        </div>
                        {selectedMedication.strengths && selectedMedication.strengths.length > 0 && (
                          <div>
                            <span className="font-medium text-gray-700">Strengths:</span>
                            <p className="text-gray-600 text-sm">{selectedMedication.strengths.join(', ')}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {selectedMedication.atcClass && (
                    <div className="bg-gray-50 rounded-xl p-4">
                      <h3 className="font-semibold text-gray-900 mb-3">ATC Classification</h3>
                      <p className="text-gray-700 text-sm">{selectedMedication.atcClass}</p>
                    </div>
                  )}

                  {selectedMedication.therapeuticIndications && selectedMedication.therapeuticIndications.length > 0 && (
                    <div className="bg-gray-50 rounded-xl p-4">
                      <h3 className="font-semibold text-gray-900 mb-3">Therapeutic Indications</h3>
                      <div className="space-y-1">
                        {selectedMedication.therapeuticIndications.map((indication, index) => (
                          <div key={index} className="text-gray-700 text-sm">• {indication}</div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Right Column */}
                <div className="space-y-6">
                  {/* Special Populations */}
                  {renderSpecialPopulationsSection(selectedMedication)}

                  {/* Source-specific information */}
                  {selectedMedication.usData && (
                    <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
                      <h3 className="font-semibold text-blue-900 mb-3 flex items-center">
                        <Shield className="w-5 h-5 mr-2" />
                        US Information
                      </h3>
                      <div className="space-y-2 text-sm">
                        {selectedMedication.usData.fdaApproved && (
                          <div className="flex items-center text-green-700">
                            <CheckCircle className="w-4 h-4 mr-2" />
                            <span>FDA Approved</span>
                          </div>
                        )}
                        {selectedMedication.usData.rxcui && (
                          <div>
                            <span className="font-medium text-blue-800">RxCUI:</span>
                            <span className="text-blue-700 ml-2">{selectedMedication.usData.rxcui}</span>
                          </div>
                        )}
                        {selectedMedication.usData.ndcCodes && selectedMedication.usData.ndcCodes.length > 0 && (
                          <div>
                            <span className="font-medium text-blue-800">NDC Codes:</span>
                            <p className="text-blue-700 text-xs mt-1">{selectedMedication.usData.ndcCodes.join(', ')}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {selectedMedication.ukData && (
                    <div className="bg-purple-50 rounded-xl p-4 border border-purple-200">
                      <h3 className="font-semibold text-purple-900 mb-3 flex items-center">
                        <BookOpen className="w-5 h-5 mr-2" />
                        UK Information
                      </h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center text-green-700">
                          <CheckCircle className="w-4 h-4 mr-2" />
                          <span>NHS Approved</span>
                        </div>
                        {selectedMedication.ukData.dmdId && (
                          <div>
                            <span className="font-medium text-purple-800">dm+d ID:</span>
                            <span className="text-purple-700 ml-2">{selectedMedication.ukData.dmdId}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {selectedMedication.euData && (
                    <div className="bg-orange-50 rounded-xl p-4 border border-orange-200">
                      <h3 className="font-semibold text-orange-900 mb-3 flex items-center">
                        <Globe className="w-5 h-5 mr-2" />
                        EU Information
                      </h3>
                      <div className="space-y-2 text-sm">
                        {selectedMedication.euData.emaApproved && (
                          <div className="flex items-center text-green-700">
                            <CheckCircle className="w-4 h-4 mr-2" />
                            <span>EMA Authorized</span>
                          </div>
                        )}
                        {selectedMedication.euData.authorizationStatus && (
                          <div>
                            <span className="font-medium text-orange-800">Status:</span>
                            <span className="text-orange-700 ml-2">{selectedMedication.euData.authorizationStatus}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Footer */}
              <div className="mt-8 pt-6 border-t border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <span>Source: {getSourceName(selectedMedication.source)}</span>
                    <span>Last Updated: {formatDate(selectedMedication.lastUpdated)}</span>
                  </div>
                  <div className="flex space-x-3">
                    {selectedMedication.sourceUrl && (
                      <a
                        href={selectedMedication.sourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="bg-blue-400 text-white px-4 py-2 rounded-lg hover:bg-blue-500 transition-all duration-300 font-medium"
                      >
                        <ExternalLink className="w-4 h-4 inline mr-2" />
                        View Source
                      </a>
                    )}
                    <button className="bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 transition-all duration-300 font-medium">
                      <Download className="w-4 h-4 inline mr-2" />
                      Export Info
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}

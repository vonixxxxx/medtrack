import { useState, useEffect } from 'react';
import { Header } from '../components/Header';
import { EnhancedPatientRecordsTable } from '../components/doctor/EnhancedPatientRecordsTable';
import { FilterSystem } from '../components/doctor/FilterSystem';
import { GraphBuilder } from '../components/doctor/GraphBuilder';
import { MedicalHistoryParser } from '../components/doctor/MedicalHistoryParser';
import { HbA1cAdjustmentModal } from '../components/doctor/HbA1cAdjustmentModal';
import { AnalyticsPanel } from '../components/doctor/AnalyticsPanel';
import { MetricsAnalytics } from '../components/doctor/MetricsAnalytics';
import { AIValidationPanel } from '../components/doctor/AIValidationPanel';
import api from '../api';

const DoctorDashboard = () => {
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isHbA1cModalOpen, setIsHbA1cModalOpen] = useState(false);
  const [isAIValidationOpen, setIsAIValidationOpen] = useState(false);
  const [showMetricsAnalytics, setShowMetricsAnalytics] = useState(false);
  const [patients, setPatients] = useState([]);
  const [filteredPatients, setFilteredPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [filters, setFilters] = useState({
    metric: 'all',
    dateRange: 'all',
    ethnicity: 'all',
    sex: 'all'
  });
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState(null);

  useEffect(() => {
    loadUserData();
    loadPatients();
  }, []);

  const loadUserData = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        console.log('No token found, redirecting to login');
        window.location.href = '/login';
        return;
      }
      
      const { data } = await api.get('auth/me');
      setUser(data);
    } catch (error) {
      console.error('Error loading user data:', error);
      // If auth fails, redirect to login
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
  };

  const loadPatients = async () => {
    try {
      setIsLoading(true);
      const token = localStorage.getItem('token');
      if (!token) {
        console.log('No token found, skipping patient load');
        return;
      }
      
      const { data } = await api.get('doctor/patients');
      setPatients(data);
      setFilteredPatients(data);
    } catch (error) {
      console.error('Error loading patients:', error);
      // If auth fails, redirect to login
      if (error.response?.status === 401) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = '/login';
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignOut = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    
    if (window.confirm('Are you sure you want to sign out?')) {
      window.location.href = '/';
    }
  };

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    // Apply filters to patients data
    let filtered = [...patients];
    
    if (newFilters.sex !== 'all') {
      filtered = filtered.filter(p => p.sex === newFilters.sex);
    }
    
    if (newFilters.ethnicity !== 'all') {
      filtered = filtered.filter(p => p.ethnicity === newFilters.ethnicity);
    }
    
    // Add more filtering logic based on other filters
    setFilteredPatients(filtered);
  };

  const handlePatientSelect = (patient) => {
    setSelectedPatient(patient);
  };

  const handleConditionsAdded = () => {
    // Refresh patients data to show updated conditions
    loadPatients();
  };

  // Check if user is authenticated
  const token = localStorage.getItem('token');
  if (!token) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-gray-400">Redirecting to login...</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header
        onSearchClick={() => {}}
        onProfileClick={() => setIsProfileOpen(true)}
        onSettingsClick={() => setIsSettingsOpen(true)}
        onSignOut={handleSignOut}
      />

      <main className="container mx-auto px-6 py-6 max-w-7xl">
        {/* Welcome Section */}
        <div className="mb-6">
          <h2 className="text-2xl font-semibold mb-1 text-foreground">Clinician Dashboard</h2>
          <p className="text-sm text-muted-foreground">
            Manage patient records and generate analytics
            {user?.hospitalCode && ` • Hospital Code: ${user.hospitalCode}`}
          </p>
          {selectedPatient && (
            <div className="mt-3 p-3 bg-blue-900/20 border border-blue-800 rounded-xl">
              <p className="text-sm text-blue-300">
                <strong>Selected Patient:</strong> {selectedPatient.name} 
                {selectedPatient.email && ` (${selectedPatient.email})`}
                {selectedPatient.conditions && selectedPatient.conditions.length > 0 && (
                  <span className="ml-2">
                    • {selectedPatient.conditions.length} condition{selectedPatient.conditions.length !== 1 ? 's' : ''}
                  </span>
                )}
              </p>
            </div>
          )}
        </div>

        {/* Filter System */}
        <div className="mb-6">
          <FilterSystem 
            filters={filters}
            onFilterChange={handleFilterChange}
          />
        </div>

        {/* Analytics Panel */}
        <div className="mb-6">
          <AnalyticsPanel 
            patients={filteredPatients}
            onGenerateGraph={() => {}}
          />
        </div>

        {/* Medical History Parser */}
        <div className="mb-6">
          <MedicalHistoryParser 
            selectedPatientId={selectedPatient?.id}
            onConditionsAdded={handleConditionsAdded}
          />
        </div>

        {/* AI Features */}
        <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={() => setIsAIValidationOpen(true)}
            className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-3 rounded-lg flex items-center space-x-2"
          >
            <span>🤖</span>
            <span>AI Data Validation</span>
          </button>
          <button
            onClick={() => setShowMetricsAnalytics(!showMetricsAnalytics)}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-3 rounded-lg flex items-center space-x-2"
          >
            <span>📊</span>
            <span>{showMetricsAnalytics ? 'Hide' : 'Show'} Metrics Analytics</span>
          </button>
        </div>

        {/* Metrics Analytics */}
        {showMetricsAnalytics && selectedPatient && (
          <div className="mb-6">
            <MetricsAnalytics 
              patientId={selectedPatient.id}
              patientName={selectedPatient.name}
            />
          </div>
        )}

        {/* Patient Records Table */}
        <div className="mb-6">
        <EnhancedPatientRecordsTable 
          patients={filteredPatients}
          onRefresh={loadPatients}
          onHbA1cAdjustment={() => setIsHbA1cModalOpen(true)}
          onPatientSelect={handlePatientSelect}
          selectedPatientId={selectedPatient?.id}
        />
        </div>

        {/* Graph Builder */}
        <div className="mb-6">
          <GraphBuilder 
            patients={filteredPatients}
            filters={filters}
          />
        </div>
      </main>

      {/* HbA1c Adjustment Modal */}
      <HbA1cAdjustmentModal
        isOpen={isHbA1cModalOpen}
        onClose={() => setIsHbA1cModalOpen(false)}
      />

      {/* AI Validation Panel */}
      {isAIValidationOpen && (
        <AIValidationPanel
          patientId={selectedPatient?.id}
          onClose={() => setIsAIValidationOpen(false)}
          onPatientSelected={(patientId) => {
            const patient = patients.find(p => p.id === patientId);
            if (patient) {
              setSelectedPatient(patient);
            }
          }}
        />
      )}
    </div>
  );
};

export default DoctorDashboard;



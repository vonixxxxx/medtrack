import { useState, useEffect } from 'react';
import { Header } from '../components/Header';
import { PatientRecordsTable } from '../components/doctor/PatientRecordsTable';
import { FilterSystem } from '../components/doctor/FilterSystem';
import { GraphBuilder } from '../components/doctor/GraphBuilder';
import { MedicalHistoryParser } from '../components/doctor/MedicalHistoryParser';
import { HbA1cAdjustmentModal } from '../components/doctor/HbA1cAdjustmentModal';
import { AnalyticsPanel } from '../components/doctor/AnalyticsPanel';
import api from '../api';

const DoctorDashboard = () => {
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isHbA1cModalOpen, setIsHbA1cModalOpen] = useState(false);
  const [patients, setPatients] = useState([]);
  const [filteredPatients, setFilteredPatients] = useState([]);
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
      const { data } = await api.get('auth/me');
      setUser(data);
    } catch (error) {
      console.error('Error loading user data:', error);
    }
  };

  const loadPatients = async () => {
    try {
      setIsLoading(true);
      // This will be implemented to fetch patients by hospital code
      const { data } = await api.get('doctor/patients');
      setPatients(data);
      setFilteredPatients(data);
    } catch (error) {
      console.error('Error loading patients:', error);
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
          <MedicalHistoryParser />
        </div>

        {/* Patient Records Table */}
        <div className="mb-6">
          <PatientRecordsTable 
            patients={filteredPatients}
            onRefresh={loadPatients}
            onHbA1cAdjustment={() => setIsHbA1cModalOpen(true)}
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
    </div>
  );
};

export default DoctorDashboard;



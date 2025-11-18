import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { DashboardHeader } from '../components/layout/DashboardHeader';
import { EnhancedPatientRecordsTable } from '../components/doctor/EnhancedPatientRecordsTable';
import { FilterSystem } from '../components/doctor/FilterSystem';
import { GraphBuilder } from '../components/doctor/GraphBuilder';
import { MedicalHistoryParser } from '../components/doctor/MedicalHistoryParser';
import { HbA1cAdjustmentModal } from '../components/doctor/HbA1cAdjustmentModal';
import { AnalyticsPanel } from '../components/doctor/AnalyticsPanel';
import { EncounterList } from '../components/clinician/encounters/EncounterList';
import { EncounterForm } from '../components/clinician/encounters/EncounterForm';
import { SoapNoteEditor } from '../components/clinician/soap/SoapNoteEditor';
import { ProblemList } from '../components/problems/ProblemList';
import { ProblemForm } from '../components/problems/ProblemForm';
import { AllergyList } from '../components/allergies/AllergyList';
import { AllergyForm } from '../components/allergies/AllergyForm';
import { ImmunizationList } from '../components/immunizations/ImmunizationList';
import { ImmunizationForm } from '../components/immunizations/ImmunizationForm';
import { PrescriptionList } from '../components/prescriptions/PrescriptionList';
import { PrescriptionForm } from '../components/prescriptions/PrescriptionForm';
import { ChargeCapture } from '../components/clinician/billing/ChargeCapture';
import { ChargeForm } from '../components/clinician/billing/ChargeForm';
import { Button } from '../components/ui/button';
import { useAuth } from '../contexts/AuthContext';
import { Users, BarChart3, FileText, Brain, Stethoscope } from 'lucide-react';
import api from '../api';

interface Patient {
  id: string;
  name: string;
  email?: string;
  sex?: string;
  ethnicity?: string;
  ethnic_group?: string;
  hba1cPercent?: number;
  baseline_bmi?: number;
  baseline_weight?: number;
  lastVisit?: string;
  conditions?: Array<{ id: string; name: string }>;
}

const DoctorDashboard = () => {
  const { user } = useAuth();
  const graphBuilderRef = useRef<HTMLDivElement>(null);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isHbA1cModalOpen, setIsHbA1cModalOpen] = useState(false);
  const [isEncounterFormOpen, setIsEncounterFormOpen] = useState(false);
  const [isSoapNoteOpen, setIsSoapNoteOpen] = useState(false);
  const [selectedEncounterId, setSelectedEncounterId] = useState<string | null>(null);
  const [isProblemFormOpen, setIsProblemFormOpen] = useState(false);
  const [isAllergyFormOpen, setIsAllergyFormOpen] = useState(false);
  const [isImmunizationFormOpen, setIsImmunizationFormOpen] = useState(false);
  const [isPrescriptionFormOpen, setIsPrescriptionFormOpen] = useState(false);
  const [isChargeFormOpen, setIsChargeFormOpen] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [filteredPatients, setFilteredPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [filters, setFilters] = useState({
    metric: 'all',
    dateRange: 'all',
    ethnicity: 'all',
    sex: 'all'
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (user) {
      loadPatients();
    }
  }, [user]);

  const loadPatients = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const token = localStorage.getItem('token');
      if (!token) {
        setError('Authentication required');
        return;
      }
      
      const { data } = await api.get('doctor/patients');
      setPatients(data);
      setFilteredPatients(data);
    } catch (err: any) {
      console.error('Error loading patients:', err);
      setError(err.response?.data?.error || 'Failed to load patients');
      if (err.response?.status === 401) {
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

  const handleFilterChange = (newFilters: any) => {
    setFilters(newFilters);
    let filtered = [...patients];
    
    // Sex filter
    if (newFilters.sex !== 'all') {
      filtered = filtered.filter(p => {
        const patientSex = p.sex?.toLowerCase();
        return patientSex === newFilters.sex.toLowerCase();
      });
    }
    
    // Ethnicity filter
    if (newFilters.ethnicity !== 'all') {
      filtered = filtered.filter(p => {
        const patientEthnicity = p.ethnicity?.toLowerCase() || p.ethnic_group?.toLowerCase();
        return patientEthnicity === newFilters.ethnicity.toLowerCase();
      });
    }
    
    // Metric-based filtering
    if (newFilters.metric !== 'all') {
      filtered = filtered.filter(p => {
        switch (newFilters.metric) {
          case 'hba1c':
            return p.hba1cPercent !== null && p.hba1cPercent !== undefined;
          case 'bp':
            // If blood pressure data exists (would need to check metrics)
            return true; // Placeholder - would check actual BP metrics
          case 'bmi':
            return p.baseline_bmi !== null && p.baseline_bmi !== undefined;
          case 'weight':
            return p.baseline_weight !== null && p.baseline_weight !== undefined;
          case 'glucose':
            // Would check glucose metrics
            return true; // Placeholder
          default:
            return true;
        }
      });
    }
    
    // Date range filtering (filter by lastVisit date)
    if (newFilters.dateRange !== 'all') {
      const now = new Date();
      filtered = filtered.filter(p => {
        if (!p.lastVisit) return false;
        const visitDate = new Date(p.lastVisit);
        const daysDiff = Math.floor((now.getTime() - visitDate.getTime()) / (1000 * 60 * 60 * 24));
        
        switch (newFilters.dateRange) {
          case 'today':
            return daysDiff === 0;
          case 'week':
            return daysDiff <= 7;
          case 'month':
            return daysDiff <= 30;
          case 'quarter':
            return daysDiff <= 90;
          case 'year':
            return daysDiff <= 365;
          default:
            return true;
        }
      });
    }
    
    setFilteredPatients(filtered);
  };

  const handleGenerateGraph = () => {
    // Scroll to GraphBuilder section
    if (graphBuilderRef.current) {
      graphBuilderRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    // GraphBuilder will use current filters and selected metric from AnalyticsPanel
  };

  const handlePatientSelect = (patient: Patient) => {
    setSelectedPatient(patient);
  };

  const handleConditionsAdded = () => {
    // Refresh patients data to show updated conditions
    loadPatients();
  };

  if (!user || isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50">
      <DashboardHeader
        onSearchClick={() => {}}
        onProfileClick={() => setIsProfileOpen(true)}
        onSettingsClick={() => setIsSettingsOpen(true)}
        onSignOut={handleSignOut}
        userRole="clinician"
        userName={user?.name || user?.email}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12 pt-24 lg:pt-28">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 lg:mb-12"
        >
          <div className="flex items-start justify-between mb-4">
            <div>
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-3">
                <span className="bg-gradient-to-r from-blue-600 via-blue-600 to-blue-400 bg-clip-text text-transparent">
                  Clinician Dashboard
                </span>
              </h1>
              <p className="text-xl text-gray-600">
                Manage patient records, analyze data, and deliver exceptional care
                {user?.hospitalCode && (
                  <span className="ml-3 px-4 py-1.5 bg-blue-50 border border-blue-200 text-blue-700 rounded-full text-sm font-medium">
                    {user.hospitalCode}
                  </span>
                )}
              </p>
            </div>
          </div>
          
          {selectedPatient && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 p-6 bg-white border border-blue-200 rounded-2xl shadow-lg shadow-blue-600/10 hover:shadow-xl hover:shadow-blue-600/20 transition-all"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-900 mb-2">
                    <span className="font-semibold text-blue-600">Selected Patient:</span> {selectedPatient.name} 
                    {selectedPatient.email && ` (${selectedPatient.email})`}
                  </p>
                  {selectedPatient.conditions && selectedPatient.conditions.length > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="px-3 py-1.5 bg-blue-50 text-blue-700 rounded-xl text-xs font-medium border border-blue-200">
                        {selectedPatient.conditions.length} condition{selectedPatient.conditions.length !== 1 ? 's' : ''}
                      </span>
                    </div>
                  )}
                </div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setSelectedPatient(null)}
                  className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-xl transition-all"
                >
                  Clear Selection
                </motion.button>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
          {[
            { icon: Users, label: "Total Patients", value: filteredPatients.length.toString(), gradient: "from-blue-500 to-blue-600" },
            { icon: BarChart3, label: "Analytics", value: "—", gradient: "from-blue-500 to-cyan-500" },
            { icon: FileText, label: "Records", value: "—", gradient: "from-blue-500 to-blue-500" },
            { icon: Brain, label: "AI Insights", value: "—", gradient: "from-blue-600 to-blue-400" },
          ].map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ y: -8, scale: 1.02 }}
                className="group relative p-6 bg-gradient-to-br from-white to-blue-50/30 rounded-2xl border border-blue-100 hover:border-blue-200 shadow-lg shadow-blue-600/5 hover:shadow-xl hover:shadow-blue-600/20 transition-all"
              >
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${stat.gradient} text-white mb-4 shadow-lg group-hover:shadow-xl transition-all`}>
                  <Icon size={24} />
                </div>
                <div className="text-3xl font-bold text-gray-900 mb-2">{stat.value}</div>
                <div className="text-sm text-gray-600 font-medium">{stat.label}</div>
              </motion.div>
            );
          })}
        </div>

        {/* Filter System */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <FilterSystem 
            filters={filters}
            onFilterChange={handleFilterChange}
          />
        </motion.div>

        {/* Analytics Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-8"
        >
          <AnalyticsPanel 
            patients={filteredPatients}
            onGenerateGraph={handleGenerateGraph}
          />
        </motion.div>

        {/* Medical History Parser */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-8"
        >
          <MedicalHistoryParser 
            selectedPatientId={selectedPatient?.id}
            onConditionsAdded={handleConditionsAdded}
          />
        </motion.div>

        {/* AI Features */}

        {/* Patient Records Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mb-8"
        >
          <EnhancedPatientRecordsTable 
            patients={filteredPatients}
            onRefresh={loadPatients}
            onHbA1cAdjustment={() => setIsHbA1cModalOpen(true)}
            onPatientSelect={handlePatientSelect}
            selectedPatientId={selectedPatient?.id}
          />
        </motion.div>

        {/* Patient-Specific Features */}
        {selectedPatient && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
            >
              <EncounterList
                patientId={selectedPatient.id}
                onAddEncounter={() => setIsEncounterFormOpen(true)}
                refreshTrigger={refreshTrigger}
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.72 }}
              className="flex items-center justify-center"
            >
              <Button
                onClick={() => {
                  setSelectedEncounterId(null);
                  setIsSoapNoteOpen(true);
                }}
                variant="primary"
                size="lg"
                className="w-full"
              >
                <Stethoscope className="w-5 h-5 mr-2" />
                Create SOAP Note
              </Button>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.75 }}
            >
              <ProblemList
                patientId={selectedPatient.id}
                onAddProblem={() => setIsProblemFormOpen(true)}
                refreshTrigger={refreshTrigger}
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.85 }}
            >
              <AllergyList
                patientId={selectedPatient.id}
                onAddAllergy={() => setIsAllergyFormOpen(true)}
                refreshTrigger={refreshTrigger}
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.9 }}
            >
              <ImmunizationList
                patientId={selectedPatient.id}
                onAddImmunization={() => setIsImmunizationFormOpen(true)}
                refreshTrigger={refreshTrigger}
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.95 }}
            >
              <PrescriptionList
                patientId={selectedPatient.id}
                onAddPrescription={() => setIsPrescriptionFormOpen(true)}
                refreshTrigger={refreshTrigger}
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.0 }}
            >
              <ChargeCapture
                patientId={selectedPatient.id}
                onAddCharge={() => setIsChargeFormOpen(true)}
                refreshTrigger={refreshTrigger}
              />
            </motion.div>
          </div>
        )}


        {/* Graph Builder */}
        <motion.div
          ref={graphBuilderRef}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="mb-8"
        >
          <GraphBuilder 
            patients={filteredPatients}
            filters={filters}
          />
        </motion.div>
      </main>

      {/* HbA1c Adjustment Modal */}
      <HbA1cAdjustmentModal
        isOpen={isHbA1cModalOpen}
        onClose={() => setIsHbA1cModalOpen(false)}
      />

      {/* Encounter Form */}
      <EncounterForm
        isOpen={isEncounterFormOpen}
        onClose={() => setIsEncounterFormOpen(false)}
        patientId={selectedPatient?.id}
        onSuccess={() => {
          setRefreshTrigger(prev => prev + 1);
          setIsEncounterFormOpen(false);
        }}
      />

      {/* SOAP Note Editor */}
      <SoapNoteEditor
        isOpen={isSoapNoteOpen}
        onClose={() => {
          setIsSoapNoteOpen(false);
          setSelectedEncounterId(null);
        }}
        encounterId={selectedEncounterId || undefined}
        soapNoteId={null}
        onSuccess={() => {
          setRefreshTrigger(prev => prev + 1);
          setIsSoapNoteOpen(false);
        }}
      />

      {/* Problem Form */}
      <ProblemForm
        isOpen={isProblemFormOpen}
        onClose={() => setIsProblemFormOpen(false)}
        patientId={selectedPatient?.id}
        onSuccess={() => {
          setRefreshTrigger(prev => prev + 1);
          setIsProblemFormOpen(false);
        }}
      />

      {/* Allergy Form */}
      <AllergyForm
        isOpen={isAllergyFormOpen}
        onClose={() => setIsAllergyFormOpen(false)}
        patientId={selectedPatient?.id}
        onSuccess={() => {
          setRefreshTrigger(prev => prev + 1);
          setIsAllergyFormOpen(false);
        }}
      />

      {/* Immunization Form */}
      <ImmunizationForm
        isOpen={isImmunizationFormOpen}
        onClose={() => setIsImmunizationFormOpen(false)}
        patientId={selectedPatient?.id}
        onSuccess={() => {
          setRefreshTrigger(prev => prev + 1);
          setIsImmunizationFormOpen(false);
        }}
      />

      {/* Prescription Form */}
      <PrescriptionForm
        isOpen={isPrescriptionFormOpen}
        onClose={() => setIsPrescriptionFormOpen(false)}
        patientId={selectedPatient?.id}
        onSuccess={() => {
          setRefreshTrigger(prev => prev + 1);
          setIsPrescriptionFormOpen(false);
        }}
      />

      {/* Charge Form */}
      <ChargeForm
        isOpen={isChargeFormOpen}
        onClose={() => setIsChargeFormOpen(false)}
        patientId={selectedPatient?.id}
        onSuccess={() => {
          setRefreshTrigger(prev => prev + 1);
          setIsChargeFormOpen(false);
        }}
      />

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          className="fixed bottom-6 right-6 bg-red-600 text-white px-6 py-4 rounded-2xl shadow-xl shadow-red-600/25 z-50 max-w-md border border-red-500"
        >
          <p className="font-semibold mb-2">Error</p>
          <p className="text-sm mb-3">{error}</p>
          <button 
            onClick={() => setError(null)}
            className="text-sm font-medium underline hover:no-underline transition-all"
          >
            Dismiss
          </button>
        </motion.div>
      )}
    </div>
  );
};

export default DoctorDashboard;



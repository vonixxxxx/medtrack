import { useState, useEffect } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { DashboardHeader } from "../components/layout/DashboardHeader";
import { TodaysMedications } from "../components/TodaysMedications";
import { MedicationSchedule } from "../components/MedicationSchedule";
import { EnhancedMetricHistory } from "../components/EnhancedMetricHistory";
import { AIHealthReport } from "../components/AIHealthReport";
import { FloatingAIButton } from "../components/FloatingAIButton";
import ChatModal from "../components/ChatModal";
import EnhancedMedicationChat from "../components/EnhancedMedicationChat";
import EnhancedMetricsLoggingChat from "../components/EnhancedMetricsLoggingChat";
import PostSignupSurvey from "../components/PostSignupSurvey";
import { StatCard } from "../components/ui/StatCard.jsx";
import { LoadingSkeleton } from "../components/dashboard/LoadingSkeleton.jsx";
import { PrescriptionList } from "../components/prescriptions/PrescriptionList";
import { SideEffectTracker } from "../components/side-effects/SideEffectTracker";
import { AdherenceCalendar } from "../components/adherence/AdherenceCalendar";
import { DiaryEntry } from "../components/diary/DiaryEntry";
import { PillRecognition } from "../components/pill-recognition/PillRecognition";
import { AdvancedReminderSettings } from "../components/reminders/AdvancedReminderSettings";
import { HealthReports } from "../components/health-reports/HealthReports";
import { ExportBackup } from "../components/export-backup/ExportBackup";
import { MedicationHistoryCalendar } from "../components/medication-history/MedicationHistoryCalendar";
import { MedicationListWithSideEffects } from "../components/medications/MedicationListWithSideEffects";
import { PatientAlert } from "../components/patient-alerts/PatientAlert";
import { MasonryGrid } from "../components/layout/MasonryGrid";
import { Calendar, Activity, TrendingUp, Heart, Pill } from "lucide-react";
import api from "../api";

const Dashboard = () => {
  const prefersReducedMotion = useReducedMotion();
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isAddMedOpen, setIsAddMedOpen] = useState(false);
  const [isAddMetricOpen, setIsAddMetricOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [patientId, setPatientId] = useState(null);
  const [showSurvey, setShowSurvey] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState(null);
  const [stats, setStats] = useState({
    medicationsToday: 0,
    healthMetrics: 0,
    progress: null,
    wellnessScore: null,
  });
  const [medications, setMedications] = useState([]);
  const [patientAlerts, setPatientAlerts] = useState([]);

  // Check survey completion status on component mount
  useEffect(() => {
    checkSurveyStatus();
    loadUser();
    loadStats();
  }, [refreshTrigger]);

  const loadUser = async () => {
    try {
      const userStr = localStorage.getItem('user');
      if (userStr) {
        const userData = JSON.parse(userStr);
        setUser(userData);
        
        // Get patient ID from patient profile
        try {
          const { data } = await api.get('auth/me');
          if (data.patientId) {
            setPatientId(data.patientId);
          } else {
            // Try to get patient ID from patient profile
            const patientResponse = await api.get('doctor/patients');
            const patients = patientResponse.data || [];
            const currentPatient = patients.find(p => p.userId === userData.id);
            if (currentPatient) {
              setPatientId(currentPatient.id);
            }
          }
        } catch (e) {
          console.error('Error loading patient ID:', e);
        }
      }
    } catch (e) {
      console.error('Error loading user:', e);
    }
  };

  const loadStats = async () => {
    try {
      // Fetch medications count
      const medsResponse = await api.get('meds/user');
      const meds = medsResponse.data.medications || [];
      // Filter out test medications
      const filteredMeds = meds.filter(med => {
        const name = (med.medication_name || med.name || med.generic_name || '').toLowerCase();
        return !name.includes('final test') && 
               !name.includes('test2') && 
               !name.includes('test medication');
      });
      setMedications(filteredMeds);
      const todayMeds = meds.filter(med => {
        if (!med.start_date) return true;
        const today = new Date().toISOString().split('T')[0];
        const startDate = med.start_date.split('T')[0];
        return today >= startDate;
      });

      // Fetch metrics count
      let metricsCount = 0;
      try {
        const metricsResponse = await api.get('meds/user/metrics');
        metricsCount = metricsResponse.data.metrics?.length || 0;
      } catch (e) {
        // Metrics endpoint might not exist
      }

      setStats({
        medicationsToday: todayMeds.length,
        healthMetrics: metricsCount,
        progress: null,
        wellnessScore: null,
      });
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const checkSurveyStatus = async () => {
    try {
      // Get user ID from localStorage
      const userStr = localStorage.getItem('user');
      const user = userStr ? JSON.parse(userStr) : null;
      const userId = user?.id;
      
      const { data } = await api.get('auth/survey-status', {
        params: userId ? { userId } : {}
      });
      
      console.log('Survey status check:', data);
      
      if (!data.surveyCompleted) {
        // Get user email for survey
        const { data: userData } = await api.get('auth/me');
        setUserEmail(userData.email);
        setShowSurvey(true);
      } else {
        setShowSurvey(false);
      }
    } catch (error) {
      console.error('Error checking survey status:', error);
      // On error, don't show survey to avoid blocking user
      setShowSurvey(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSurveyComplete = () => {
    setShowSurvey(false);
    // Refresh the page to ensure all data is loaded
    window.location.reload();
  };

  // Handle sign out
  const handleSignOut = () => {
    // Clear any stored authentication data
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    
    // Show confirmation
    if (window.confirm('Are you sure you want to sign out?')) {
      // Redirect to login page or home
      window.location.href = '/';
    }
  };

  // Show loading state while checking survey status
  if (isLoading) {
    return (
      <div className="min-h-screen bg-neutral-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-primary-200 border-t-primary-600 mx-auto mb-4"></div>
          <p className="text-neutral-600 font-medium">Loading your dashboard...</p>
        </div>
        {/* Survey popup should still be available during loading */}
        <PostSignupSurvey
          isOpen={showSurvey}
          onComplete={handleSurveyComplete}
          userEmail={userEmail}
        />
      </div>
    );
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05,
        delayChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: prefersReducedMotion ? 0 : 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.3,
        ease: [0.16, 1, 0.3, 1], // ease-out-quint
      },
    },
  };

  return (
    <div className="min-h-screen bg-neutral-50">
      <DashboardHeader
        onSearchClick={() => setIsChatOpen(true)}
        onProfileClick={() => setIsProfileOpen(true)}
        onSettingsClick={() => setIsSettingsOpen(true)}
        onSignOut={handleSignOut}
        userRole="patient"
        userName={user?.name || user?.email}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
        {/* Welcome Section - Apple-grade typography */}
        <motion.div
          variants={itemVariants}
          initial="hidden"
          animate="visible"
          className="mb-12"
        >
          <h1 className="text-4xl sm:text-5xl font-semibold text-neutral-900 mb-3 tracking-tight">
            Welcome back{user?.name ? `, ${user.name.split(' ')[0]}` : ''}
          </h1>
          <p className="text-lg text-neutral-600 font-normal leading-relaxed">
            Track your medications, health metrics, and stay on top of your wellness journey
          </p>
        </motion.div>

        {/* Quick Stats - Apple-inspired stat cards */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12"
        >
          <StatCard
            icon={Pill}
            label="Today's Medications"
            value={stats.medicationsToday}
            color="primary"
            delay={0.1}
          />
          <StatCard
            icon={Activity}
            label="Health Metrics"
            value={stats.healthMetrics}
            color="medical"
            delay={0.15}
          />
          <StatCard
            icon={TrendingUp}
            label="Progress"
            value={stats.progress || "—"}
            color="primary"
            delay={0.2}
          />
          <StatCard
            icon={Heart}
            label="Wellness Score"
            value={stats.wellnessScore || "—"}
            color="medical"
            delay={0.25}
          />
        </motion.div>

        {/* Dashboard Grid - True masonry layout with 2 columns */}
        <MasonryGrid
          columns={2}
          gap={24}
          className="w-full"
        >
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <TodaysMedications 
              onAddMedication={() => setIsAddMedOpen(true)} 
              refreshTrigger={refreshTrigger}
              onRefresh={() => setRefreshTrigger(prev => prev + 1)}
            />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <MedicationSchedule refreshTrigger={refreshTrigger} />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <SideEffectTracker
              patientId={patientId}
              medications={medications}
            />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <AdherenceCalendar
              medicationId={medications[0]?.id}
              patientId={patientId}
            />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <DiaryEntry
              patientId={patientId}
            />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <PillRecognition
              patientId={patientId}
            />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <EnhancedMetricHistory onAddMetric={() => setIsAddMetricOpen(true)} />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <AIHealthReport />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <HealthReports
              patientId={patientId}
              userId={user?.id}
            />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <ExportBackup
              userId={user?.id}
            />
          </motion.div>
          <motion.div variants={itemVariants} initial="hidden" animate="visible">
            <PrescriptionList
              patientId={patientId}
              readOnly={true}
              refreshTrigger={refreshTrigger}
            />
          </motion.div>
        </MasonryGrid>
      </main>

      <FloatingAIButton onClick={() => setIsChatOpen(true)} />

      {/* Modals */}
      <ChatModal
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        title="AI Health Assistant"
      />
      <EnhancedMedicationChat
        isOpen={isAddMedOpen}
        onClose={() => {
          setIsAddMedOpen(false);
        }}
        onSuccess={(medication) => {
          console.log('Medication added successfully:', medication);
          // Force refresh medications list immediately
          setRefreshTrigger(prev => prev + 1);
          // Also trigger a manual refresh after a short delay to ensure data is loaded
          setTimeout(() => {
            setRefreshTrigger(prev => prev + 1);
          }, 500);
        }}
      />
      <EnhancedMetricsLoggingChat
        isOpen={isAddMetricOpen}
        onClose={() => setIsAddMetricOpen(false)}
        onSuccess={(metrics) => {
          console.log('Metrics logged successfully:', metrics);
          setRefreshTrigger(prev => prev + 1);
          setIsAddMetricOpen(false);
        }}
      />
      <ChatModal
        isOpen={isProfileOpen}
        onClose={() => setIsProfileOpen(false)}
        title="Your Profile"
      />
      <ChatModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        title="Settings"
      />

      {/* Post-signup Survey Popup */}
      <PostSignupSurvey
        isOpen={showSurvey}
        onComplete={handleSurveyComplete}
        userEmail={userEmail}
      />

    </div>
  );
};

export default Dashboard;

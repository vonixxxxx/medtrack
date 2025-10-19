import { useState, useEffect } from "react";
import { Header } from "../components/Header";
import { TodaysMedications } from "../components/TodaysMedications";
import { MedicationSchedule } from "../components/MedicationSchedule";
import { EnhancedMetricHistory } from "../components/EnhancedMetricHistory";
import { AIHealthReport } from "../components/AIHealthReport";
import { FloatingAIButton } from "../components/FloatingAIButton";
import ChatModal from "../components/ChatModal";
import EnhancedMedicationChat from "../components/EnhancedMedicationChat";
import EnhancedMetricsLoggingChat from "../components/EnhancedMetricsLoggingChat";
import PostSignupSurvey from "../components/PostSignupSurvey";
import api from "../api";

const Dashboard = () => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isAddMedOpen, setIsAddMedOpen] = useState(false);
  const [isAddMetricOpen, setIsAddMetricOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [showSurvey, setShowSurvey] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // Check survey completion status on component mount
  useEffect(() => {
    checkSurveyStatus();
  }, []);

  const checkSurveyStatus = async () => {
    try {
      const { data } = await api.get('auth/survey-status');
      if (!data.surveyCompleted) {
        // Get user email for survey
        const userData = await api.get('auth/me');
        setUserEmail(userData.email);
        setShowSurvey(true);
      }
    } catch (error) {
      console.error('Error checking survey status:', error);
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
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header
        onSearchClick={() => setIsChatOpen(true)}
        onProfileClick={() => setIsProfileOpen(true)}
        onSettingsClick={() => setIsSettingsOpen(true)}
        onSignOut={handleSignOut}
      />

      <main className="container mx-auto px-6 py-6 max-w-7xl">
        {/* Welcome Section */}
        <div className="mb-6">
          <h2 className="text-2xl font-semibold mb-1 text-foreground">Dashboard</h2>
          <p className="text-sm text-muted-foreground">Track your medications and health metrics</p>
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <TodaysMedications 
            onAddMedication={() => setIsAddMedOpen(true)} 
            refreshTrigger={refreshTrigger}
            onRefresh={() => setRefreshTrigger(prev => prev + 1)}
          />
          <MedicationSchedule refreshTrigger={refreshTrigger} />
          <EnhancedMetricHistory onAddMetric={() => setIsAddMetricOpen(true)} />
          <AIHealthReport />
        </div>
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
        onClose={() => setIsAddMedOpen(false)}
        onSuccess={(medication) => {
          console.log('Medication added successfully:', medication);
          // Refresh medications list
          setRefreshTrigger(prev => prev + 1);
          // Close the modal
          setIsAddMedOpen(false);
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

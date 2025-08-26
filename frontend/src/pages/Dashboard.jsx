import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import Navigation from '../components/Navigation';
import MedicationValidationPopup from '../components/MedicationValidationPopup';
import AddMetricPopup from '../components/AddMetricPopup';
import { Plus, Activity, Eye, TrendingUp, Heart, Target, Calendar, Timer, AlertCircle, CheckCircle } from 'lucide-react';

export default function Dashboard() {
  const navigate = useNavigate();
  
  // State for medication validation popup
  const [isMedicationPopupOpen, setIsMedicationPopupOpen] = useState(false);
  
  // State for metric popup
  const [isMetricPopupOpen, setIsMetricPopupOpen] = useState(false);
  
  // Real data state - loaded from localStorage
  const [userData, setUserData] = useState({
    medications: [],
    metrics: [],
    lastActivity: [],
    upcomingEvents: []
  });

  // Dashboard metrics calculated from real data
  const [dashboardMetrics, setDashboardMetrics] = useState({
    activeMedications: 0,
    totalMetrics: 0,
    overdueMetrics: 0,
    upcomingDoses: 0,
    adherenceRate: 0,
    nextCheckup: null,
    nextRefill: null
  });

  const handleMedicationSelected = (medicationData) => {
    // Ensure medications array exists
    const currentMedications = userData.medications || [];
    
    // Create new medication object with all the new fields
    const newMedication = {
      id: Date.now(), // Simple ID generation
      name: medicationData.name,
      generic: medicationData.generic || '',
      brand: medicationData.brand || '',
      class: medicationData.class || '',
      dosage: medicationData.dosage,
      form: medicationData.form,
      frequency: medicationData.frequency,
      startDate: medicationData.startDate,
      endDate: medicationData.endDate || null,
      notes: medicationData.notes || '',
      monitoringMetrics: medicationData.monitoringMetrics || [],
      metricUpdateFrequency: medicationData.metricUpdateFrequency || 'daily',
      intakeType: medicationData.intakeType || 'tablet',
      intakePlace: medicationData.intakePlace || 'at home',
      customValues: medicationData.customValues || {
        dose: false,
        frequency: false,
        intakeType: false,
        intakePlace: false
      },
      isActive: true,
      nextDose: new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString(), // 4 hours from now
      doseLogs: [],
      source: medicationData.source,
      addedAt: new Date().toISOString()
    };

    // Add to user data
    const updatedMedications = [...currentMedications, newMedication];
    const updatedUserData = { ...userData, medications: updatedMedications };
    
    setUserData(updatedUserData);
    
    // Save to localStorage
    localStorage.setItem('medtrack_dashboard_data', JSON.stringify(updatedUserData));
    
    // Recalculate metrics
    calculateDashboardMetrics(updatedUserData);
    
    // Close popup
    setIsMedicationPopupOpen(false);
  };

  const handleMetricAdded = (metricData) => {
    // Add to user data
    const updatedMetrics = [...userData.metrics, metricData];
    const updatedUserData = { ...userData, metrics: updatedMetrics };
    
    setUserData(updatedUserData);
    
    // Save to localStorage
    localStorage.setItem('medtrack_dashboard_data', JSON.stringify(updatedUserData));
    
    // Recalculate metrics
    calculateDashboardMetrics(updatedUserData);
    
    // Close popup
    setIsMetricPopupOpen(false);
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = () => {
    try {
      // Load signup data
      const signupData = localStorage.getItem('signupData');
      const parsedSignupData = signupData ? JSON.parse(signupData) : {};
      
      // Load updated metrics
      const updatedMetrics = localStorage.getItem('medtrack_updated_metrics');
      const parsedMetrics = updatedMetrics ? JSON.parse(updatedMetrics) : {};
      
      // Load any additional dashboard data
      const dashboardData = localStorage.getItem('medtrack_dashboard_data');
      const parsedDashboardData = dashboardData ? JSON.parse(dashboardData) : {};
      
      // Combine all data
      const combinedData = {
        ...parsedSignupData,
        ...parsedMetrics,
        ...parsedDashboardData
      };
      
      // If no data exists, initialize with sample data to demonstrate functionality
      if (!signupData && !updatedMetrics && !dashboardData) {
        initializeSampleDashboardData();
        return;
      }
      
      setUserData(combinedData);
      calculateDashboardMetrics(combinedData);
      
    } catch (error) {
      console.log('Error loading dashboard data:', error);
    }
  };

  const initializeSampleDashboardData = () => {
    const sampleData = {
      medications: [
        {
          id: 1,
          name: 'Metformin',
          dosage: '500mg',
          isActive: true,
          frequency: 1, // daily
          nextDose: new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString(), // 4 hours from now
          doseLogs: [
            { timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString() }, // yesterday
            { timestamp: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString() }, // 2 days ago
            { timestamp: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString() }, // 3 days ago
            { timestamp: new Date(Date.now() - 96 * 60 * 60 * 1000).toISOString() }, // 4 days ago
            { timestamp: new Date(Date.now() - 120 * 60 * 60 * 1000).toISOString() }, // 5 days ago
            { timestamp: new Date(Date.now() - 144 * 60 * 60 * 1000).toISOString() }, // 6 days ago
            { timestamp: new Date(Date.now() - 168 * 60 * 60 * 1000).toISOString() }  // 7 days ago
          ]
        },
        {
          id: 2,
          name: 'Aspirin',
          dosage: '81mg',
          isActive: true,
          frequency: 1, // daily
          nextDose: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(), // 8 hours from now
          doseLogs: [
            { timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString() },
            { timestamp: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString() },
            { timestamp: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString() },
            { timestamp: new Date(Date.now() - 96 * 60 * 60 * 1000).toISOString() },
            { timestamp: new Date(Date.now() - 120 * 60 * 60 * 1000).toISOString() },
            { timestamp: new Date(Date.now() - 144 * 60 * 60 * 1000).toISOString() },
            { timestamp: new Date(Date.now() - 168 * 60 * 60 * 1000).toISOString() }
          ]
        }
      ],
      metrics: [
        {
          id: 1,
          name: 'Blood Pressure',
          lastUpdated: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 days ago
          frequency: 7 // weekly
        },
        {
          id: 2,
          name: 'Weight',
          lastUpdated: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), // 5 days ago
          frequency: 7 // weekly
        },
        {
          id: 3,
          name: 'Blood Sugar',
          lastUpdated: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), // 1 day ago
          frequency: 3 // every 3 days
        }
      ],
      lastActivity: [
        {
          id: 1,
          description: 'Metformin taken',
          timestamp: '2 hours ago'
        },
        {
          id: 2,
          description: 'Blood pressure logged',
          timestamp: '2 days ago'
        },
        {
          id: 3,
          description: 'Weight updated',
          timestamp: '5 days ago'
        }
      ],
      nextCheckup: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString(), // 2 weeks from now
      nextRefill: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() // 1 week from now
    };
    
    // Save sample data to localStorage
    localStorage.setItem('medtrack_dashboard_data', JSON.stringify(sampleData));
    
    setUserData(sampleData);
    calculateDashboardMetrics(sampleData);
  };

  const calculateDashboardMetrics = (data) => {
    const today = new Date();
    
    // Calculate active medications
    const activeMeds = data.medications?.filter(m => m.isActive) || [];
    
    // Calculate overdue metrics (metrics that haven't been updated in their frequency period)
    const overdueMetrics = (data.metrics || []).filter(metric => {
      if (!metric.lastUpdated || !metric.frequency) return false;
      const lastUpdate = new Date(metric.lastUpdated);
      const daysSinceUpdate = (today - lastUpdate) / (1000 * 60 * 60 * 24);
      return daysSinceUpdate > metric.frequency;
    });
    
    // Calculate upcoming doses (next 24 hours)
    const upcomingDoses = (data.medications || []).filter(med => {
      if (!med.nextDose) return false;
      const nextDose = new Date(med.nextDose);
      const timeUntilDose = nextDose - today;
      return timeUntilDose > 0 && timeUntilDose <= 24 * 60 * 60 * 1000;
    });
    
    // Calculate adherence rate (based on taken vs scheduled doses in last 7 days)
    const adherenceRate = calculateAdherenceRate(data.medications || []);
    
    // Get next checkup and refill dates
    const nextCheckup = data.nextCheckup ? new Date(data.nextCheckup) : null;
    const nextRefill = data.nextRefill ? new Date(data.nextRefill) : null;
    
    setDashboardMetrics({
      activeMedications: activeMeds.length,
      totalMetrics: data.metrics?.length || 0,
      overdueMetrics: overdueMetrics.length,
      upcomingDoses: upcomingDoses.length,
      adherenceRate,
      nextCheckup,
      nextRefill
    });
  };

  const calculateAdherenceRate = (medications) => {
    if (!medications.length) return 100;
    
    const today = new Date();
    const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
    
    let totalScheduled = 0;
    let totalTaken = 0;
    
    medications.forEach(med => {
      if (med.isActive && med.frequency) {
        const scheduledDoses = Math.floor(7 / med.frequency);
        totalScheduled += scheduledDoses;
        
        // Count taken doses in the last week (this would come from dose logs)
        const takenDoses = med.doseLogs?.filter(log => {
          const logDate = new Date(log.timestamp);
          return logDate >= weekAgo && logDate <= today;
        }).length || 0;
        
        totalTaken += takenDoses;
      }
    });
    
    return totalScheduled > 0 ? Math.round((totalTaken / totalScheduled) * 100) : 100;
  };

  const handleQuickAction = (action) => {
    switch (action) {
      case 'addMedication':
        setIsMedicationPopupOpen(true);
        break;
      case 'logMetrics':
        setIsMetricPopupOpen(true);
        break;
      case 'viewHistory':
        navigate('/demographics');
        break;
      default:
        break;
    }
  };

  const getAdherenceMessage = (rate) => {
    if (rate >= 90) return "Excellent adherence! Keep it up!";
    if (rate >= 75) return "Good adherence. You're doing well!";
    if (rate >= 60) return "Fair adherence. Try to improve consistency.";
    return "Low adherence. Consider setting reminders.";
  };

  const getHealthTrendMessage = () => {
    const hasRecentMetrics = userData.metrics?.some(m => {
      if (!m.lastUpdated) return false;
      const lastUpdate = new Date(m.lastUpdated);
      const daysSinceUpdate = (Date.now() - lastUpdate.getTime()) / (1000 * 60 * 60 * 24);
      return daysSinceUpdate <= 7;
    });
    
    return hasRecentMetrics 
      ? "Regular monitoring detected. Great job!" 
      : "Consider logging your metrics regularly.";
  };

  const formatTimeUntil = (date) => {
    if (!date) return 'Not scheduled';
    
    const now = new Date();
    const targetDate = new Date(date);
    const diffTime = targetDate - now;
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays < 0) return 'Overdue';
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Tomorrow';
    if (diffDays < 7) return `In ${diffDays} days`;
    if (diffDays < 30) return `In ${Math.ceil(diffDays / 7)} weeks`;
    return `In ${Math.ceil(diffDays / 30)} months`;
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <Navigation />
      
      <motion.div 
        className="max-w-7xl mx-auto px-6 py-8"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Header */}
        <motion.div 
          className="mb-8 text-center"
          variants={cardVariants}
        >
          <h1 className="text-4xl font-bold text-blue-400 mb-3">Dashboard</h1>
        </motion.div>

        {/* Key Metrics Overview */}
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
          variants={cardVariants}
        >
          <div className="bg-white rounded-2xl p-6 text-center shadow-lg border border-gray-200">
            <div className="text-3xl font-bold text-blue-400 mb-2">{dashboardMetrics.activeMedications}</div>
            <div className="text-sm font-medium text-gray-400">Active Medications</div>
          </div>
          
          <div className="bg-white rounded-2xl p-6 text-center shadow-lg border border-gray-200">
            <div className="text-3xl font-bold text-blue-400 mb-2">{dashboardMetrics.totalMetrics}</div>
            <div className="text-sm font-medium text-gray-400">Health Metrics</div>
          </div>
          
          <div className="bg-white rounded-2xl p-6 text-center shadow-lg border border-gray-200">
            <div className="text-3xl font-bold text-blue-400 mb-2">{dashboardMetrics.overdueMetrics}</div>
            <div className="text-sm font-medium text-gray-400">Overdue</div>
          </div>
          
          <div className="bg-white rounded-2xl p-6 text-center shadow-lg border border-gray-200">
            <div className="text-3xl font-bold text-blue-400 mb-2">{dashboardMetrics.upcomingDoses}</div>
            <div className="text-sm font-medium text-gray-400">Next Doses</div>
          </div>
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            {/* Today's Overview */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Today's Overview
              </h2>
              
              <div className="space-y-4">
                <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                  <h3 className="text-lg font-semibold text-gray-600 mb-3">Medication Schedule</h3>
                  {dashboardMetrics.activeMedications > 0 ? (
                    <div className="space-y-3">
                      {userData.medications?.slice(0, 3).map((med, index) => (
                        <div key={index} className="flex items-center space-x-3">
                          <div className="w-3 h-3 bg-gradient-to-r from-blue-300 to-blue-400 rounded-full"></div>
                          <span className="text-gray-500 font-medium">{med.name}</span>
                          <span className="text-gray-400 text-sm">{med.dosage}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-400 text-center py-4">No medications scheduled today</p>
                  )}
                </div>

                <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                  <h3 className="text-lg font-semibold text-gray-600 mb-3">Health Reminders</h3>
                  {dashboardMetrics.overdueMetrics > 0 ? (
                    <div className="space-y-3">
                      {userData.metrics?.filter(m => {
                        if (!m.lastUpdated || !m.frequency) return false;
                        const lastUpdate = new Date(m.lastUpdated);
                        const daysSinceUpdate = (Date.now() - lastUpdate.getTime()) / (1000 * 60 * 60 * 24);
                        return daysSinceUpdate > m.frequency;
                      }).slice(0, 3).map((metric, index) => (
                        <div key={index} className="flex items-center space-x-3">
                          <AlertCircle className="w-4 h-4 text-orange-500" />
                          <span className="text-gray-500 font-medium">{metric.name}</span>
                          <span className="text-orange-500 text-sm font-medium">Overdue</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-center justify-center space-x-2 py-4 text-blue-600">
                      <CheckCircle className="w-5 h-5" />
                      <span className="text-sm font-medium">All metrics up to date</span>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>

            {/* Recent Activity */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Recent Activity
              </h2>
              
              {userData.lastActivity?.length > 0 ? (
                <div className="space-y-4">
                  {userData.lastActivity.slice(0, 3).map((activity, index) => (
                    <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-xl">
                      <div className="w-3 h-3 bg-gradient-to-r from-blue-300 to-blue-400 rounded-full"></div>
                      <div>
                        <p className="font-medium text-gray-500">{activity.description}</p>
                        <p className="text-sm text-gray-400">{activity.timestamp}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <Activity className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                  <p className="text-lg font-medium text-gray-500">No recent activity</p>
                  <p className="text-sm text-gray-400">Start using MedTrack to see your activity here</p>
                </div>
              )}
            </motion.div>
          </div>

          {/* Right Column - Quick Actions & Insights */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm text-center"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center justify-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Quick Actions
              </h2>
              <div className="space-y-4">
                <button 
                  onClick={() => handleQuickAction('addMedication')}
                  className="w-full bg-white text-blue-400 py-3 rounded-xl border border-blue-200 transition-all duration-300 font-medium flex items-center justify-center space-x-2 cursor-pointer relative overflow-hidden group hover:border-transparent hover:bg-blue-50"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-100 to-blue-200 opacity-0 group-hover:opacity-100 transition-all duration-500 ease-out transform -translate-x-full group-hover:translate-x-0"></div>
                  <Plus className="w-4 h-4 relative z-10" />
                  <span className="relative z-10">Add Medication</span>
                </button>
                <button 
                  onClick={() => handleQuickAction('logMetrics')}
                  className="w-full bg-white text-blue-400 py-3 rounded-xl border border-blue-200 transition-all duration-300 font-medium flex items-center justify-center space-x-2 cursor-pointer relative overflow-hidden group hover:border-transparent hover:bg-blue-50"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-100 to-blue-200 opacity-0 group-hover:opacity-100 transition-all duration-500 ease-out transform -translate-x-full group-hover:translate-x-0"></div>
                  <Activity className="w-4 h-4 relative z-10" />
                  <span className="relative z-10">Log Metrics</span>
                </button>
                <button 
                  onClick={() => handleQuickAction('viewHistory')}
                  className="w-full bg-white text-blue-400 py-3 rounded-xl border border-blue-200 transition-all duration-300 font-medium flex items-center justify-center space-x-2 cursor-pointer relative overflow-hidden group hover:border-transparent hover:bg-blue-50"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-100 to-blue-200 opacity-0 group-hover:opacity-100 transition-all duration-500 ease-out transform -translate-x-full group-hover:translate-x-0"></div>
                  <Eye className="w-4 h-4 relative z-10" />
                  <span className="relative z-10">View History</span>
                </button>
              </div>
            </motion.div>

            {/* Health Insights */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Health Insights
              </h2>
              
              <div className="space-y-4">
                <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                  <div className="flex items-center space-x-3">
                    <TrendingUp className="w-6 h-6 text-blue-400" />
                    <div>
                      <p className="font-medium text-gray-600">Medication Adherence</p>
                      <p className="text-sm text-gray-400">{getAdherenceMessage(dashboardMetrics.adherenceRate)}</p>
                      <p className="text-xs text-blue-600 font-medium">{dashboardMetrics.adherenceRate}% adherence rate</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                  <div className="flex items-center space-x-3">
                    <Heart className="w-6 h-6 text-blue-400" />
                    <div>
                      <p className="font-medium text-gray-600">Health Trends</p>
                      <p className="text-sm text-gray-400">{getHealthTrendMessage()}</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                  <div className="flex items-center space-x-3">
                    <Target className="w-6 h-6 text-blue-400" />
                    <div>
                      <p className="font-medium text-gray-600">Goals</p>
                      <p className="text-sm text-gray-400">Set health targets</p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Upcoming Events */}
            <motion.div 
              className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm"
              variants={cardVariants}
            >
              <h2 className="text-xl font-bold text-blue-400 mb-4 flex items-center">
                <div className="w-2 h-6 bg-blue-300 rounded-full mr-3"></div>
                Upcoming Events
              </h2>
              
              <div className="space-y-4">
                <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-xl">
                  <Calendar className="w-5 h-5 text-blue-400" />
                  <div>
                    <p className="font-medium text-gray-600">Next Check-up</p>
                    <p className="text-sm text-gray-400">{formatTimeUntil(dashboardMetrics.nextCheckup)}</p>
                  </div>
                </div>

                <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-xl">
                  <Timer className="w-5 h-5 text-blue-400" />
                  <div>
                    <p className="font-medium text-gray-600">Medication Refill</p>
                    <p className="text-sm text-gray-400">{formatTimeUntil(dashboardMetrics.nextRefill)}</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* Medication Validation Popup */}
      {isMedicationPopupOpen && (
        <MedicationValidationPopup
          isOpen={isMedicationPopupOpen}
          onClose={() => setIsMedicationPopupOpen(false)}
          onMedicationSelected={handleMedicationSelected}
        />
      )}

      {/* Add Metric Popup */}
      {isMetricPopupOpen && (
        <AddMetricPopup
          isOpen={isMetricPopupOpen}
          onClose={() => setIsMetricPopupOpen(false)}
          onMetricAdded={handleMetricAdded}
        />
      )}
    </div>
  );
}

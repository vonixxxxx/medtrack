import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send, Bot, User, CheckCircle, AlertCircle, Loader2, Activity, TrendingUp, Heart, Thermometer } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import api from '../api';

const EnhancedMetricsLoggingChat = ({ isOpen, onClose, onSuccess }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationState, setConversationState] = useState('welcome');
  const [userMedications, setUserMedications] = useState([]);
  const [selectedMedication, setSelectedMedication] = useState(null);
  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [currentMetric, setCurrentMetric] = useState(null);
  const [loggedMetrics, setLoggedMetrics] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Available metrics for tracking
  const allMetrics = [
    'Blood Pressure', 'Heart Rate', 'Blood Glucose', 'Weight', 'Temperature',
    'Pain Level', 'Sleep Quality', 'Mood', 'Energy Level', 'Side Effects',
    'Blood Sugar', 'Cholesterol', 'Blood Oxygen', 'General Health',
    'Blood Pressure (Systolic)', 'Blood Pressure (Diastolic)', 'BMI',
    'Waist Circumference', 'Hip Circumference', 'Body Fat Percentage',
    'Muscle Mass', 'Bone Density', 'Vitamin D Level', 'Iron Level',
    'Thyroid Function', 'Kidney Function', 'Liver Function', 'Blood Count',
    'Inflammation Markers', 'Allergy Symptoms', 'Digestive Health',
    'Mental Health', 'Cognitive Function', 'Physical Activity',
    'Exercise Duration', 'Exercise Intensity', 'Steps Count',
    'Calories Burned', 'Water Intake', 'Alcohol Consumption',
    'Caffeine Intake', 'Smoking Status', 'Stress Level', 'Anxiety Level',
    'Depression Score', 'Quality of Life', 'Medication Adherence',
    'Drug Interactions', 'Allergic Reactions', 'Emergency Symptoms'
  ];

  // Scroll to bottom when new messages are added
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Real-time metric due checking
  useEffect(() => {
    const checkMetricsDue = () => {
      if (userMedications.length > 0) {
        const dueNow = userMedications.filter(areMetricsDueNow);
        console.log(`[Metric Check] ${dueNow.length} medications due for metrics right now`);
        console.log(`[Metric Check] Current time: ${new Date().toLocaleString()}`);
        console.log(`[Metric Check] Medications checked:`, userMedications.map(med => ({
          name: med.medication_name || med.name,
          frequency: med.metrics_tracking_frequency,
          lastLog: med.last_metric_log,
          due: areMetricsDueNow(med)
        })));
        
        // Update conversation state if no metrics are due
        if (dueNow.length === 0 && conversationState === 'medication_selected') {
          setConversationState('no_metrics_due');
        }
      }
    };

    // Check immediately
    checkMetricsDue();
    
    // Check every minute for real-time updates
    const interval = setInterval(checkMetricsDue, 60000);
    
    return () => clearInterval(interval);
  }, [userMedications, conversationState]);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current.focus(), 100);
    }
  }, [isOpen]);

  // Initialize conversation
  useEffect(() => {
    if (isOpen) {
      setMessages([
        {
          id: `welcome-${Date.now()}`,
          type: 'bot',
          content: "Hi! I'll help you log your health metrics for today. Let me first check which medications you're taking so I can suggest the right metrics to track.",
          timestamp: new Date(),
          showTyping: false
        }
      ]);
      setConversationState('medication_selection');
      setSelectedMedication(null);
      setSelectedMetrics([]);
      setCurrentMetric(null);
      setLoggedMetrics([]);
      
      // Automatically start medication selection
      handleMedicationSelection();
    }
  }, [isOpen]);

  // Add message with animation
  const addMessage = (type, content, options = {}) => {
    const message = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      content,
      timestamp: new Date(),
      showTyping: false,
      ...options
    };
    
    setMessages(prev => [...prev, message]);
  };

  // Fetch user medications
  const fetchMedications = async () => {
    try {
      const response = await api.get('meds/user');
      return response.data.medications || [];
    } catch (error) {
      console.error('Error fetching medications:', error);
      return [];
    }
  };

  // Parse natural language metric input using Llama
  const parseMetricInput = (metricText, metricType) => {
    // Handle blood pressure (systolic/diastolic)
    if (metricType.toLowerCase().includes('blood pressure')) {
      const bpRegex = /(\d+)\s*\/\s*(\d+)/;
      const match = metricText.match(bpRegex);
      
      if (match) {
        return {
          success: true,
          systolic: parseInt(match[1]),
          diastolic: parseInt(match[2]),
          value: `${match[1]}/${match[2]}`,
          unit: 'mmHg'
        };
      } else {
        // Ask for both values
        const singleValue = parseInt(metricText);
        if (!isNaN(singleValue)) {
          return {
            success: false,
            needsBoth: true,
            partialValue: singleValue
          };
        }
      }
    }

    // Handle other metrics
    const valueRegex = /(\d+(?:\.\d+)?)\s*(mg\/dl|mg\/dL|bpm|°f|°c|kg|lbs|ml|glasses|hours?|steps?|cal|%)?/i;
    const match = metricText.match(valueRegex);
    
    if (match) {
      const [, amount, unit] = match;
      const normalizedUnit = unit?.toLowerCase() || getDefaultUnit(metricType);
      
      return {
        success: true,
        value: parseFloat(amount),
        unit: normalizedUnit
      };
    }

    return {
      success: false,
      error: 'Please enter a valid number with unit (e.g., "120", "98.6°F", "75 bpm")'
    };
  };

  // Get default unit for metric type
  const getDefaultUnit = (metricType) => {
    const unitMap = {
      'blood pressure': 'mmHg',
      'heart rate': 'bpm',
      'blood glucose': 'mg/dL',
      'weight': 'kg',
      'temperature': '°F',
      'pain level': '/10',
      'sleep quality': '/10',
      'mood': '/10',
      'energy level': '/10',
      'bmi': 'kg/m²',
      'cholesterol': 'mg/dL',
      'blood oxygen': '%',
      'steps count': 'steps',
      'calories burned': 'cal',
      'water intake': 'glasses'
    };

    const normalizedType = metricType.toLowerCase();
    for (const [key, unit] of Object.entries(unitMap)) {
      if (normalizedType.includes(key)) {
        return unit;
      }
    }
    return '';
  };

  // Validate metric value ranges
  const validateMetricValue = (metricType, value, unit) => {
    const validationRules = {
      'blood pressure': { min: 50, max: 250, unit: 'mmHg' },
      'blood pressure (systolic)': { min: 70, max: 250, unit: 'mmHg' },
      'blood pressure (diastolic)': { min: 40, max: 150, unit: 'mmHg' },
      'heart rate': { min: 30, max: 220, unit: 'bpm' },
      'blood glucose': { min: 20, max: 600, unit: 'mg/dL' },
      'blood sugar': { min: 20, max: 600, unit: 'mg/dL' },
      'weight': { min: 20, max: 500, unit: 'lbs' },
      'temperature': { min: 95, max: 110, unit: '°F' },
      'pain level': { min: 0, max: 10, unit: '/10' },
      'sleep quality': { min: 0, max: 10, unit: '/10' },
      'mood': { min: 0, max: 10, unit: '/10' },
      'energy level': { min: 0, max: 10, unit: '/10' },
      'bmi': { min: 10, max: 50, unit: 'kg/m²' },
      'cholesterol': { min: 50, max: 400, unit: 'mg/dL' },
      'blood oxygen': { min: 70, max: 100, unit: '%' },
      'steps count': { min: 0, max: 50000, unit: 'steps' },
      'calories burned': { min: 0, max: 10000, unit: 'cal' },
      'water intake': { min: 0, max: 20, unit: 'glasses' },
      'waist circumference': { min: 20, max: 80, unit: 'inches' },
      'hip circumference': { min: 25, max: 100, unit: 'inches' },
      'body fat percentage': { min: 3, max: 50, unit: '%' },
      'muscle mass': { min: 20, max: 200, unit: 'lbs' },
      'bone density': { min: 0.5, max: 2.0, unit: 'g/cm²' },
      'vitamin d level': { min: 10, max: 100, unit: 'ng/mL' },
      'iron level': { min: 10, max: 200, unit: 'μg/dL' },
      'thyroid function': { min: 0.1, max: 10, unit: 'mIU/L' },
      'kidney function': { min: 10, max: 200, unit: 'mL/min' },
      'liver function': { min: 5, max: 100, unit: 'U/L' },
      'blood count': { min: 3, max: 20, unit: 'million/μL' },
      'inflammation markers': { min: 0, max: 10, unit: 'mg/L' },
      'stress level': { min: 0, max: 10, unit: '/10' },
      'anxiety level': { min: 0, max: 10, unit: '/10' },
      'depression score': { min: 0, max: 30, unit: 'points' },
      'quality of life': { min: 0, max: 10, unit: '/10' },
      'medication adherence': { min: 0, max: 100, unit: '%' }
    };

    const rule = validationRules[metricType.toLowerCase()];
    if (!rule) return { valid: true, value, unit };

    if (value < rule.min || value > rule.max) {
      return { 
        valid: false, 
        error: `${metricType} should be between ${rule.min} and ${rule.max} ${rule.unit}. Please enter a valid value.` 
      };
    }

    return { valid: true, value, unit };
  };

  // Check if metrics are due right now for a medication based on local time
  const areMetricsDueNow = (medication) => {
    if (!medication.start_date) return true; // If no start date, assume due
    
    const now = new Date();
    const today = now.toISOString().split('T')[0];
    const startDate = medication.start_date.split('T')[0];
    
    // Check if medication started today or before
    if (today < startDate) return false;
    
    // Check if metrics tracking is enabled for this medication
    if (!medication.metrics_tracking_frequency) return false;
    
    // Check frequency-based scheduling with time awareness
    const frequency = medication.metrics_tracking_frequency;
    const currentHour = now.getHours();
    const currentMinute = now.getMinutes();
    const currentTime = currentHour * 60 + currentMinute;
    
    switch (frequency) {
      case 'daily':
        // Check if it's been 24 hours since last metric log
        if (medication.last_metric_log) {
          const lastLog = new Date(medication.last_metric_log);
          const hoursSinceLastLog = (now - lastLog) / (1000 * 60 * 60);
          return hoursSinceLastLog >= 24;
        }
        return true; // First time logging
        
      case 'twice_daily':
        // Check if it's been 12 hours since last metric log
        if (medication.last_metric_log) {
          const lastLog = new Date(medication.last_metric_log);
          const hoursSinceLastLog = (now - lastLog) / (1000 * 60 * 60);
          return hoursSinceLastLog >= 12;
        }
        return true; // First time logging
        
      case 'three_times_daily':
        // Check if it's been 8 hours since last metric log
        if (medication.last_metric_log) {
          const lastLog = new Date(medication.last_metric_log);
          const hoursSinceLastLog = (now - lastLog) / (1000 * 60 * 60);
          return hoursSinceLastLog >= 8;
        }
        return true; // First time logging
        
      case 'weekly':
        // Check if it's been a week since last metric log
        if (medication.last_metric_log) {
          const lastLog = new Date(medication.last_metric_log);
          const daysSinceLastLog = (now - lastLog) / (1000 * 60 * 60 * 24);
          return daysSinceLastLog >= 7;
        }
        return true; // First time logging
        
      case 'as_needed':
        // Always allow logging for as-needed medications
        return true;
      case 'monthly':
        // Check if it's been a month since start date
        const startMonth = new Date(startDate);
        const monthsDiff = (now.getFullYear() - startMonth.getFullYear()) * 12 + 
                          (now.getMonth() - startMonth.getMonth());
        return monthsDiff >= 1;
      case 'as_needed':
        return true; // As needed medications can always be logged
      default:
        return true;
    }
  };

  // Handle medication selection
  const handleMedicationSelection = async () => {
    setIsLoading(true);
    setIsTyping(true);
    
    try {
      const medications = await fetchMedications();
      setUserMedications(medications);
      
      if (medications.length === 0) {
        addMessage('bot', "You don't have any medications added yet. Please add a medication first before logging metrics.", {
          type: 'error',
          actions: [
            { text: 'Add Medication', action: 'add_medication' }
          ]
        });
        setConversationState('no_medications');
      } else {
        // Filter medications that are due for metrics right now
        const dueMedications = medications.filter(areMetricsDueNow);
        
        if (dueMedications.length === 0) {
          addMessage('bot', "No metrics due right now! Your medications don't require metric logging at this time based on their schedule.", {
            type: 'info',
            actions: [
              { text: 'Add Medication', action: 'add_medication' }
            ]
          });
          setConversationState('no_metrics_due');
        } else {
          addMessage('bot', `Great! I found ${dueMedications.length} medication(s) that need metrics logged right now. Which one would you like to log metrics for?`, {
            type: 'success',
            actions: dueMedications.map(med => ({
              text: med.medication_name || med.name || med.generic_name,
              action: 'select_medication',
              data: med
            }))
          });
          setConversationState('medication_selection');
        }
      }
    } catch (error) {
      addMessage('bot', "I'm sorry, there was an error loading your medications. Please try again.", {
        type: 'error',
        actions: [
          { text: 'Try again', action: 'retry_medication_selection' }
        ]
      });
    } finally {
      setIsLoading(false);
      setIsTyping(false);
    }
  };

  // Handle metric value input
  const handleMetricValue = (metric, value) => {
    const parsed = parseMetricInput(value, metric);
    
    if (!parsed.success) {
      if (parsed.needsBoth) {
        addMessage('bot', `I see you entered ${parsed.partialValue} for blood pressure. Please enter both systolic and diastolic values like "120/80" or "120 over 80".`, {
          type: 'error'
        });
        return;
      } else {
        addMessage('bot', parsed.error, {
          type: 'error'
        });
        return;
      }
    }

    // Validate the parsed value
    const validation = validateMetricValue(metric, parsed.value, parsed.unit);
    
    if (!validation.valid) {
      addMessage('bot', validation.error, {
        type: 'error'
      });
      return;
    }

    const metricEntry = {
      id: Date.now().toString(),
      medicationId: selectedMedication.id,
      medicationName: selectedMedication.medication_name || selectedMedication.name,
      metric: metric,
      value: validation.value,
      unit: validation.unit,
      timestamp: new Date().toISOString(),
      date: new Date().toISOString().split('T')[0]
    };

    // Update logged metrics
    const updatedLoggedMetrics = [...loggedMetrics, metricEntry];
    setLoggedMetrics(updatedLoggedMetrics);
    
    addMessage('bot', `Perfect! ${metric}: ${validation.value}${validation.unit} logged successfully.`, {
      type: 'success'
    });

    // Check if there are more metrics to log
    const remainingMetrics = selectedMetrics.filter(m => 
      !updatedLoggedMetrics.some(logged => logged.metric === m)
    );

    if (remainingMetrics.length > 0) {
      const nextMetric = remainingMetrics[0];
      setCurrentMetric(nextMetric);
      addMessage('bot', `Now let's log ${nextMetric}. What was your ${nextMetric.toLowerCase()} today?`, {
        type: 'info'
      });
    } else {
      // All metrics logged
      addMessage('bot', "Excellent! You've logged all the selected metrics. Would you like to log metrics for another medication or finish here?", {
        type: 'success',
        actions: [
          { text: 'Another medication', action: 'log_another_medication' },
          { text: 'Finish logging', action: 'finish_logging' }
        ]
      });
      setConversationState('completion');
    }
  };

  // Save metrics to database
  const saveMetrics = async () => {
    try {
      for (const metric of loggedMetrics) {
        await api.post('meds/user/metrics', metric);
      }
      
      addMessage('bot', `✅ All metrics saved successfully! You've logged ${loggedMetrics.length} metrics for ${selectedMedication.medication_name || selectedMedication.name}.`, {
        type: 'success_card',
        metrics: loggedMetrics
      });
      
      setTimeout(() => {
        onSuccess?.(loggedMetrics);
        onClose();
      }, 3000);
      
    } catch (error) {
      console.error('Error saving metrics:', error);
      addMessage('bot', "I'm sorry, there was an error saving your metrics. Please try again.", {
        type: 'error'
      });
    }
  };

  // Handle action clicks
  const handleActionClick = (action, data) => {
    switch (action) {
      case 'add_medication':
        onClose();
        // This would trigger the medication addition flow
        break;
        
      case 'retry_medication_selection':
        handleMedicationSelection();
        break;
        
      case 'log_anyway':
        // Show all medications for logging even if not due today
        addMessage('bot', "No problem! Here are all your medications. Which one would you like to log metrics for?", {
          type: 'success',
          actions: userMedications.map(med => ({
            text: med.medication_name || med.name || med.generic_name,
            action: 'select_medication',
            data: med
          }))
        });
        setConversationState('medication_selection');
        break;
        
      case 'select_medication':
        setSelectedMedication(data);
        
        // Get suggested metrics for this medication
        const suggestedMetrics = data.selected_metrics || data.monitoringMetrics || [];
        const availableSuggestedMetrics = suggestedMetrics.filter(metric => 
          allMetrics.includes(metric)
        );

        if (availableSuggestedMetrics.length > 0) {
          setSelectedMetrics(availableSuggestedMetrics);
          setCurrentMetric(availableSuggestedMetrics[0]);
          addMessage('bot', `Perfect! For ${data.medication_name || data.name}, I'll track these metrics: ${availableSuggestedMetrics.join(', ')}. Let's start with ${availableSuggestedMetrics[0]}. What was your ${availableSuggestedMetrics[0].toLowerCase()} today?`, {
            type: 'success'
          });
          setConversationState('metric_value_input');
        } else {
          addMessage('bot', `Great! Now let's choose which metrics you'd like to track for ${data.medication_name || data.name}.`, {
            type: 'info',
            actions: [
              { text: 'Select metrics', action: 'show_metric_selection' }
            ]
          });
          setConversationState('metric_selection');
        }
        break;
        
      case 'show_metric_selection':
        addMessage('bot', "Select the metrics you'd like to track:", {
          type: 'metric_selection',
          metrics: allMetrics.slice(0, 20)
        });
        setConversationState('metric_selection');
        break;
        
      case 'select_metrics':
        setSelectedMetrics(data);
        setCurrentMetric(data[0]);
        addMessage('bot', `Perfect! Let's start with ${data[0]}. What was your ${data[0].toLowerCase()} today?`, {
          type: 'info'
        });
        setConversationState('metric_value_input');
        break;
        
      case 'log_another_medication':
        setSelectedMedication(null);
        setSelectedMetrics([]);
        setCurrentMetric(null);
        setLoggedMetrics([]);
        handleMedicationSelection();
        break;
        
      case 'finish_logging':
        saveMetrics();
        break;
        
      default:
        console.log('Unknown action:', action);
    }
  };

  // Handle send message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    addMessage('user', userMessage);

    switch (conversationState) {
      case 'metric_value_input':
        if (currentMetric) {
          handleMetricValue(currentMetric, userMessage);
        }
        break;
      default:
        addMessage('bot', "I'm not sure what you mean. Let's start over.", {
          type: 'error'
        });
        setConversationState('medication_selection');
        handleMedicationSelection();
    }
  };

  // Handle key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Render message content
  const renderMessageContent = (message) => {
    if (message.type === 'metric_selection') {
      return (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
            {message.metrics.map((metric, index) => (
              <Button
                key={index}
                onClick={() => {
                  const newSelection = selectedMetrics.includes(metric)
                    ? selectedMetrics.filter(m => m !== metric)
                    : [...selectedMetrics, metric];
                  setSelectedMetrics(newSelection);
                }}
                className={cn(
                  "justify-start text-left rounded-xl px-3 py-2 text-sm font-medium transition-all duration-200",
                  selectedMetrics.includes(metric)
                    ? "bg-white text-black border border-gray-300"
                    : "bg-gray-800 text-white hover:bg-gray-700 border border-gray-700"
                )}
                size="sm"
              >
                {selectedMetrics.includes(metric) && <CheckCircle className="w-4 h-4 mr-2" />}
                {metric}
              </Button>
            ))}
          </div>
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-400">
              {selectedMetrics.length} selected
            </div>
            <Button
              onClick={() => handleActionClick('select_metrics', selectedMetrics)}
              disabled={selectedMetrics.length === 0}
              className="px-4 py-2 bg-white hover:bg-gray-200 text-black rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              size="sm"
            >
              Continue
            </Button>
          </div>
        </div>
      );
    }
    
    if (message.type === 'success_card') {
      return (
        <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <span className="font-semibold text-green-500">Metrics Logged Successfully!</span>
          </div>
          <div className="space-y-1 text-sm">
            <div><strong>Medication:</strong> {selectedMedication?.medication_name || selectedMedication?.name}</div>
            <div><strong>Metrics Logged:</strong> {message.metrics.length}</div>
            <div className="mt-2">
              {message.metrics.map((metric, index) => (
                <div key={index} className="text-xs">
                  • {metric.metric}: {metric.value}{metric.unit}
                </div>
              ))}
            </div>
          </div>
        </div>
      );
    }
    
    return (
      <div className="prose prose-sm max-w-none">
        {message.content.split('**').map((part, index) => 
          index % 2 === 1 ? <strong key={index}>{part}</strong> : part
        )}
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className="relative w-full max-w-2xl max-h-[90vh] bg-black border border-gray-800 rounded-3xl shadow-2xl overflow-hidden flex flex-col"
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-gray-900 to-black border-b border-gray-800 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center">
                <Activity className="w-5 h-5 text-black" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-white">Log Health Metrics</h2>
                <p className="text-sm text-gray-400">Track your health data</p>
              </div>
            </div>
            <Button
              onClick={onClose}
              variant="ghost"
              size="sm"
              className="text-gray-400 hover:bg-gray-800 hover:text-white"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6 bg-black">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={cn(
                  "flex",
                  message.type === 'user' ? "justify-end" : "justify-start"
                )}
              >
                <div className={cn(
                  "flex items-start space-x-4 max-w-[85%]",
                  message.type === 'user' ? "flex-row-reverse space-x-reverse" : ""
                )}>
                  {/* Avatar */}
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1",
                    message.type === 'user' 
                      ? "bg-white text-black" 
                      : "bg-gray-800 text-white"
                  )}>
                    {message.type === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>
                  
                  {/* Message Content */}
                  <div className={cn(
                    "flex-1 min-w-0",
                    message.type === 'user' ? "text-right" : "text-left"
                  )}>
                    <div className={cn(
                      "inline-block p-4 rounded-2xl max-w-full text-[15px] leading-relaxed",
                      message.type === 'user'
                        ? "bg-white text-black rounded-br-lg"
                        : "bg-gray-900 text-white border border-gray-800 rounded-bl-lg"
                    )}>
                      {renderMessageContent(message)}
                      
                      {/* Actions */}
                      {message.actions && (
                        <div className="flex flex-wrap gap-2 mt-4">
                          {message.actions.map((action, index) => (
                            <Button
                              key={index}
                              onClick={() => handleActionClick(action.action, action.data)}
                              size="sm"
                              variant={action.action === 'retry_medication_selection' ? "outline" : "default"}
                              className={cn(
                                "text-xs font-medium transition-all",
                                message.type === 'user' 
                                  ? "bg-primary-foreground/20 text-primary-foreground hover:bg-primary-foreground/30" 
                                  : "hover:bg-primary/10"
                              )}
                            >
                              {action.text}
                            </Button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {/* Typing Indicator */}
          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="flex items-start space-x-4 max-w-[85%]">
                <div className="w-8 h-8 rounded-full bg-gray-800 text-white flex items-center justify-center flex-shrink-0 mt-1">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="inline-block p-4 rounded-2xl rounded-bl-lg bg-gray-900 text-white border border-gray-800">
                    <div className="flex items-center space-x-3">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                      <span className="text-sm text-gray-400">Thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-800 p-6 bg-black">
          <div className="flex items-center space-x-3">
            <Input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your response..."
              disabled={isLoading}
              className="flex-1 bg-gray-900 border border-gray-800 text-white placeholder-gray-400 focus:border-white focus:ring-1 focus:ring-white focus:outline-none rounded-2xl px-4 py-3"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="px-6 py-3 bg-white hover:bg-gray-200 text-black rounded-2xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              size="sm"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default EnhancedMetricsLoggingChat;
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send, Bot, User, CheckCircle, AlertCircle, Loader2, Activity, TrendingUp } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import api from '../api';

const MetricsLoggingChat = ({ isOpen, onClose, onSuccess }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationState, setConversationState] = useState('medication_selection');
  const [selectedMedication, setSelectedMedication] = useState(null);
  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [currentMetric, setCurrentMetric] = useState(null);
  const [loggedMetrics, setLoggedMetrics] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Available metrics for logging
  const availableMetrics = [
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
          content: "Hi! I'll help you log your health metrics for today. Let me first check your medications to see what metrics we should track.",
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

  // Validate metric input
  const validateMetricInput = (metric, value) => {
    const validationRules = {
      'Blood Pressure': { min: 50, max: 250, unit: 'mmHg' },
      'Blood Pressure (Systolic)': { min: 50, max: 250, unit: 'mmHg' },
      'Blood Pressure (Diastolic)': { min: 30, max: 150, unit: 'mmHg' },
      'Heart Rate': { min: 30, max: 200, unit: 'bpm' },
      'Blood Glucose': { min: 50, max: 500, unit: 'mg/dL' },
      'Blood Sugar': { min: 50, max: 500, unit: 'mg/dL' },
      'Weight': { min: 20, max: 300, unit: 'kg' },
      'Temperature': { min: 95, max: 110, unit: '°F' },
      'Pain Level': { min: 0, max: 10, unit: '/10' },
      'Sleep Quality': { min: 0, max: 10, unit: '/10' },
      'Mood': { min: 0, max: 10, unit: '/10' },
      'Energy Level': { min: 0, max: 10, unit: '/10' },
      'BMI': { min: 10, max: 50, unit: 'kg/m²' },
      'Cholesterol': { min: 100, max: 400, unit: 'mg/dL' },
      'Blood Oxygen': { min: 70, max: 100, unit: '%' },
      'Steps Count': { min: 0, max: 50000, unit: 'steps' },
      'Calories Burned': { min: 0, max: 10000, unit: 'cal' },
      'Water Intake': { min: 0, max: 20, unit: 'glasses' }
    };

    const rule = validationRules[metric];
    if (!rule) return { valid: true, value: parseFloat(value) };

    const numValue = parseFloat(value);
    if (isNaN(numValue)) {
      return { valid: false, error: `Please enter a valid number for ${metric}` };
    }

    if (numValue < rule.min || numValue > rule.max) {
      return { 
        valid: false, 
        error: `${metric} should be between ${rule.min} and ${rule.max} ${rule.unit}` 
      };
    }

    return { valid: true, value: numValue, unit: rule.unit };
  };

  // Handle medication selection
  const handleMedicationSelection = async () => {
    setIsLoading(true);
    setIsTyping(true);
    
    try {
      const medications = await fetchMedications();
      
      if (medications.length === 0) {
        addMessage('bot', "You don't have any medications added yet. Please add a medication first before logging metrics.", {
          type: 'error',
          actions: [
            { text: 'Add Medication', action: 'add_medication' }
          ]
        });
        setConversationState('no_medications');
      } else {
        addMessage('bot', "Great! Here are your medications. Which one would you like to track metrics for?", {
          type: 'success',
          actions: medications.map(med => ({
            text: med.medication_name || med.name || med.generic_name,
            action: 'select_medication',
            data: med
          }))
        });
        setConversationState('medication_selection');
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

  // Handle metric selection
  const handleMetricSelection = (medication) => {
    setSelectedMedication(medication);
    
    // Get suggested metrics for this medication
    const suggestedMetrics = medication.selected_metrics || medication.monitoringMetrics || [];
    const availableSuggestedMetrics = suggestedMetrics.filter(metric => 
      availableMetrics.includes(metric)
    );

    if (availableSuggestedMetrics.length > 0) {
      addMessage('bot', `Perfect! For ${medication.medication_name || medication.name}, I suggest tracking these metrics: ${availableSuggestedMetrics.join(', ')}. Would you like to track these or choose different ones?`, {
        type: 'success',
        actions: [
          { text: 'Use suggested', action: 'use_suggested_metrics', data: availableSuggestedMetrics },
          { text: 'Choose different', action: 'choose_different_metrics' }
        ]
      });
    } else {
      addMessage('bot', `Great! Now let's choose which metrics you'd like to track for ${medication.medication_name || medication.name}.`, {
        type: 'info',
        actions: [
          { text: 'Select metrics', action: 'show_metric_selection' }
        ]
      });
    }
    
    setConversationState('metric_selection');
  };

  // Handle metric value input
  const handleMetricValue = (metric, value) => {
    const validation = validateMetricInput(metric, value);
    
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
      unit: validation.unit || '',
      timestamp: new Date().toISOString(),
      date: new Date().toISOString().split('T')[0]
    };

    setLoggedMetrics(prev => [...prev, metricEntry]);
    
    addMessage('bot', `Perfect! ${metric}: ${validation.value}${validation.unit || ''} logged successfully.`, {
      type: 'success'
    });

    // Check if there are more metrics to log
    const remainingMetrics = selectedMetrics.filter(m => 
      !loggedMetrics.some(logged => logged.metric === m)
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
        
      case 'select_medication':
        handleMetricSelection(data);
        break;
        
      case 'use_suggested_metrics':
        setSelectedMetrics(data);
        setCurrentMetric(data[0]);
        addMessage('bot', `Great! Let's start with ${data[0]}. What was your ${data[0].toLowerCase()} today?`, {
          type: 'info'
        });
        setConversationState('metric_value_input');
        break;
        
      case 'choose_different_metrics':
        addMessage('bot', "Which metrics would you like to track?", {
          type: 'metric_selection',
          metrics: availableMetrics.slice(0, 20) // Show first 20 metrics
        });
        setConversationState('metric_selection');
        break;
        
      case 'show_metric_selection':
        addMessage('bot', "Select the metrics you'd like to track:", {
          type: 'metric_selection',
          metrics: availableMetrics.slice(0, 20)
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
      case 'medication_selection':
        // This should be handled by action clicks
        break;
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
                variant={selectedMetrics.includes(metric) ? "default" : "outline"}
                className="justify-start text-left"
                size="sm"
              >
                {selectedMetrics.includes(metric) && <CheckCircle className="w-4 h-4 mr-2" />}
                {metric}
              </Button>
            ))}
          </div>
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              {selectedMetrics.length} selected
            </div>
            <Button
              onClick={() => handleActionClick('select_metrics', selectedMetrics)}
              disabled={selectedMetrics.length === 0}
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
        className="relative w-full max-w-2xl max-h-[90vh] bg-card border border-border rounded-2xl shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-primary to-primary/90 text-primary-foreground p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-full bg-primary-foreground/20 flex items-center justify-center">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-xl font-semibold">Log Health Metrics</h2>
                <p className="text-sm text-primary-foreground/80">Track your health data</p>
              </div>
            </div>
            <Button
              onClick={onClose}
              variant="ghost"
              size="sm"
              className="text-primary-foreground hover:bg-primary-foreground/20"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 max-h-96">
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
                  "flex items-start space-x-3 max-w-[80%]",
                  message.type === 'user' ? "flex-row-reverse space-x-reverse" : ""
                )}>
                  {/* Avatar */}
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
                    message.type === 'user' 
                      ? "bg-primary text-primary-foreground" 
                      : "bg-secondary text-secondary-foreground"
                  )}>
                    {message.type === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>
                  
                  {/* Message Content */}
                  <Card className={cn(
                    "max-w-full",
                    message.type === 'user' 
                      ? "bg-primary text-primary-foreground" 
                      : "bg-card text-card-foreground border-border"
                  )}>
                    <CardContent className="p-4">
                      {renderMessageContent(message)}
                      
                      {/* Actions */}
                      {message.actions && (
                        <div className="flex flex-wrap gap-2 mt-3">
                          {message.actions.map((action, index) => (
                            <Button
                              key={index}
                              onClick={() => handleActionClick(action.action, action.data)}
                              size="sm"
                              variant={action.action === 'retry_medication_selection' ? "outline" : "default"}
                              className="text-xs"
                            >
                              {action.text}
                            </Button>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {/* Typing Indicator */}
          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start space-x-3"
            >
              <div className="w-8 h-8 rounded-full bg-secondary text-secondary-foreground flex items-center justify-center">
                <Bot className="w-4 h-4" />
              </div>
              <Card className="bg-card text-card-foreground border-border">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-border p-6 bg-muted/30">
          <div className="flex items-center space-x-3">
            <Input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your response..."
              disabled={isLoading}
              className="flex-1 bg-background text-foreground border-border focus:ring-2 focus:ring-primary/20 focus:border-primary"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              size="sm"
              className="px-4"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default MetricsLoggingChat;
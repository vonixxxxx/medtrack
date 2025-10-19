import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send, Bot, User, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import api from '../api';

const ProductionMedicationChat = ({ isOpen, onClose, onSuccess }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationState, setConversationState] = useState('medication_name');
  const [medicationData, setMedicationData] = useState({});
  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [availableMetrics, setAvailableMetrics] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Predefined health metrics
  const healthMetrics = [
    'Blood Pressure', 'Heart Rate', 'Blood Glucose', 'Weight', 'Temperature',
    'Pain Level', 'Sleep Quality', 'Mood', 'Energy Level', 'Side Effects',
    'Blood Sugar', 'Cholesterol', 'Blood Oxygen', 'General Health'
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
          content: "Hi! I'm your MedTrack Assistant. What medication would you like to add?",
          timestamp: new Date()
        }
      ]);
      setConversationState('medication_name');
      setMedicationData({});
      setSelectedMetrics([]);
    }
  }, [isOpen]);

  const addMessage = (type, content, options = {}) => {
    const newMessage = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      content,
      timestamp: new Date(),
      ...options
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const showTypingIndicator = () => {
    setIsTyping(true);
    setTimeout(() => setIsTyping(false), 1500);
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    addMessage('user', userMessage);

    setIsLoading(true);
    showTypingIndicator();

    try {
      await processUserInput(userMessage);
    } catch (error) {
      console.error('Error processing input:', error);
      addMessage('bot', "I'm sorry, I encountered an error. Please try again.", { 
        type: 'error' 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const processUserInput = async (input) => {
    switch (conversationState) {
      case 'medication_name':
        await handleMedicationName(input);
        break;
      case 'dosage_strength':
        await handleDosageStrength(input);
        break;
      case 'frequency':
        await handleFrequency(input);
        break;
      case 'metrics_selection':
        await handleMetricsSelection(input);
        break;
      case 'confirmation':
        await handleConfirmation(input);
        break;
      default:
        break;
    }
  };

  const handleMedicationName = async (input) => {
    try {
      const response = await api.post('medications/validateMedication', {
        medication_name: input
      });

      if (response.data.success) {
        const validationData = response.data.data;
        setMedicationData(prev => ({ ...prev, validation: validationData }));
        
        if (validationData.confidence >= 0.8) {
          // High confidence - confirm directly
          addMessage('bot', `Got it — you meant **${validationData.generic_name}**, right?`, {
            type: 'success',
            actions: [
              { text: 'Yes', action: 'confirm_medication', data: validationData },
              { text: 'No, try again', action: 'retry_medication' }
            ]
          });
          setConversationState('medication_confirmation');
        } else if (validationData.confidence >= 0.5) {
          // Medium confidence - show options
          const alternatives = validationData.alternatives || [];
          const options = [validationData.generic_name, ...alternatives].slice(0, 3);
          addMessage('bot', `I found a few options. Did you mean ${options.join(', ')}?`, {
            type: 'options',
            options: options.map(option => ({
              text: option,
              action: 'select_medication',
              data: { ...validationData, generic_name: option }
            }))
          });
          setConversationState('medication_selection');
        } else {
          // Low confidence - ask for clarification
          addMessage('bot', "I couldn't find that medication. Could you try retyping the name or check the spelling?", {
            type: 'error'
          });
        }
      } else {
        addMessage('bot', "I'm having trouble searching for medications right now. Please try again.", {
          type: 'error'
        });
      }
    } catch (error) {
      console.error('Medication validation error:', error);
      addMessage('bot', "I'm having trouble searching for medications right now. Please try again.", {
        type: 'error'
      });
    }
  };

  const handleDosageStrength = (input) => {
    const dosageMatch = input.match(/(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)/i);
    
    if (dosageMatch) {
      const strength = dosageMatch[1];
      const unit = dosageMatch[2];
      
      setMedicationData(prev => ({
        ...prev,
        strength,
        unit
      }));
      
      addMessage('bot', `Perfect! ${strength}${unit}. How often do you take it?`, {
        type: 'success'
      });
      setConversationState('frequency');
    } else {
      addMessage('bot', "I need the dosage in a format like '500mg' or '10ml'. Could you try again?", {
        type: 'error'
      });
    }
  };

  const handleFrequency = (input) => {
    const frequency = parseFrequency(input);
    
    if (frequency) {
      setMedicationData(prev => ({
        ...prev,
        frequency
      }));
      
      // Show metrics selection
      const suggestedMetrics = medicationData.validation?.suggested_metrics || healthMetrics.slice(0, 4);
      setAvailableMetrics(suggestedMetrics);
      
      addMessage('bot', `Got it! ${frequency}. Which metrics would you like to track for this medication?`, {
        type: 'success',
        metrics: suggestedMetrics
      });
      setConversationState('metrics_selection');
    } else {
      addMessage('bot', "I need to know how often you take it. Try something like 'twice daily', 'once a day', or 'every 8 hours'.", {
        type: 'error'
      });
    }
  };

  const handleMetricsSelection = (input) => {
    if (input.toLowerCase().includes('done') || input.toLowerCase().includes('finish')) {
      if (selectedMetrics.length === 0) {
        addMessage('bot', "Please select at least one metric to track.", {
          type: 'error'
        });
        return;
      }
      
      addMessage('bot', `Perfect! I'll track ${selectedMetrics.join(', ')} for ${medicationData.validation?.generic_name} (${medicationData.strength}${medicationData.unit}, ${medicationData.frequency}). Ready to add it?`, {
        type: 'success',
        actions: [
          { text: 'Yes, add it', action: 'confirm_all' },
          { text: 'Make changes', action: 'edit_details' }
        ]
      });
      setConversationState('confirmation');
    } else {
      // Parse metrics from input
      const inputMetrics = parseMetricsFromInput(input);
      if (inputMetrics.length > 0) {
        setSelectedMetrics(prev => [...new Set([...prev, ...inputMetrics])]);
        addMessage('bot', `Added: ${inputMetrics.join(', ')}. Type 'done' when finished, or add more metrics.`, {
          type: 'info'
        });
      } else {
        addMessage('bot', "I didn't recognize those metrics. Available options: " + availableMetrics.join(', '), {
          type: 'info'
        });
      }
    }
  };

  const handleConfirmation = async (input) => {
    if (input.toLowerCase().includes('yes') || input.toLowerCase().includes('add')) {
      await handleFinalSubmission();
    } else {
      addMessage('bot', "No problem! Let me know if you'd like to make any changes.", {
        type: 'info'
      });
    }
  };

  const handleFinalSubmission = async () => {
    try {
      const medicationEntry = {
        medication_name: medicationData.validation?.generic_name || medicationData.medicationName,
        strength: medicationData.strength,
        unit: medicationData.unit,
        frequency: medicationData.frequency,
        selectedMetrics: selectedMetrics,
        startDate: new Date().toISOString().split('T')[0],
        aiValidated: true,
        confidence: medicationData.validation?.confidence || 0.9,
        drug_class: medicationData.validation?.drug_class,
        indications: medicationData.validation?.indications || []
      };

      await api.post('meds/user', medicationEntry);
      
      // Show success card
      addMessage('bot', "✅ Medication added successfully!", {
        type: 'success_card',
        medication: medicationEntry
      });
      
      setTimeout(() => {
        onSuccess?.(medicationEntry);
        onClose();
      }, 3000);
      
    } catch (error) {
      console.error('Error adding medication:', error);
      addMessage('bot', "I'm sorry, there was an error adding your medication. Please try again.", {
        type: 'error'
      });
    }
  };

  const handleActionClick = (action, data) => {
    switch (action) {
      case 'confirm_medication':
        // Use data parameter if available, otherwise fall back to stored validation data
        const validationToUse = data || medicationData.validation;
        if (validationToUse?.generic_name) {
          setMedicationData(prev => ({ ...prev, medicationName: validationToUse.generic_name }));
          addMessage('bot', `Great! What's the dosage and strength?`, {
            type: 'success'
          });
          setConversationState('dosage_strength');
        } else {
          addMessage('bot', "I'm sorry, I don't have the medication details. Let's try again.", {
            type: 'error'
          });
          setConversationState('medication_name');
        }
        break;
      case 'retry_medication':
        addMessage('bot', "No problem! What medication would you like to add?", {
          type: 'info'
        });
        setConversationState('medication_name');
        break;
      case 'select_medication':
        if (data?.generic_name) {
          setMedicationData(prev => ({ ...prev, validation: data, medicationName: data.generic_name }));
          addMessage('bot', `Perfect! ${data.generic_name}. What's the dosage and strength?`, {
            type: 'success'
          });
          setConversationState('dosage_strength');
        } else {
          addMessage('bot', "I'm sorry, I don't have the medication details. Let's try again.", {
            type: 'error'
          });
          setConversationState('medication_name');
        }
        break;
      case 'confirm_all':
        handleFinalSubmission();
        break;
      case 'edit_details':
        addMessage('bot', "What would you like to change?", {
          type: 'info'
        });
        setConversationState('medication_name');
        break;
      default:
        break;
    }
  };

  const parseFrequency = (input) => {
    const text = input.toLowerCase();
    
    if (text.includes('twice') || text.includes('two times')) {
      return 'Twice daily';
    } else if (text.includes('once') || text.includes('one time')) {
      return 'Once daily';
    } else if (text.includes('three times') || text.includes('thrice')) {
      return 'Three times daily';
    } else if (text.includes('weekly') || text.includes('once a week')) {
      return 'Weekly';
    } else if (text.includes('monthly') || text.includes('once a month')) {
      return 'Monthly';
    } else if (text.includes('as needed') || text.includes('prn')) {
      return 'As needed';
    } else if (text.includes('every 8 hours')) {
      return 'Every 8 hours';
    } else if (text.includes('every 12 hours')) {
      return 'Every 12 hours';
    } else if (text.includes('daily')) {
      return 'Once daily';
    }
    
    return null;
  };

  const parseMetricsFromInput = (input) => {
    const text = input.toLowerCase();
    const foundMetrics = healthMetrics.filter(metric => 
      text.includes(metric.toLowerCase()) ||
      (metric === 'Blood Pressure' && text.includes('bp')) ||
      (metric === 'Heart Rate' && text.includes('hr')) ||
      (metric === 'Blood Glucose' && (text.includes('glucose') || text.includes('blood sugar')))
    );
    
    return foundMetrics;
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="bg-white rounded-3xl shadow-2xl w-full max-w-4xl h-[800px] flex flex-col border border-gray-200"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-teal-600 to-blue-600 text-white p-6 rounded-t-3xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center">
                  <Bot className="w-7 h-7" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">Add New Medication</h2>
                  <p className="text-white/90 text-lg">AI Health Assistant</p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="text-white/80 hover:text-white hover:bg-white/10"
              >
                <X className="w-6 h-6" />
              </Button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 30, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ 
                  type: "spring", 
                  damping: 20, 
                  stiffness: 300,
                  delay: 0.1 
                }}
                className={cn(
                  "flex",
                  message.type === 'user' ? "justify-end" : "justify-start"
                )}
              >
                <div className={cn(
                  "flex items-start space-x-3 max-w-[85%]",
                  message.type === 'user' ? "flex-row-reverse space-x-reverse" : ""
                )}>
                  <div className={cn(
                    "w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg",
                    message.type === 'user' 
                      ? "bg-blue-500 text-white" 
                      : "bg-gradient-to-r from-teal-500 to-blue-500 text-white"
                  )}>
                    {message.type === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                  </div>
                  
                  <Card className={cn(
                    "max-w-full",
                    message.type === 'user'
                      ? "bg-blue-500 text-white border-blue-500"
                      : message.type === 'error'
                      ? "bg-red-50 text-red-800 border-red-200"
                      : message.type === 'success'
                      ? "bg-green-50 text-green-800 border-green-200"
                      : message.type === 'success_card'
                      ? "bg-gradient-to-r from-green-50 to-teal-50 text-green-800 border-green-200"
                      : "bg-gray-50 text-gray-800 border-gray-200"
                  )}>
                    <CardContent className="p-4">
                      <div className="text-lg leading-relaxed whitespace-pre-wrap" dangerouslySetInnerHTML={{ 
                        __html: message.content.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold">$1</strong>')
                      }} />
                      
                      {/* Success Card */}
                      {message.type === 'success_card' && message.medication && (
                        <div className="mt-4 p-4 bg-white rounded-xl border border-green-200">
                          <div className="flex items-center space-x-2 mb-3">
                            <CheckCircle className="w-6 h-6 text-green-600" />
                            <h3 className="text-lg font-semibold text-green-800">Medication Added Successfully!</h3>
                          </div>
                          <div className="space-y-2 text-sm">
                            <div><strong>Name:</strong> {message.medication.medication_name}</div>
                            <div><strong>Dosage:</strong> {message.medication.strength}{message.medication.unit}</div>
                            <div><strong>Frequency:</strong> {message.medication.frequency}</div>
                            <div><strong>Metrics:</strong> {message.medication.selectedMetrics.join(', ')}</div>
                          </div>
                        </div>
                      )}
                      
                      {/* Metrics Selection */}
                      {message.metrics && (
                        <div className="mt-4">
                          <p className="text-sm font-medium mb-2">Select metrics to track:</p>
                          <div className="flex flex-wrap gap-2">
                            {message.metrics.map((metric, index) => (
                              <Badge
                                key={index}
                                variant={selectedMetrics.includes(metric) ? "default" : "outline"}
                                className="cursor-pointer hover:bg-teal-100"
                                onClick={() => {
                                  if (selectedMetrics.includes(metric)) {
                                    setSelectedMetrics(prev => prev.filter(m => m !== metric));
                                  } else {
                                    setSelectedMetrics(prev => [...prev, metric]);
                                  }
                                }}
                              >
                                {metric}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Action buttons */}
                      {message.actions && (
                        <div className="mt-4 flex flex-wrap gap-3">
                          {message.actions.map((action, index) => (
                            <motion.div
                              key={index}
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                            >
                              <Button
                                onClick={() => handleActionClick(action.action, action.data)}
                                className="bg-gradient-to-r from-teal-500 to-blue-500 hover:from-teal-600 hover:to-blue-600 text-white"
                              >
                                {action.text}
                              </Button>
                            </motion.div>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </motion.div>
            ))}
            
            {/* Typing Indicator */}
            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 30, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                className="flex justify-start"
              >
                <div className="flex items-start space-x-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-r from-teal-500 to-blue-500 text-white flex items-center justify-center shadow-lg">
                    <Bot className="w-5 h-5" />
                  </div>
                  <Card className="bg-gray-50 text-gray-800 border-gray-200">
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-teal-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                        <span className="text-sm text-gray-600">AI is thinking...</span>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 p-6 bg-gray-50 rounded-b-3xl">
            <div className="flex items-center space-x-4">
              <Input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your response..."
                disabled={isLoading}
                className="flex-1 text-lg h-12 border-2 border-gray-300 rounded-xl focus:ring-4 focus:ring-teal-500/20 focus:border-teal-500"
              />
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim() || isLoading}
                  className="h-12 px-6 bg-gradient-to-r from-teal-500 to-blue-500 hover:from-teal-600 hover:to-blue-600 text-white rounded-xl"
                >
                  <Send className="w-5 h-5" />
                </Button>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default ProductionMedicationChat;
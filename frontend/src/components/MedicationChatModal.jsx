import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send, CheckCircle, AlertCircle, Loader2, Bot, User } from 'lucide-react';
import { cn } from '../lib/utils';
import api from '../api';

const MedicationChatModal = ({ isOpen, onClose, onSuccess }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationState, setConversationState] = useState('medication_name');
  const [medicationData, setMedicationData] = useState({});
  const [suggestedMetrics, setSuggestedMetrics] = useState([]);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

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
          content: "Hi! I'm here to help you add a new medication. What medication would you like to add?",
          timestamp: new Date()
        }
      ]);
      setConversationState('medication_name');
      setMedicationData({});
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

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    addMessage('user', userMessage);

    setIsLoading(true);

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
      case 'metrics':
        await handleMetrics(input);
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
      const response = await api.post('ai/search-med', {
        query: input,
        limit: 5,
        min_confidence: 0.3
      });

      const results = response.data.results || [];
      
      if (results.length === 0) {
        addMessage('bot', "I couldn't find that medication. Could you try retyping the name or check the spelling?", {
          type: 'error'
        });
        return;
      }

      if (results.length === 1) {
        const medication = results[0];
        setMedicationData(prev => ({ ...prev, medication }));
        addMessage('bot', `Got it — you meant **${medication.name}**, right?`, {
          type: 'success',
          actions: [
            { text: 'Yes', action: 'confirm_medication' },
            { text: 'No, try again', action: 'retry_medication' }
          ]
        });
        setConversationState('medication_confirmation');
      } else {
        const medicationOptions = results.slice(0, 3).map(med => med.name).join(', ');
        addMessage('bot', `I found a few options. Did you mean ${medicationOptions}?`, {
          type: 'options',
          options: results.slice(0, 3).map(med => ({
            text: med.name,
            action: 'select_medication',
            data: med
          }))
        });
        setConversationState('medication_selection');
      }
    } catch (error) {
      console.error('Medication search error:', error);
      addMessage('bot', "I'm having trouble searching for medications right now. Please try again.", {
        type: 'error'
      });
    }
  };

  const handleDosageStrength = (input) => {
    // Parse dosage input
    const dosageMatch = input.match(/(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units)/i);
    
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
    // Parse frequency input
    const frequency = parseFrequency(input);
    
    if (frequency) {
      setMedicationData(prev => ({
        ...prev,
        frequency
      }));
      
      // Suggest relevant metrics based on medication type
      const metrics = suggestMetrics(medicationData.medication);
      setSuggestedMetrics(metrics);
      
      addMessage('bot', `Got it! ${frequency}. We'll track ${metrics.join(', ')} for this medication. Does that sound good?`, {
        type: 'success',
        actions: [
          { text: 'Yes, add it', action: 'confirm_all' },
          { text: 'Change metrics', action: 'change_metrics' }
        ]
      });
      setConversationState('confirmation');
    } else {
      addMessage('bot', "I need to know how often you take it. Try something like 'twice daily', 'once a day', or 'every 8 hours'.", {
        type: 'error'
      });
    }
  };

  const handleMetrics = (input) => {
    // Handle metric selection
    if (input.toLowerCase().includes('yes') || input.toLowerCase().includes('good')) {
      addMessage('bot', "Perfect! Adding your medication now...", {
        type: 'success'
      });
      setConversationState('confirmation');
      handleFinalSubmission();
    } else {
      // Parse custom metrics from user input
      const customMetrics = parseCustomMetrics(input);
      if (customMetrics.length > 0) {
        setSuggestedMetrics(customMetrics);
        addMessage('bot', `Great! I'll track ${customMetrics.join(', ')} for this medication. Ready to add it?`, {
          type: 'success',
          actions: [
            { text: 'Yes, add it', action: 'confirm_all' },
            { text: 'Change metrics', action: 'change_metrics' }
          ]
        });
        setConversationState('confirmation');
      } else {
        addMessage('bot', "I didn't catch that. What specific metrics would you like to track? (e.g., heart rate, blood pressure, pain level)", {
          type: 'info'
        });
      }
    }
  };

  const parseCustomMetrics = (input) => {
    const text = input.toLowerCase();
    const allMetrics = [
      'heart rate', 'blood pressure', 'temperature', 'pain level', 'sleep quality',
      'mood', 'energy level', 'weight', 'blood glucose', 'side effects',
      'general health', 'blood sugar', 'glucose', 'bp', 'hr', 'temp'
    ];
    
    const foundMetrics = allMetrics.filter(metric => 
      text.includes(metric) || 
      (metric === 'blood pressure' && text.includes('bp')) ||
      (metric === 'heart rate' && text.includes('hr')) ||
      (metric === 'temperature' && text.includes('temp')) ||
      (metric === 'blood glucose' && (text.includes('glucose') || text.includes('blood sugar')))
    );
    
    // If no specific metrics found, return general health
    return foundMetrics.length > 0 ? foundMetrics : ['General Health'];
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
        medication_name: medicationData.medication.name,
        strength: medicationData.strength,
        unit: medicationData.unit,
        frequency: medicationData.frequency,
        selectedMetrics: suggestedMetrics,
        startDate: new Date().toISOString().split('T')[0],
        aiValidated: true,
        confidence: 0.9
      };

      await api.post('meds/user', medicationEntry);
      
      addMessage('bot', "✅ Perfect! Your medication has been added successfully. You can now see it in your dashboard.", {
        type: 'success'
      });
      
      setTimeout(() => {
        onSuccess?.(medicationEntry);
        onClose();
      }, 2000);
      
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
        setMedicationData(prev => ({ ...prev, medication: medicationData.medication }));
        addMessage('bot', `Great! What's the dosage and strength?`, {
          type: 'success'
        });
        setConversationState('dosage_strength');
        break;
      case 'retry_medication':
        addMessage('bot', "No problem! What medication would you like to add?", {
          type: 'info'
        });
        setConversationState('medication_name');
        break;
      case 'select_medication':
        setMedicationData(prev => ({ ...prev, medication: data }));
        addMessage('bot', `Perfect! ${data.name}. What's the dosage and strength?`, {
          type: 'success'
        });
        setConversationState('dosage_strength');
        break;
      case 'confirm_all':
        handleFinalSubmission();
        break;
      case 'change_metrics':
        addMessage('bot', "What metrics would you like to track for this medication?", {
          type: 'info'
        });
        setConversationState('metrics');
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

  const suggestMetrics = (medication) => {
    if (!medication) return ['General Health'];
    
    const name = medication.name.toLowerCase();
    const suggestions = medication.suggestions?.map(s => s.toLowerCase()) || [];
    
    if (name.includes('diabetes') || name.includes('metformin') || name.includes('insulin') || 
        suggestions.some(s => s.includes('glucose') || s.includes('diabetes'))) {
      return ['Blood Glucose', 'Weight', 'Blood Pressure'];
    } else if (name.includes('heart') || name.includes('cardio') || name.includes('blood pressure') ||
               suggestions.some(s => s.includes('heart') || s.includes('cardio'))) {
      return ['Heart Rate', 'Blood Pressure', 'Weight'];
    } else if (name.includes('pain') || name.includes('ibuprofen') || name.includes('aspirin')) {
      return ['Pain Level', 'Sleep Quality'];
    } else if (name.includes('mental') || name.includes('depression') || name.includes('anxiety')) {
      return ['Mood', 'Sleep Quality', 'Energy Level'];
    } else {
      return ['General Health', 'Side Effects'];
    }
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
        className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="bg-white rounded-3xl shadow-2xl w-full max-w-3xl h-[700px] flex flex-col border border-gray-200"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-t-3xl">
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
              <button
                onClick={onClose}
                className="text-white/80 hover:text-white transition-colors p-3 rounded-xl hover:bg-white/10"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
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
                      : "bg-gradient-to-r from-purple-500 to-pink-500 text-white"
                  )}>
                    {message.type === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                  </div>
                  <div className={cn(
                    "rounded-2xl px-6 py-4 shadow-lg max-w-full",
                    message.type === 'user'
                      ? "bg-blue-500 text-white"
                      : message.type === 'error'
                      ? "bg-red-50 text-red-800 border-2 border-red-200"
                      : message.type === 'success'
                      ? "bg-green-50 text-green-800 border-2 border-green-200"
                      : "bg-gray-50 text-gray-800 border border-gray-200"
                  )}>
                    <div className="text-lg leading-relaxed whitespace-pre-wrap" dangerouslySetInnerHTML={{ 
                      __html: message.content.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold">$1</strong>')
                    }} />
                    
                    {/* Action buttons */}
                    {message.actions && (
                      <div className="mt-4 flex flex-wrap gap-3">
                        {message.actions.map((action, index) => (
                          <motion.button
                            key={index}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => handleActionClick(action.action, action.data)}
                            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white rounded-xl text-lg font-medium shadow-lg transition-all duration-200"
                          >
                            {action.text}
                          </motion.button>
                        ))}
                      </div>
                    )}
                    
                    {/* Option buttons */}
                    {message.options && (
                      <div className="mt-4 flex flex-wrap gap-3">
                        {message.options.map((option, index) => (
                          <motion.button
                            key={index}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => handleActionClick(option.action, option.data)}
                            className="px-6 py-3 bg-gradient-to-r from-green-500 to-teal-500 hover:from-green-600 hover:to-teal-600 text-white rounded-xl text-lg font-medium shadow-lg transition-all duration-200"
                          >
                            {option.text}
                          </motion.button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
            
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 30, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ type: "spring", damping: 20, stiffness: 300 }}
                className="flex justify-start"
              >
                <div className="flex items-start space-x-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 text-white flex items-center justify-center shadow-lg">
                    <Bot className="w-5 h-5" />
                  </div>
                  <div className="bg-gray-50 text-gray-800 rounded-2xl px-6 py-4 shadow-lg border border-gray-200">
                    <div className="flex items-center space-x-3">
                      <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
                      <span className="text-lg font-medium">Thinking...</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 p-6 bg-gray-50 rounded-b-3xl">
            <div className="flex items-center space-x-4">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your response..."
                disabled={isLoading}
                className="flex-1 px-6 py-4 border-2 border-gray-300 rounded-2xl focus:outline-none focus:ring-4 focus:ring-blue-500/20 focus:border-blue-500 disabled:opacity-50 text-lg placeholder-gray-500 bg-white shadow-lg"
              />
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="p-4 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white rounded-2xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg"
              >
                <Send className="w-6 h-6" />
              </motion.button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default MedicationChatModal;
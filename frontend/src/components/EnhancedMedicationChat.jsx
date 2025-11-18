import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send, Bot, User, CheckCircle, AlertCircle, Loader2, Pill, Clock, Activity, Plus, Search, Edit3, Sparkles } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Checkbox } from './ui/checkbox';
import { cn } from '../lib/utils';
import api from '../api';

const EnhancedMedicationChat = ({ isOpen, onClose, onSuccess }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationState, setConversationState] = useState('welcome');
  const [medicationData, setMedicationData] = useState({});
  const [suggestedStrengths, setSuggestedStrengths] = useState([]);
  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [customMetricInput, setCustomMetricInput] = useState('');
  const [showCustomMetricInput, setShowCustomMetricInput] = useState(false);
  const [specialInstructions, setSpecialInstructions] = useState('');
  const [aiGeneratedNotes, setAiGeneratedNotes] = useState('');
  const [isEditingNotes, setIsEditingNotes] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // All available metrics for tracking
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

  // Common frequency options
  const frequencyOptions = [
    { value: 'daily', label: 'Once daily', description: 'Every day at the same time' },
    { value: 'twice_daily', label: 'Twice daily', description: 'Morning and evening' },
    { value: 'three_times_daily', label: 'Three times daily', description: 'Morning, afternoon, evening' },
    { value: 'four_times_daily', label: 'Four times daily', description: 'Every 6 hours' },
    { value: 'every_other_day', label: 'Every other day', description: 'Alternate days' },
    { value: 'weekly', label: 'Once weekly', description: 'Once per week' },
    { value: 'monthly', label: 'Once monthly', description: 'Once per month' },
    { value: 'as_needed', label: 'As needed', description: 'When symptoms occur' }
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
          content: "Hi! I'm your MedTrack Assistant. I'll help you add a new medication with AI-powered validation and personalized recommendations. What medication would you like to add?",
          timestamp: new Date(),
          showTyping: false
        }
      ]);
      setConversationState('medication_input');
      setMedicationData({});
      setSuggestedStrengths([]);
      setSelectedMetrics([]);
      setCustomMetricInput('');
      setShowCustomMetricInput(false);
      setSpecialInstructions('');
      setAiGeneratedNotes('');
      setIsEditingNotes(false);
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

  // Simulate typing delay
  const simulateTyping = (callback, delay = 1000) => {
    setIsTyping(true);
    setTimeout(() => {
      setIsTyping(false);
      callback();
    }, delay);
  };

  // Get common strengths for a medication
  const getCommonStrengths = (medication) => {
    const strengths = medication.typical_strengths || [];
    if (strengths.length > 0) {
      return strengths.slice(0, 6); // Show up to 6 common strengths
    }
    
    // Fallback common strengths based on drug class
    const classStrengths = {
      'Analgesic': ['200mg', '400mg', '600mg', '800mg'],
      'Antibiotic': ['250mg', '500mg', '750mg', '1000mg'],
      'Antihypertensive': ['5mg', '10mg', '20mg', '40mg'],
      'Antidiabetic': ['500mg', '850mg', '1000mg', '2mg'],
      'Stimulant': ['5mg', '10mg', '15mg', '20mg', '25mg', '30mg'],
      'SSRI': ['25mg', '50mg', '100mg', '200mg'],
      'Biguanide': ['500mg', '850mg', '1000mg'],
      'GLP-1 Receptor Agonist': ['0.25mg', '0.5mg', '1mg', '2.4mg', '3mg', '7mg', '14mg']
    };
    
    return classStrengths[medication.drug_class] || ['10mg', '25mg', '50mg', '100mg', '250mg', '500mg'];
  };

  // Parse natural language frequency
  const parseFrequency = (input) => {
    const lowerInput = input.toLowerCase().trim();
    
    // Direct matches
    const directMatches = {
      'daily': 'daily',
      'once daily': 'daily',
      'once a day': 'daily',
      'every day': 'daily',
      'twice daily': 'twice_daily',
      'twice a day': 'twice_daily',
      'two times daily': 'twice_daily',
      'three times daily': 'three_times_daily',
      'three times a day': 'three_times_daily',
      'four times daily': 'four_times_daily',
      'four times a day': 'four_times_daily',
      'weekly': 'weekly',
      'once weekly': 'weekly',
      'once a week': 'weekly',
      'monthly': 'monthly',
      'once monthly': 'monthly',
      'once a month': 'monthly',
      'as needed': 'as_needed',
      'when needed': 'as_needed',
      'prn': 'as_needed'
    };
    
    if (directMatches[lowerInput]) {
      return directMatches[lowerInput];
    }
    
    // Pattern matching
    if (lowerInput.includes('every 2 days') || lowerInput.includes('every other day')) {
      return 'every_other_day';
    }
    
    if (lowerInput.includes('twice') && lowerInput.includes('week')) {
      return 'twice_weekly';
    }
    
    if (lowerInput.includes('every') && lowerInput.includes('day')) {
      return 'daily';
    }
    
    if (lowerInput.includes('every') && lowerInput.includes('week')) {
      return 'weekly';
    }
    
    if (lowerInput.includes('every') && lowerInput.includes('month')) {
      return 'monthly';
    }
    
    // Default to daily if unclear
    return 'daily';
  };

  // Get frequency display text
  const getFrequencyDisplay = (frequency) => {
    const option = frequencyOptions.find(opt => opt.value === frequency);
    return option ? option.label : frequency;
  };

  // Handle medication input with AI validation
  const handleMedicationInput = async (input) => {
    if (isProcessing) return;
    setIsProcessing(true);
    setIsLoading(true);
    addMessage('user', input);
    
    try {
      console.log('Making API call for medication:', input);
      const response = await api.post('medications/validateMedication', {
        medication: input,
        medication_name: input,
        name: input
      }, {
        timeout: 10000 // 10 second timeout for validation (includes RxNorm calls)
      });
      
      console.log('API response:', response.data);
      
      // Check if medication was found using new response format
      if (response.data.found && response.data.data) {
        const match = response.data.data;
        const confidence = match.confidence || 0.5;
        
        // Check confidence threshold
        if (confidence < 0.80) {
          // Low confidence - ask for confirmation
          setMedicationData(match);
          const strengths = getCommonStrengths(match);
          setSuggestedStrengths(strengths);
          
          simulateTyping(() => {
            addMessage('bot', `I'm not certain — did you mean **${match.generic_name}** (${match.drug_class})? Confidence: ${Math.round(confidence * 100)}%`, {
              type: 'medication_confirmation',
              medication: match,
              confidence: confidence,
              actions: [
                { text: 'Yes, that\'s correct', action: 'confirm_medication', data: match },
                { text: 'No, try again', action: 'retry_medication' }
              ]
            });
          });
        } else {
          // High confidence - confirm directly
          setMedicationData(match);
          const strengths = getCommonStrengths(match);
          setSuggestedStrengths(strengths);
          
          simulateTyping(() => {
            const sourceInfo = match.source === 'rxnorm' ? ' (verified via RxNorm)' : 
                              match.source === 'dmd' ? ' (verified via NHS dm+d)' : '';
            const bioInfo = match.bio ? ` • BioGPT confidence: ${Math.round(match.bio.confidence * 100)}%` : '';
            addMessage('bot', `Found **${match.generic_name}** (${match.drug_class})${sourceInfo}${bioInfo}. Is this the medication you're looking for?`, {
              type: 'medication_confirmation',
              medication: match,
              confidence: confidence,
              actions: [
                { text: 'Yes, that\'s correct', action: 'confirm_medication', data: match },
                { text: 'No, try again', action: 'retry_medication' }
              ]
            });
          });
        }
      } else {
        // No match found - show proper error message
        const errorMessage = response.data.error || response.data.message || `No medication found for "${input}"`;
        const suggestions = response.data.suggestions || [];
        
        simulateTyping(() => {
          let messageContent = errorMessage;
          
          if (suggestions.length > 0) {
            messageContent += '\n\nDid you mean:';
            const suggestionButtons = suggestions.slice(0, 5).map(suggestion => ({
              text: typeof suggestion === 'string' ? suggestion : (suggestion.generic_name || suggestion.name || suggestion),
              action: 'select_suggestion',
              data: typeof suggestion === 'string' ? suggestion : suggestion
            }));
            
            addMessage('bot', messageContent, {
              type: 'error',
              suggestions: suggestionButtons,
              actions: [
                { text: 'Try again', action: 'retry_medication' }
              ]
            });
          } else {
            addMessage('bot', messageContent + '\n\nPlease try:\n• Using the generic name (e.g., "acetaminophen" instead of "Tylenol")\n• Checking your spelling\n• Using a different medication name', {
              type: 'error',
              actions: [
                { text: 'Try again', action: 'retry_medication' }
              ]
            });
          }
        });
      }
    } catch (error) {
      console.error('Medication validation error:', error);
      
      // Handle timeout or network errors
      if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
        simulateTyping(() => {
          addMessage('bot', 'The validation service is taking longer than expected. Please try again in a moment, or check your internet connection.', {
            type: 'error',
            actions: [
              { text: 'Try again', action: 'retry_medication' }
            ]
          });
        });
      } else {
        simulateTyping(() => {
          addMessage('bot', 'I\'m having trouble connecting to the medication database. Please try again.', {
            type: 'error',
            actions: [
              { text: 'Try again', action: 'retry_medication' }
            ]
          });
        });
      }
    } finally {
      setIsLoading(false);
      setIsProcessing(false);
    }
  };

  // Old fallback functions removed - all matching now handled by backend service

  // Handle strength selection
  const handleStrengthInput = (strength) => {
    addMessage('user', strength);
    
    // Parse strength and unit
    const match = strength.match(/^(\d+(?:\.\d+)?)\s*(mg|ml|g|mcg|units?|iu|tablets?|capsules?)$/i);
    if (match) {
      const [, value, unit] = match;
      setMedicationData(prev => ({
        ...prev,
        strength: value,
        unit: unit.toLowerCase()
      }));
    } else {
      setMedicationData(prev => ({
        ...prev,
        strength: strength,
        unit: 'mg'
      }));
    }
    
    simulateTyping(() => {
      addMessage('bot', `Great! **${strength}** selected. Now, how often do you take this medication?`, {
        type: 'frequency_selection',
        frequencies: frequencyOptions
      });
    });
  };

  // Handle frequency selection
  const handleFrequencyInput = (frequency) => {
    addMessage('user', frequency);
    
    const parsedFrequency = parseFrequency(frequency);
    const displayText = getFrequencyDisplay(parsedFrequency);
    
    setMedicationData(prev => ({
      ...prev,
      frequency: parsedFrequency,
      frequency_display: displayText
    }));
    
    simulateTyping(() => {
      handleMetricSelection();
    });
  };

  // Handle metric selection
  const handleMetricSelection = () => {
    // Get suggested metrics based on drug class
    const suggestedMetrics = getSuggestedMetrics(medicationData.drug_class);
    setSelectedMetrics(suggestedMetrics);
    
    addMessage('bot', `For **${medicationData.generic_name}** (${medicationData.drug_class}), I suggest tracking these metrics:`, {
      type: 'metric_selection',
      suggestedMetrics,
      allMetrics: allMetrics
    });
  };

  // Get suggested metrics based on drug class
  const getSuggestedMetrics = (drugClass) => {
    const classMetrics = {
      'Analgesic': ['Pain Level', 'Side Effects', 'General Health'],
      'Antibiotic': ['Temperature', 'Side Effects', 'General Health'],
      'Antihypertensive': ['Blood Pressure', 'Heart Rate', 'Weight'],
      'Antidiabetic': ['Blood Glucose', 'Weight', 'BMI', 'Side Effects'],
      'Stimulant': ['Heart Rate', 'Blood Pressure', 'Sleep Quality', 'Mood'],
      'SSRI': ['Mood', 'Sleep Quality', 'Energy Level', 'Side Effects'],
      'Biguanide': ['Blood Glucose', 'Weight', 'BMI', 'Side Effects'],
      'GLP-1 Receptor Agonist': ['Weight', 'BMI', 'Blood Glucose', 'Side Effects']
    };
    
    return classMetrics[drugClass] || ['General Health', 'Side Effects'];
  };

  // Toggle metric selection
  const toggleMetric = (metric) => {
    setSelectedMetrics(prev => 
      prev.includes(metric) 
        ? prev.filter(m => m !== metric)
        : [...prev, metric]
    );
  };

  // Add custom metric
  const addCustomMetric = () => {
    if (!customMetricInput.trim()) return;
    
    const newMetric = customMetricInput.trim();
    if (!selectedMetrics.includes(newMetric)) {
      setSelectedMetrics(prev => [...prev, newMetric]);
    }
    setCustomMetricInput('');
    setShowCustomMetricInput(false);
  };

  // Handle metrics confirmation
  const handleMetricsConfirmation = () => {
    setMedicationData(prev => ({
      ...prev,
      selected_metrics: selectedMetrics
    }));

    addMessage('bot', `Excellent! You've selected **${selectedMetrics.length} metrics** to track. Now, let's add some helpful notes and instructions for this medication.`, {
      type: 'notes_selection',
      actions: [
        { text: 'Add My Own Notes', action: 'add_manual_notes', secondary: true },
        { text: 'Generate AI Notes', action: 'generate_ai_notes', primary: true },
        { text: 'Skip Notes', action: 'skip_notes', tertiary: true }
      ]
    });
  };

  // Generate AI notes using BioGPT
  const generateAINotes = async () => {
    if (isLoading || isProcessing) return; // Prevent duplicate calls
    
    // Check if AI notes have already been generated
    if (aiGeneratedNotes) {
      // Check if AI notes preview message already exists
      const hasAiNotesPreview = messages.some(msg => msg.type === 'ai_notes_preview');
      if (!hasAiNotesPreview) {
        addMessage('bot', `Here are the AI-generated notes for **${medicationData.generic_name}**:`, {
          type: 'ai_notes_preview',
          notes: aiGeneratedNotes,
          actions: [
            { text: 'Use these notes', action: 'use_ai_notes' },
            { text: 'Edit notes', action: 'edit_ai_notes' },
            { text: 'Add my own notes', action: 'add_manual_notes' }
          ]
        });
      }
      return;
    }
    
    setIsLoading(true);
    addMessage('bot', `Let me generate some helpful notes about **${medicationData.generic_name}** using AI...`, {
      type: 'info'
    });
    
    try {
      // Simulate AI note generation (in production, this would call BioGPT)
      const notes = `**How to take ${medicationData.generic_name}:**\n• Take with food to reduce stomach upset\n• Drink plenty of water\n• Avoid alcohol while taking this medication\n\n**Potential side effects to watch for:**\n• Nausea, dizziness, or headache\n• Contact your doctor if side effects persist\n\n**Important reminders:**\n• Take at the same time each day\n• Don't skip doses without consulting your doctor\n• Store in a cool, dry place`;
      
      setAiGeneratedNotes(notes);
      setMedicationData(prev => ({
        ...prev,
        ai_generated_notes: notes
      }));
      
      simulateTyping(() => {
        // Check if AI notes preview message already exists
        const hasAiNotesPreview = messages.some(msg => msg.type === 'ai_notes_preview');
        if (!hasAiNotesPreview) {
          addMessage('bot', `Here are some helpful notes I generated for **${medicationData.generic_name}**:`, {
            type: 'ai_notes_preview',
            notes: notes,
            actions: [
              { text: 'Use these notes', action: 'use_ai_notes' },
              { text: 'Edit notes', action: 'edit_ai_notes' },
              { text: 'Add my own notes', action: 'add_manual_notes' }
            ]
          });
        }
      });
      
    } catch (error) {
      addMessage('bot', `I couldn't generate notes right now. Would you like to add your own notes or skip this step?`, {
        type: 'error',
        actions: [
          { text: 'Skip notes', action: 'skip_notes' }
        ]
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle manual notes input
  const handleManualNotes = (notes) => {
    setSpecialInstructions(notes);
    setMedicationData(prev => ({
      ...prev,
      special_instructions: notes
    }));
    
    addMessage('bot', `Perfect! I've noted your instructions: "${notes}". Let me confirm all the details:`, {
      type: 'final_confirmation',
      medication: { ...medicationData, special_instructions: notes },
      metrics: selectedMetrics
    });
  };

  // Save medication to database
  const saveMedication = async () => {
    if (isProcessing) return; // Prevent duplicate submissions
    
    setIsProcessing(true);
    try {
      // Get user ID from localStorage
      const userStr = localStorage.getItem('user');
      const user = userStr ? JSON.parse(userStr) : null;
      const userId = user?.id;
      
      const response = await api.post('meds/user', {
        userId: userId, // Include user ID
        medication_name: medicationData.medication_name || medicationData.generic_name,
        generic_name: medicationData.generic_name,
        strength: medicationData.strength,
        unit: medicationData.unit,
        frequency: medicationData.frequency,
        frequency_display: medicationData.frequency_display,
        drug_class: medicationData.drug_class,
        selected_metrics: selectedMetrics,
        special_instructions: medicationData.special_instructions || '',
        ai_generated_notes: medicationData.ai_generated_notes || '',
        confidence: medicationData.confidence || 0.9,
        aiValidated: true,
        start_date: new Date().toISOString().split('T')[0]
      });

      if (response.status === 201 || response.status === 200 || response.data?.success) {
        const savedMedication = response.data?.medication || response.data;
        
        // Immediately close modal and trigger refresh
        onClose();
        // Call onSuccess immediately to refresh dashboard
        onSuccess?.(savedMedication);
      } else {
        throw new Error('Failed to save medication');
      }
    } catch (error) {
      console.error('Error saving medication:', error);
      addMessage('bot', `I'm sorry, there was an error saving your medication. Please try again.`, {
        type: 'error',
        actions: [
          { text: 'Try again', action: 'retry_save' }
        ]
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle action clicks
  const handleActionClick = (action, data) => {
    switch (action) {
      case 'confirm_medication':
        setMedicationData(data);
        const strengths = getCommonStrengths(data);
        setSuggestedStrengths(strengths);
        addMessage('bot', `Great! Now, what's the dosage and strength? I suggest these common options:`, {
          type: 'strength_selection',
          strengths: strengths
        });
        setConversationState('strength_input');
        break;
        
      case 'retry_medication':
        addMessage('bot', `What medication would you like to add?`, {
          type: 'info'
        });
        setConversationState('medication_input');
        break;
        
      case 'select_suggestion':
        // User selected a suggestion - validate it again
        if (data) {
          const suggestionName = data.generic_name || data.name || data;
          handleMedicationInput(suggestionName);
        }
        break;
        
      case 'retry_save':
        saveMedication();
        break;
        
      case 'generate_ai_notes':
        generateAINotes();
        break;
        
      case 'use_ai_notes':
        addMessage('bot', `Perfect! I've added the AI-generated notes. Let me confirm all the details:`, {
          type: 'final_confirmation',
          medication: medicationData,
          metrics: selectedMetrics
        });
        break;
        
      case 'edit_ai_notes':
        setIsEditingNotes(true);
        addMessage('bot', `Please edit the notes below:`, {
          type: 'edit_notes',
          notes: aiGeneratedNotes
        });
        break;
        
      case 'add_manual_notes':
        addMessage('bot', `Please enter your special instructions or notes for **${medicationData.generic_name}**:`, {
          type: 'manual_notes_input'
        });
        setConversationState('manual_notes_input');
        break;
        
      case 'skip_notes':
        addMessage('bot', `No problem! Let me confirm all the details:`, {
          type: 'final_confirmation',
          medication: medicationData,
          metrics: selectedMetrics
        });
        break;
        
      default:
        console.log('Unknown action:', action);
    }
  };

  // Handle send message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading || isProcessing) return;

    const userMessage = inputValue.trim();
    setInputValue('');

    switch (conversationState) {
      case 'medication_input':
        await handleMedicationInput(userMessage);
        break;
      case 'strength_input':
        handleStrengthInput(userMessage);
        break;
      case 'frequency_input':
        handleFrequencyInput(userMessage);
        break;
      case 'manual_notes_input':
        handleManualNotes(userMessage);
        break;
      case 'edit_notes_input':
        setAiGeneratedNotes(userMessage);
        setMedicationData(prev => ({
          ...prev,
          ai_generated_notes: userMessage
        }));
        setIsEditingNotes(false);
        addMessage('bot', `Perfect! I've updated the notes. Let me confirm all the details:`, {
          type: 'final_confirmation',
          medication: { ...medicationData, ai_generated_notes: userMessage },
          metrics: selectedMetrics
        });
        break;
      default:
        addMessage('bot', "I'm not sure what you mean. Let's start over.", {
          type: 'error'
        });
        setConversationState('medication_input');
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
    if (message.type === 'strength_selection') {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            {message.strengths.map((strength, index) => (
              <Button
                key={index}
                onClick={() => handleStrengthInput(strength)}
                className="justify-start text-left bg-white text-neutral-900 hover:bg-primary-50 border border-neutral-200 hover:border-primary-300 rounded-xl px-3 py-2 text-sm font-medium transition-all duration-200"
                size="sm"
              >
                {strength}
              </Button>
            ))}
          </div>
          <p className="text-sm text-neutral-600">
            Or type a custom strength (e.g., "500mg", "10ml")
          </p>
        </div>
      );
    }
    
    if (message.type === 'frequency_selection') {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-3">
            {message.frequencies.map((freq, index) => (
              <Button
                key={index}
                onClick={() => handleFrequencyInput(freq.value)}
                className="justify-start text-left bg-white text-neutral-900 hover:bg-primary-50 border border-neutral-200 hover:border-primary-300 rounded-xl px-3 py-2 text-sm font-medium transition-all duration-200"
                size="sm"
              >
                <div className="text-left">
                  <div className="font-medium">{freq.label}</div>
                  <div className="text-xs text-neutral-400">{freq.description}</div>
                </div>
              </Button>
            ))}
          </div>
          <p className="text-sm text-neutral-600">
            Or type your own frequency (e.g., "twice a day", "every 2 days")
          </p>
        </div>
      );
    }
    
    if (message.type === 'metric_selection') {
      return (
        <div className="space-y-6">
            <div className="text-sm text-neutral-600 mb-4">
            Suggested metrics (pre-selected):
          </div>
          <div className="flex flex-wrap gap-3 mb-6">
            {message.suggestedMetrics.map((metric, index) => (
              <Badge
                key={index}
                className="cursor-pointer bg-white text-neutral-900 hover:bg-neutral-200 rounded-xl px-3 py-1 text-xs font-medium transition-all duration-200"
                onClick={() => toggleMetric(metric)}
              >
                {metric}
              </Badge>
            ))}
          </div>
          
            <div className="text-sm text-neutral-600 mb-4">
            All available health metrics:
          </div>
          <div className="grid grid-cols-2 gap-3 max-h-80 overflow-y-auto pr-2">
            {message.allMetrics.map((metric, index) => (
              <div
                key={index}
                className="flex items-center space-x-3 p-3 rounded-xl border border-neutral-200 hover:bg-primary-50 transition-colors cursor-pointer"
                onClick={() => toggleMetric(metric)}
              >
                <Checkbox
                  checked={selectedMetrics.includes(metric)}
                  className="data-[state=checked]:bg-white data-[state=checked]:border-white"
                />
                <span className="text-sm text-neutral-900">{metric}</span>
              </div>
            ))}
          </div>
          
          {/* Custom Metric Input */}
          <div className="space-y-3">
            <div className="text-sm text-neutral-600">Add custom metric:</div>
            <div className="flex space-x-2">
              <Input
                value={customMetricInput}
                onChange={(e) => setCustomMetricInput(e.target.value)}
                placeholder="Enter custom metric name..."
                className="flex-1 bg-white border border-neutral-200 text-neutral-900 placeholder-neutral-500 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 focus:outline-none rounded-xl px-3 py-2"
                onKeyPress={(e) => e.key === 'Enter' && addCustomMetric()}
              />
              <Button
                onClick={addCustomMetric}
                disabled={!customMetricInput.trim()}
                className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                size="sm"
              >
                Add
              </Button>
            </div>
          </div>
          
          <div className="flex items-center justify-between pt-4">
            <div className="text-sm text-muted-foreground">
              {selectedMetrics.length} selected
            </div>
            <Button
              onClick={handleMetricsConfirmation}
              disabled={selectedMetrics.length === 0}
              size="sm"
            >
              Continue
            </Button>
          </div>
        </div>
      );
    }
    
    if (message.type === 'ai_notes_preview') {
      return (
        <div className="space-y-4">
          <div className="bg-primary-500/10 border border-primary-500/20 rounded-lg p-4">
            <div className="prose prose-sm max-w-none text-foreground">
              {message.notes.split('\n').map((line, index) => (
                <div key={index} className="mb-2">
                  {line.split('**').map((part, partIndex) => 
                    partIndex % 2 === 1 ? <strong key={partIndex} className="text-foreground">{part}</strong> : part
                  )}
                </div>
              ))}
            </div>
          </div>
          <div className="flex gap-2">
            {message.actions?.map((action, index) => (
              <Button
                key={index}
                onClick={() => handleActionClick(action.action)}
                variant={action.primary ? "default" : action.secondary ? "outline" : "ghost"}
                className={cn(
                  action.primary && "bg-primary text-primary-foreground",
                  action.secondary && "border-primary text-primary hover:bg-primary/10",
                  action.tertiary && "text-muted-foreground hover:text-foreground"
                )}
                size="sm"
              >
                {action.text}
              </Button>
            ))}
          </div>
        </div>
      );
    }
    
    if (message.type === 'final_confirmation') {
      return (
        <div className="bg-primary-500/10 border border-primary-500/20 rounded-lg p-4">
          <div className="space-y-3">
            <div className="text-lg font-semibold mb-4">Medication Summary</div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div><strong>Medication:</strong> {message.medication.generic_name}</div>
              <div><strong>Strength:</strong> {message.medication.strength}{message.medication.unit}</div>
              <div><strong>Frequency:</strong> {message.medication.frequency_display}</div>
              <div><strong>Drug Class:</strong> {message.medication.drug_class}</div>
              <div><strong>Metrics:</strong> {message.metrics.length} selected</div>
            </div>
            {message.medication.special_instructions && (
              <div><strong>Instructions:</strong> {message.medication.special_instructions}</div>
            )}
            {message.medication.ai_generated_notes && (
              <div><strong>AI Notes:</strong> {message.medication.ai_generated_notes}</div>
            )}
            <div className="flex gap-2 mt-4">
              <Button
                onClick={saveMedication}
                size="sm"
                className="bg-medical-600 hover:bg-medical-700"
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                Add Medication
              </Button>
              <Button
                onClick={() => setConversationState('medication_input')}
                variant="outline"
                size="sm"
              >
                Start Over
              </Button>
            </div>
          </div>
        </div>
      );
    }
    
    if (message.type === 'success_card') {
      return (
        <div className="bg-medical-500/10 border border-medical-500/20 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <CheckCircle className="w-5 h-5 text-medical-500" />
            <span className="font-semibold text-medical-500">Medication Added Successfully!</span>
          </div>
          <div className="space-y-1 text-sm">
            <div><strong>Name:</strong> {message.medication.medication_name || message.medication.name}</div>
            <div><strong>Strength:</strong> {message.medication.strength}{message.medication.unit}</div>
            <div><strong>Frequency:</strong> {message.medication.frequency_display || message.medication.frequency}</div>
            <div><strong>Metrics:</strong> {message.medication.selected_metrics?.join(', ') || 'None'}</div>
          </div>
        </div>
      );
    }
    
    return (
      <div className="prose prose-sm max-w-none text-foreground">
        {message.content.split('**').map((part, index) => 
          index % 2 === 1 ? <strong key={index} className="text-foreground">{part}</strong> : part
        )}
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Grok-style Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      
        {/* Grok-style Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className="relative w-full max-w-4xl max-h-[90vh] bg-white border border-neutral-200 rounded-3xl shadow-2xl overflow-hidden flex flex-col"
      >
        {/* Grok-style Header */}
        <div className="bg-gradient-to-r from-primary-600 to-primary-700 border-b border-primary-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center">
                <Pill className="w-5 h-5 text-primary-600" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-white">Medication Assistant</h2>
                <p className="text-sm text-primary-100">AI-powered medication validation</p>
              </div>
            </div>
            <Button
              onClick={onClose}
              variant="ghost"
              size="sm"
              className="text-white hover:bg-primary-800 hover:text-white"
              aria-label="Close chat"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
        </div>

        {/* Grok-style Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6 bg-neutral-50">
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
                  {/* Grok-style Avatar */}
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1",
                    message.type === 'user' 
                      ? "bg-primary-600 text-white" 
                      : "bg-white text-neutral-900 border border-neutral-200"
                  )} aria-label={message.type === 'user' ? 'User avatar' : 'AI assistant avatar'}>
                    {message.type === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>
                  
                  {/* Grok-style Message Content */}
                  <div className={cn(
                    "flex-1 min-w-0",
                    message.type === 'user' ? "text-right" : "text-left"
                  )}>
                    <div className={cn(
                      "inline-block p-4 rounded-2xl max-w-full text-sm leading-relaxed",
                      message.type === 'user'
                        ? "bg-primary-600 text-white rounded-br-lg"
                        : "bg-white text-neutral-900 border border-neutral-200 rounded-bl-lg"
                    )}>
                      {renderMessageContent(message)}
                      
                      {/* Grok-style Actions */}
                      {message.actions && (
                        <div className="flex flex-wrap gap-2 mt-4">
                          {message.actions.map((action, index) => (
                            <Button
                              key={index}
                              onClick={() => handleActionClick(action.action, action.data)}
                              size="sm"
                              className={cn(
                                "text-xs font-medium transition-all duration-200 rounded-xl px-3 py-2",
                                message.type === 'user' 
                                  ? "bg-primary-700 text-white hover:bg-primary-800 border border-primary-800" 
                                  : "bg-primary-50 text-primary-700 hover:bg-primary-100 border border-primary-200"
                              )}
                              aria-label={action.text}
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
          
          {/* Grok-style Typing Indicator */}
          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="flex items-start space-x-4 max-w-[85%]">
                <div className="w-8 h-8 rounded-full bg-white text-primary-600 border border-neutral-200 flex items-center justify-center flex-shrink-0 mt-1" aria-label="AI assistant typing">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="inline-block p-4 rounded-2xl rounded-bl-lg bg-white text-neutral-900 border border-neutral-200">
                    <div className="flex items-center space-x-3">
                      <div className="flex space-x-1" aria-label="Typing indicator">
                        <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                      <span className="text-sm text-neutral-600">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Grok-style Input */}
        <div className="border-t border-neutral-200 p-6 bg-white">
          <div className="flex space-x-3">
            <div className="flex-1 relative">
              <Input
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about your medication..."
                disabled={isLoading}
                className="w-full px-4 py-3 bg-white border border-neutral-200 rounded-2xl text-neutral-900 placeholder-neutral-500 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 focus:outline-none"
              />
            </div>
            <Button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-2xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default EnhancedMedicationChat;
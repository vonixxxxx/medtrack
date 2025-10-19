import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Sparkles, X } from 'lucide-react';

const AIChat = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'ai',
      content: 'Hello! I\'m your AI health assistant. How can I help you today?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse = generateAIResponse(userMessage.content);
      const aiMessage = {
        id: Date.now() + 1,
        role: 'ai',
        content: aiResponse,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiMessage]);
      setIsTyping(false);
    }, 1000);
  };

  const generateAIResponse = (userInput) => {
    const input = userInput.toLowerCase();
    
    if (input.includes('blood pressure') || input.includes('bp')) {
      return 'Blood pressure is an important vital sign. Normal blood pressure is typically below 120/80 mmHg. If you have concerns about your blood pressure, please consult with your healthcare provider for personalized advice.';
    }
    
    if (input.includes('medication') || input.includes('med')) {
      return 'I can help you understand your medications! Make sure to take them as prescribed by your doctor. If you have questions about side effects, interactions, or dosing, it\'s always best to consult your pharmacist or healthcare provider.';
    }
    
    if (input.includes('diabetes') || input.includes('blood sugar')) {
      return 'Managing diabetes involves monitoring blood sugar levels, taking medications as prescribed, maintaining a healthy diet, and regular exercise. Always follow your healthcare provider\'s recommendations for your specific situation.';
    }
    
    if (input.includes('side effect') || input.includes('adverse')) {
      return 'If you\'re experiencing side effects from your medications, contact your healthcare provider immediately. Some side effects are normal, but others may require medical attention. Never stop taking prescribed medications without consulting your doctor first.';
    }
    
    if (input.includes('schedule') || input.includes('time')) {
      return 'Consistent medication timing is important for effectiveness. Try to take your medications at the same time each day. Setting reminders or using a pill organizer can help maintain your schedule.';
    }
    
    return 'I understand you\'re asking about health-related topics. While I can provide general information, please remember that I\'m not a substitute for professional medical advice. For specific health concerns, always consult with your healthcare provider.';
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    
      {/* Messages */}
      
        
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  message.role === 'user' 
                    ? 'bg-gradient-primary text-white' 
                    : 'bg-gradient-to-r from-success-500 to-success-600 text-white'
                }`}>
                  {message.role === 'user' ?  : }
                
                
                {/* Message bubble */}
                <div className={`px-4 py-3 rounded-2xl ${
                  message.role === 'user'
                    ? 'bg-gradient-primary text-white'
                    : 'bg-secondary-100 text-secondary-900 border border-secondary-200'
                }`}>
                  {message.content}
                  
                    {message.timestamp.toLocaleTimeString()}
                  
                
              
            
          ))}
        
        
        {/* Typing indicator */}
        {isTyping && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start"
          >
            
              
                
              
              
                
                  
                  
                  
                
              
            
          
        )}
        
        
      

      {/* Input */}
      
        
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your medications or health..."
            className="input pr-12"
            disabled={isTyping}
          />
          
            
          
        
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleSend}
          disabled={!input.trim() || isTyping}
          className="btn-primary px-4 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          
        
      
    
  );
};

export default AIChat;
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { MessageCircle, X } from 'lucide-react';
import { Button } from './ui/button';

export const FloatingAIButton = ({ onChatOpen }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="fixed bottom-6 right-6 z-40">
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        className="flex flex-col items-end space-y-3"
      >
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="bg-white border border-neutral-200 rounded-2xl p-4 shadow-medium"
          >
            <p className="text-sm text-neutral-600 mb-3">Need help?</p>
            <Button
              onClick={onChatOpen}
              size="sm"
              className="w-full"
            >
              <MessageCircle className="w-4 h-4 mr-2" />
              Chat with AI
            </Button>
          </motion.div>
        )}
        
        <Button
          onClick={() => setIsExpanded(!isExpanded)}
          size="lg"
          className="w-14 h-14 rounded-full shadow-lg hover:shadow-xl transition-shadow"
          aria-label={isExpanded ? "Close AI assistant menu" : "Open AI assistant menu"}
        >
          <motion.div
            animate={{ rotate: isExpanded ? 45 : 0 }}
            transition={{ duration: 0.2 }}
          >
            {isExpanded ? (
              <X className="w-6 h-6" />
            ) : (
              <MessageCircle className="w-6 h-6" />
            )}
          </motion.div>
        </Button>
      </motion.div>
    </div>
  );
};






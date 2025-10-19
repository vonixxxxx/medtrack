import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent } from './ui/card';

export const MedTrackCard = ({ children, className = "", ...props }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={className}
      {...props}
    >
      <Card className="bg-card border-border shadow-sm hover:shadow-md transition-shadow">
        <CardContent className="p-6">
          {children}
        </CardContent>
      </Card>
    </motion.div>
  );
};






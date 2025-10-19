import React from 'react';
import { motion } from 'framer-motion';

const Card = ({ 
  children, 
  className = '', 
  hover = true, 
  glow = false, 
  gradient = false,
  onClick, 
  ...props 
}) => {
  const baseClasses = 'card';
  const hoverClasses = hover ? 'card-hover' : '';
  const glowClasses = glow ? 'shadow-glow' : '';
  const gradientClasses = gradient ? 'card-gradient' : '';
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={hover ? { scale: 1.02, y: -2 } : {}}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={`${baseClasses} ${hoverClasses} ${glowClasses} ${gradientClasses} ${className}`}
      onClick={onClick}
      {...props}
    >
      {children}
    
  );
};

export default Card;
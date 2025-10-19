import React from 'react';
import { motion } from 'framer-motion';

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  onClick, 
  disabled = false, 
  icon, 
  loading = false,
  fullWidth = false,
  ...props 
}) => {
  const baseClasses = 'font-semibold rounded-xl transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
  const variants = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    ghost: 'btn-ghost',
    danger: 'btn-danger',
    success: 'btn-success',
  };
  const sizes = { 
    sm: 'px-4 py-2 text-sm', 
    md: 'px-6 py-3 text-base', 
    lg: 'px-8 py-4 text-lg' 
  };
  const fullWidthClass = fullWidth ? 'w-full' : '';
  const loadingClasses = loading ? 'btn-loading' : '';
  
  return (
    <motion.button
      whileHover={!disabled && !loading ? { scale: 1.05, y: -1 } : {}}
      whileTap={!disabled && !loading ? { scale: 0.95 } : {}}
      className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${fullWidthClass} ${loadingClasses} ${className}`}
      onClick={onClick}
      disabled={disabled || loading}
      {...props}
    >
      
        {icon && {icon}}
        {children}
        {loading && Loading...}
      
    
  );
};

export default Button;
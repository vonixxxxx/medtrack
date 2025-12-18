import { motion } from "framer-motion";
import { useReducedMotion } from "framer-motion";
import React from "react";

/**
 * Reusable card component with consistent styling
 * Supports both default export (for landing page) and named exports (for existing components)
 */
export function Card({ 
  children, 
  className = "",
  hover = true,
  delay = 0,
  onClick,
  href
}) {
  const shouldReduceMotion = useReducedMotion();
  const Component = href ? "a" : onClick ? "div" : "div";

  const baseClasses = "p-6 lg:p-8 bg-white rounded-xl border border-gray-200 transition-all";
  const hoverClasses = hover 
    ? "hover:border-blue-300 hover:shadow-lg cursor-pointer" 
    : "";

  const props = {
    className: `${baseClasses} ${hoverClasses} ${className}`,
    onClick,
    href,
    ...(onClick && { role: "button", tabIndex: 0 }),
  };

  return (
    <motion.div
      initial={shouldReduceMotion ? { opacity: 1 } : { opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay }}
      whileHover={hover && !shouldReduceMotion ? { y: -4, scale: 1.02 } : {}}
    >
      <Component {...props}>
        {children}
      </Component>
    </motion.div>
  );
}

/**
 * CardContent - Simple wrapper for card content (for compatibility with existing code)
 */
export function CardContent({ children, className = "" }) {
  return (
    <div className={`p-6 ${className}`}>
      {children}
    </div>
  );
}

// Default export for backward compatibility with landing page components
export default Card;

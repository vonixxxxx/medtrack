import { motion, useReducedMotion } from "framer-motion";

/**
 * Reusable fade-in animation component
 */
export default function FadeIn({ 
  children, 
  delay = 0, 
  duration = 0.6,
  className = "",
  direction = "up"
}) {
  const shouldReduceMotion = useReducedMotion();

  const variants = {
    up: { opacity: 0, y: 20 },
    down: { opacity: 0, y: -20 },
    left: { opacity: 0, x: -20 },
    right: { opacity: 0, x: 20 },
    fade: { opacity: 0 },
  };

  return (
    <motion.div
      initial={shouldReduceMotion ? { opacity: 1 } : variants[direction]}
      whileInView={{ opacity: 1, x: 0, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration, delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}






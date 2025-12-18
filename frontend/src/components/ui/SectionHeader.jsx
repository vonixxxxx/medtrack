import { motion } from "framer-motion";
import { useReducedMotion } from "framer-motion";

/**
 * Consistent section headers with animation
 */
export default function SectionHeader({ 
  title, 
  subtitle, 
  className = "",
  align = "center",
  tag = "h2"
}) {
  const shouldReduceMotion = useReducedMotion();
  const Tag = tag;

  const alignClasses = {
    center: "text-center",
    left: "text-left",
    right: "text-right",
  };

  return (
    <motion.div
      initial={shouldReduceMotion ? { opacity: 1 } : { opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      className={`${alignClasses[align]} mb-12 ${className}`}
    >
      <Tag className="text-3xl lg:text-4xl font-semibold text-gray-900 mb-4">
        {title}
      </Tag>
      {subtitle && (
        <p className="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
          {subtitle}
        </p>
      )}
    </motion.div>
  );
}






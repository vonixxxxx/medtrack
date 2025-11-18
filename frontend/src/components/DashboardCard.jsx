import { motion, useReducedMotion } from 'framer-motion';

export default function DashboardCard({ 
  title, 
  children, 
  className = "",
  icon,
  action,
  variant = "default"
}) {
  const prefersReducedMotion = useReducedMotion();
  
  return (
    <motion.div 
      initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.3,
        ease: [0.16, 1, 0.3, 1], // ease-out-quint
      }}
      whileHover={prefersReducedMotion ? {} : { y: -2 }}
      className={`
        bg-white rounded-2xl border border-neutral-200 shadow-soft
        hover:shadow-medium transition-all duration-200
        ${variant === "patient" ? "hover:border-primary-300" : ""}
        ${variant === "clinician" ? "hover:border-primary-300" : ""}
        p-6 lg:p-8
        ${className}
      `}
    >
      {(title || icon) && (
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            {icon && (
              <div className="p-2.5 bg-primary-50 rounded-xl text-primary-600">
                {icon}
              </div>
            )}
            {title && (
              <h3 className="text-xl font-semibold text-neutral-900 tracking-tight">
                {title}
              </h3>
            )}
          </div>
          {action && (
            <div className="flex-shrink-0">
              {action}
            </div>
          )}
        </div>
      )}
      <div className="text-neutral-700">
        {children}
      </div>
    </motion.div>
  );
}

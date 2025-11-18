import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

export function StatCard({
  icon: Icon,
  label,
  value,
  trend,
  color = "primary",
  delay = 0,
  className,
}) {
  const colorClasses = {
    primary: {
      icon: "bg-primary-50 text-primary-600",
      trend: "text-primary-600",
    },
    medical: {
      icon: "bg-medical-50 text-medical-600",
      trend: "text-medical-600",
    },
    warning: {
      icon: "bg-warning-50 text-warning-600",
      trend: "text-warning-600",
    },
    error: {
      icon: "bg-error-50 text-error-600",
      trend: "text-error-600",
    },
  };

  const colors = colorClasses[color];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.3,
        delay,
        ease: [0.16, 1, 0.3, 1], // ease-out-quint
      }}
      whileHover={{ y: -2 }}
      className={cn(
        "bg-white rounded-2xl p-6 border border-neutral-200 shadow-soft",
        "hover:shadow-medium transition-all duration-200",
        className
      )}
    >
      <div className="flex items-start justify-between mb-4">
        <div className={cn("p-3 rounded-xl", colors.icon)}>
          <Icon size={20} strokeWidth={2} />
        </div>
        {trend && (
          <div className={cn("text-xs font-semibold", colors.trend)}>
            {trend.isPositive ? "↑" : "↓"} {Math.abs(trend.value)}%
          </div>
        )}
      </div>
      <div className="space-y-1">
        <div className="text-2xl font-bold text-neutral-900 tracking-tight">
          {value}
        </div>
        <div className="text-sm font-medium text-neutral-600">{label}</div>
      </div>
    </motion.div>
  );
}


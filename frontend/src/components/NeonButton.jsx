import { motion } from "framer-motion";
import { cn } from "../lib/utils";

export const NeonButton = ({ 
  children, 
  onClick, 
  variant = "primary",
  size = "default",
  className,
  disabled = false 
}) => {
  const baseStyles = "px-5 py-2.5 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed text-sm relative overflow-hidden";
  
  const sizeStyles = {
    sm: "px-3 py-1.5 text-xs",
    default: "px-5 py-2.5 text-sm",
    lg: "px-6 py-3 text-base"
  };
  
  const variants = {
    primary: "bg-gradient-to-r from-foreground to-foreground/90 text-background hover:from-foreground/90 hover:to-foreground shadow-lg hover:shadow-xl",
    secondary: "bg-gradient-to-r from-secondary to-secondary/90 text-foreground border border-border hover:from-muted hover:to-muted shadow-md hover:shadow-lg",
    ghost: "text-foreground hover:bg-secondary/50",
    success: "bg-gradient-to-r from-green-600 to-green-700 text-white hover:from-green-700 hover:to-green-800 shadow-lg hover:shadow-xl",
    danger: "bg-gradient-to-r from-red-600 to-red-700 text-white hover:from-red-700 hover:to-red-800 shadow-lg hover:shadow-xl"
  };

  return (
    <motion.button
      whileHover={!disabled ? { 
        scale: 1.02,
        boxShadow: "0 10px 25px rgba(0, 0, 0, 0.2)"
      } : {}}
      whileTap={!disabled ? { scale: 0.98 } : {}}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        baseStyles, 
        sizeStyles[size], 
        variants[variant], 
        className
      )}
    >
      <span className="relative z-10">{children}</span>
      {variant === "primary" && (
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -skew-x-12 transform translate-x-[-100%] hover:translate-x-[100%] transition-transform duration-700"></div>
      )}
    </motion.button>
  );
};
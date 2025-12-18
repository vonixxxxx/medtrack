"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useInView } from "framer-motion";
import { useRef } from "react";
import { ArrowRight } from "lucide-react";

interface WorkflowStepProps {
  step: number;
  title: string;
  description: string;
  icon: React.ReactNode;
  delay?: number;
  isLast?: boolean;
  className?: string;
}

export function WorkflowStep({
  step,
  title,
  description,
  icon,
  delay = 0,
  isLast = false,
  className,
}: WorkflowStepProps) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });

  return (
    <div className={cn("relative", className)}>
      <motion.div
        ref={ref}
        initial={{ opacity: 0, x: -20 }}
        animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }}
        transition={{ duration: 0.6, delay }}
        className="flex flex-col items-center text-center"
      >
        {/* Step Number Circle */}
        <div className="relative mb-6">
          <div className="w-16 h-16 md:w-20 md:h-20 rounded-full bg-gradient-to-br from-blue-600 to-blue-700 flex items-center justify-center shadow-lg">
            <div className="text-white text-xl md:text-2xl font-bold">{step}</div>
          </div>
          <div className="absolute inset-0 rounded-full bg-blue-400 opacity-20 animate-ping" />
        </div>

        {/* Icon */}
        <div className="mb-4 text-blue-600">{icon}</div>

        {/* Content */}
        <h3 className="text-xl md:text-2xl font-bold text-gray-900 mb-2">
          {title}
        </h3>
        <p className="text-gray-600 text-sm md:text-base leading-relaxed max-w-xs">
          {description}
        </p>
      </motion.div>

      {/* Connector Arrow (Desktop) */}
      {!isLast && (
        <div className="hidden lg:block absolute top-10 left-full w-full -translate-x-1/2">
          <motion.div
            initial={{ scaleX: 0 }}
            animate={isInView ? { scaleX: 1 } : { scaleX: 0 }}
            transition={{ duration: 0.8, delay: delay + 0.3 }}
            className="h-0.5 bg-gradient-to-r from-blue-400 to-blue-600 origin-left"
          />
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -10 }}
            transition={{ duration: 0.5, delay: delay + 0.6 }}
            className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 text-blue-600"
          >
            <ArrowRight className="w-6 h-6" />
          </motion.div>
        </div>
      )}

      {/* Connector Arrow (Mobile) */}
      {!isLast && (
        <div className="lg:hidden flex justify-center my-8">
          <motion.div
            initial={{ opacity: 0, rotate: -90 }}
            animate={isInView ? { opacity: 1, rotate: -90 } : { opacity: 0, rotate: -90 }}
            transition={{ duration: 0.5, delay: delay + 0.3 }}
            className="text-blue-600"
          >
            <ArrowRight className="w-6 h-6" />
          </motion.div>
        </div>
      )}
    </div>
  );
}




"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useInView } from "framer-motion";
import { useRef } from "react";

interface InfoCardProps {
  title: string;
  description: string;
  icon?: React.ReactNode;
  delay?: number;
  className?: string;
  variant?: "default" | "highlight";
}

export function InfoCard({
  title,
  description,
  icon,
  delay = 0,
  className,
  variant = "default",
}: InfoCardProps) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.5, delay }}
      className={cn(
        "p-6 md:p-8 rounded-xl border transition-all duration-300",
        variant === "highlight"
          ? "bg-gradient-to-br from-blue-50 to-white border-blue-200 shadow-lg"
          : "bg-white border-gray-200 shadow-md hover:shadow-lg",
        className
      )}
    >
      {icon && (
        <div className="mb-4 text-blue-600">{icon}</div>
      )}
      <h3 className="text-xl md:text-2xl font-bold text-gray-900 mb-3">
        {title}
      </h3>
      <p className="text-gray-600 leading-relaxed">{description}</p>
    </motion.div>
  );
}




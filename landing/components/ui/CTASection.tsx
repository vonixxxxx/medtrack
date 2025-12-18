"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import { ArrowRight, Calendar } from "lucide-react";
import { cn } from "@/lib/utils";

interface CTASectionProps {
  headline?: string;
  subheadline?: string;
  primaryCTA?: {
    text: string;
    href: string;
    onClick?: () => void;
  };
  secondaryCTA?: {
    text: string;
    href: string;
    onClick?: () => void;
  };
  variant?: "default" | "gradient" | "minimal";
  className?: string;
}

export function CTASection({
  headline = "Ready to Transform Healthcare?",
  subheadline = "Join leading healthcare institutions using MedTrack for enterprise-grade clinical intelligence.",
  primaryCTA = {
    text: "Request Enterprise Demo",
    href: "/enterprise",
  },
  secondaryCTA = {
    text: "Schedule a Meeting",
    href: "/contact",
  },
  variant = "gradient",
  className,
}: CTASectionProps) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  const variantStyles = {
    default: "bg-white border-t border-gray-200",
    gradient: "bg-gradient-to-br from-blue-600 via-blue-700 to-blue-800",
    minimal: "bg-gray-50",
  };

  const textColor = variant === "gradient" ? "text-white" : "text-gray-900";
  const subtextColor = variant === "gradient" ? "text-blue-100" : "text-gray-600";

  return (
    <section
      ref={ref}
      className={cn("py-16 md:py-24 relative overflow-hidden", variantStyles[variant], className)}
    >
      {variant === "gradient" && (
        <div className="absolute inset-0 opacity-10">
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: `radial-gradient(circle at 2px 2px, white 1px, transparent 0)`,
              backgroundSize: "40px 40px",
            }}
          />
        </div>
      )}

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
          transition={{ duration: 0.6 }}
          className="max-w-4xl mx-auto text-center"
        >
          <h2 className={cn("text-3xl md:text-5xl font-bold mb-6", textColor)}>
            {headline}
          </h2>
          {subheadline && (
            <p className={cn("text-lg md:text-xl mb-10 leading-relaxed", subtextColor)}>
              {subheadline}
            </p>
          )}

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 md:gap-6">
            <motion.a
              href={primaryCTA.href}
              onClick={primaryCTA.onClick}
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              className={cn(
                "group px-8 py-4 rounded-lg font-semibold text-lg shadow-lg hover:shadow-xl transition-all flex items-center gap-2",
                variant === "gradient"
                  ? "bg-white text-blue-600 hover:bg-blue-50"
                  : "bg-blue-600 text-white hover:bg-blue-700"
              )}
            >
              {primaryCTA.text}
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </motion.a>

            {secondaryCTA && (
              <motion.a
                href={secondaryCTA.href}
                onClick={secondaryCTA.onClick}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className={cn(
                  "group px-8 py-4 rounded-lg font-semibold text-lg border-2 transition-all flex items-center gap-2",
                  variant === "gradient"
                    ? "bg-blue-800/50 text-white border-blue-400 hover:bg-blue-700/50"
                    : "bg-white text-gray-900 border-gray-300 hover:border-blue-600 hover:text-blue-600"
                )}
              >
                <Calendar className="w-5 h-5" />
                {secondaryCTA.text}
              </motion.a>
            )}
          </div>
        </motion.div>
      </div>
    </section>
  );
}




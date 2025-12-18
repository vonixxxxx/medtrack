"use client";

import { useEffect, useState, useRef } from "react";
import { motion, useInView } from "framer-motion";
import { Users, Stethoscope, Cpu } from "lucide-react";

interface HighlightItem {
  label: string;
  value: number;
  suffix: string;
  icon: typeof Users;
  description?: string;
}

const items: HighlightItem[] = [
  {
    label: "Patient features",
    value: 25,
    suffix: "+",
    icon: Users,
    description: "Continuously expanding",
  },
  {
    label: "Clinician features",
    value: 17,
    suffix: "+",
    icon: Stethoscope,
    description: "Enterprise-ready",
  },
  {
    label: "Core engines",
    value: 5,
    suffix: "",
    icon: Cpu,
    description: "AI-powered",
  },
];

export default function PlatformHighlights() {
  const [counts, setCounts] = useState(items.map(() => 0));
  const [hasAnimated, setHasAnimated] = useState(false);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  useEffect(() => {
    if (isInView && !hasAnimated) {
      setHasAnimated(true);
      const anim = setInterval(() => {
        setCounts((prev) =>
          prev.map((v, i) => {
            const target = items[i].value;
            if (v >= target) {
              clearInterval(anim);
              return target;
            }
            return Math.min(target, v + Math.ceil(target / 20));
          })
        );
      }, 40);

      return () => clearInterval(anim);
    }
  }, [isInView, hasAnimated]);

  return (
    <section className="py-12 lg:py-16 bg-gradient-to-br from-gray-50 to-primary-50/30">
      <div className="container mx-auto px-6" ref={ref}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
          {items.map((it, idx) => {
            const Icon = it.icon;
            return (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: idx * 0.1, duration: 0.6 }}
                whileHover={{ y: -4, scale: 1.02 }}
                className="p-6 lg:p-8 bg-white/80 backdrop-blur-sm rounded-xl border border-gray-200 hover:border-primary-300 transition-all shadow-sm hover:shadow-lg"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div className="p-3 bg-primary-100 rounded-lg">
                    <Icon className="w-6 h-6 text-primary-600" />
                  </div>
                  <div className="flex-1">
                    <div className="text-3xl lg:text-4xl font-bold text-gray-900 mb-1">
                      {counts[idx]}
                      {it.suffix}
                    </div>
                    <div className="text-sm font-medium text-gray-700">{it.label}</div>
                  </div>
                </div>
                {it.description && (
                  <p className="text-xs text-gray-500 mt-2">{it.description}</p>
                )}
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}






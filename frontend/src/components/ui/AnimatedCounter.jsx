import { useEffect, useState, useRef } from "react";
import { useInView, useReducedMotion } from "framer-motion";

/**
 * Reusable animated counter component
 */
export default function AnimatedCounter({ 
  value, 
  suffix = "", 
  duration = 2000,
  className = ""
}) {
  const [count, setCount] = useState(0);
  const [hasAnimated, setHasAnimated] = useState(false);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  const shouldReduceMotion = useReducedMotion();

  useEffect(() => {
    if (isInView && !hasAnimated && !shouldReduceMotion) {
      setHasAnimated(true);
      const steps = Math.ceil(duration / 40);
      const increment = value / steps;
      
      const anim = setInterval(() => {
        setCount((prev) => {
          const next = Math.min(value, prev + increment);
          if (next >= value) {
            clearInterval(anim);
            return value;
          }
          return next;
        });
      }, 40);

      return () => clearInterval(anim);
    } else if (isInView && shouldReduceMotion) {
      setCount(value);
    }
  }, [isInView, hasAnimated, value, duration, shouldReduceMotion]);

  return (
    <div ref={ref} className={className}>
      {Math.floor(count)}{suffix}
    </div>
  );
}






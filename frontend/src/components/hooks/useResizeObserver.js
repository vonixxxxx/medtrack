import { useEffect, useState, useRef } from 'react';

/**
 * Hook to observe element resize
 * Returns the size of the element whenever it changes
 */
export const useResizeObserver = (ref) => {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const observerRef = useRef(null);

  useEffect(() => {
    if (!ref.current) return;

    const element = ref.current;

    // Create ResizeObserver
    observerRef.current = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setSize({ width, height });
      }
    });

    // Start observing
    observerRef.current.observe(element);

    // Initial size
    setSize({
      width: element.clientWidth,
      height: element.clientHeight
    });

    // Cleanup
    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [ref]);

  return size;
};



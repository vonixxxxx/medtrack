import { useEffect, useState, useRef } from 'react';

/**
 * Hook to observe element resize using ResizeObserver
 * Returns the size of the observed element
 */
export function useResizeObserver(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const observerRef = useRef(null);

  useEffect(() => {
    if (!ref.current) return;

    const element = ref.current;

    // Create ResizeObserver
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setSize({ width, height });
      }
    });

    observer.observe(element);
    observerRef.current = observer;

    // Initial size
    setSize({
      width: element.clientWidth || 0,
      height: element.clientHeight || 0,
    });

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [ref]);

  return size;
}


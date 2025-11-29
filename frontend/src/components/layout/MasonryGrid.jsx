import { useEffect, useRef, useState, useCallback } from 'react';
import { useResizeObserver } from '../../hooks/useResizeObserver';

/**
 * MasonryGrid Component
 * Creates a true masonry layout where tiles flow in columns
 * with consistent vertical and horizontal margins
 * Handles dynamic content loading and resizing
 */
export const MasonryGrid = ({ children, columns = 2, gap = 24, className = '' }) => {
  const containerRef = useRef(null);
  const tilesRef = useRef([]);
  const [positions, setPositions] = useState([]);
  const [containerHeight, setContainerHeight] = useState(0);
  const resizeObserversRef = useRef([]);
  const mutationObserverRef = useRef(null);

  // Calculate masonry layout with proper height measurement
  const calculateLayout = useCallback(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const tiles = Array.from(container.children).filter(
      child => child.nodeType === Node.ELEMENT_NODE
    );
    
    if (tiles.length === 0) {
      setContainerHeight(0);
      return;
    }

    // Calculate column width
    const containerWidth = container.clientWidth;
    if (containerWidth === 0) return; // Not yet rendered
    
    const totalGap = gap * (columns - 1);
    const columnWidth = (containerWidth - totalGap) / columns;

    // Track the bottom Y coordinate of each column
    const columnHeights = new Array(columns).fill(0);
    const newPositions = [];

    tiles.forEach((tile, index) => {
      // Find the column with the smallest current height
      const minCol = columnHeights.indexOf(Math.min(...columnHeights));
      
      // Calculate position
      const x = minCol * (columnWidth + gap);
      const y = columnHeights[minCol];

      // Get tile height - temporarily reset positioning to measure accurately
      const originalPosition = tile.style.position;
      const originalLeft = tile.style.left;
      const originalTop = tile.style.top;
      const originalWidth = tile.style.width;
      const originalTransform = tile.style.transform;
      const originalVisibility = tile.style.visibility;
      
      // Reset to static positioning for accurate measurement
      tile.style.position = 'static';
      tile.style.left = 'auto';
      tile.style.top = 'auto';
      tile.style.width = `${columnWidth}px`;
      tile.style.transform = 'none';
      tile.style.visibility = 'hidden'; // Hide during measurement to prevent flash
      
      // Force reflow to get accurate height
      void tile.offsetHeight;
      
      // Get the actual content height
      const tileHeight = Math.max(
        tile.offsetHeight || 0,
        tile.scrollHeight || 0,
        200 // Minimum height
      );
      
      // Restore original styles
      tile.style.position = originalPosition;
      tile.style.left = originalLeft;
      tile.style.top = originalTop;
      tile.style.width = originalWidth;
      tile.style.transform = originalTransform;
      tile.style.visibility = originalVisibility;

      newPositions.push({
        index,
        left: x,
        top: y,
        width: columnWidth,
        height: tileHeight
      });

      // Update column height
      columnHeights[minCol] += tileHeight + gap;
    });

    setPositions(newPositions);
    const maxHeight = Math.max(...columnHeights);
    setContainerHeight(maxHeight > 0 ? maxHeight - gap : 0); // Remove last gap
  }, [columns, gap]);

  // Recalculate on container resize
  const containerSize = useResizeObserver(containerRef);

  // Set up ResizeObserver for each tile to detect content changes
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const tiles = Array.from(container.children).filter(
      child => child.nodeType === Node.ELEMENT_NODE
    );

    // Clean up previous observers
    resizeObserversRef.current.forEach(observer => observer.disconnect());
    resizeObserversRef.current = [];

    // Create ResizeObserver for each tile
    tiles.forEach((tile) => {
      const observer = new ResizeObserver(() => {
        // Debounce recalculations
        clearTimeout(window.masonryRecalcTimeout);
        window.masonryRecalcTimeout = setTimeout(() => {
          calculateLayout();
        }, 100);
      });
      
      observer.observe(tile);
      resizeObserversRef.current.push(observer);
    });

    return () => {
      resizeObserversRef.current.forEach(observer => observer.disconnect());
      resizeObserversRef.current = [];
    };
  }, [children, calculateLayout]);

  // Set up MutationObserver to watch for DOM changes
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new MutationObserver(() => {
      // Debounce recalculations
      clearTimeout(window.masonryMutationTimeout);
      window.masonryMutationTimeout = setTimeout(() => {
        calculateLayout();
      }, 150);
    });

    observer.observe(containerRef.current, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['style', 'class']
    });

    mutationObserverRef.current = observer;

    return () => {
      if (mutationObserverRef.current) {
        mutationObserverRef.current.disconnect();
      }
    };
  }, [calculateLayout]);

  // Initial layout calculation and recalculation on dependencies
  useEffect(() => {
    // Wait for tiles to render, then calculate layout
    const timer = setTimeout(() => {
      calculateLayout();
    }, 50); // Slightly longer delay to ensure content is loaded
    
    return () => clearTimeout(timer);
  }, [children, columns, gap, containerSize, calculateLayout]);

  // Apply positions to tiles
  useEffect(() => {
    if (!containerRef.current || positions.length === 0) return;

    const tiles = Array.from(containerRef.current.children);
    tiles.forEach((tile, index) => {
      const position = positions[index];
      if (position) {
        // Use requestAnimationFrame for smooth updates
        requestAnimationFrame(() => {
          tile.style.position = 'absolute';
          tile.style.left = `${position.left}px`;
          tile.style.top = `${position.top}px`;
          tile.style.width = `${position.width}px`;
          tile.style.minHeight = `${position.height}px`; // Ensure minimum height
          tile.style.transition = 'top 0.3s ease, left 0.3s ease, width 0.3s ease';
          tile.style.visibility = 'visible'; // Make visible after positioning
        });
      }
    });
  }, [positions]);

  return (
    <div
      ref={containerRef}
      className={`relative ${className}`}
      style={{
        height: containerHeight > 0 ? `${containerHeight}px` : 'auto',
        minHeight: '200px',
        width: '100%'
      }}
    >
      {children}
    </div>
  );
};


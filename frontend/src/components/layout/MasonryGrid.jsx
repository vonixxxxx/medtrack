import { useEffect, useRef, useState } from 'react';
import { useResizeObserver } from '../hooks/useResizeObserver';

/**
 * MasonryGrid Component
 * Creates a true masonry layout where tiles flow in columns
 * with consistent vertical and horizontal margins
 */
export const MasonryGrid = ({ children, columns = 2, gap = 24, className = '' }) => {
  const containerRef = useRef(null);
  const [positions, setPositions] = useState([]);
  const [containerHeight, setContainerHeight] = useState(0);

  // Calculate masonry layout
  const calculateLayout = () => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const tiles = Array.from(container.children).filter(
      child => child.nodeType === Node.ELEMENT_NODE
    );
    
    if (tiles.length === 0) return;

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

      // Get tile height - temporarily reset positioning to measure
      const originalPosition = tile.style.position;
      const originalLeft = tile.style.left;
      const originalTop = tile.style.top;
      const originalWidth = tile.style.width;
      
      tile.style.position = 'static';
      tile.style.left = 'auto';
      tile.style.top = 'auto';
      tile.style.width = `${columnWidth}px`;
      
      // Force reflow to get accurate height
      void tile.offsetHeight;
      
      const tileHeight = tile.offsetHeight || tile.scrollHeight || 200;
      
      // Restore original styles
      tile.style.position = originalPosition;
      tile.style.left = originalLeft;
      tile.style.top = originalTop;
      tile.style.width = originalWidth;

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
    setContainerHeight(Math.max(...columnHeights) - gap); // Remove last gap
  };

  // Recalculate on resize
  const containerSize = useResizeObserver(containerRef);

  useEffect(() => {
    // Wait for tiles to render, then calculate layout
    const timer = setTimeout(() => {
      calculateLayout();
    }, 0);
    
    return () => clearTimeout(timer);
  }, [children, columns, gap, containerSize]);

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
          tile.style.transition = 'top 0.3s ease, left 0.3s ease, width 0.3s ease';
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
        minHeight: '100px'
      }}
    >
      {children}
    </div>
  );
};


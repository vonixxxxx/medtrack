"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

interface ShaderAnimationProps {
  primaryColor?: string;
  secondaryColor?: string;
  accentColor?: string;
  speed?: number;
  className?: string;
}

// Convert hex color to normalized RGB [0, 1]
function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [
        parseInt(result[1], 16) / 255,
        parseInt(result[2], 16) / 255,
        parseInt(result[3], 16) / 255,
      ]
    : [0, 0, 0];
}

export default function ShaderAnimation({
  primaryColor = "#0284c7", // blue-600
  secondaryColor = "#0369a1", // blue-700
  accentColor = "#0ea5e9", // blue-500
  speed = 1.0,
  className = "",
}: ShaderAnimationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const renderer = new THREE.WebGLRenderer({ 
      alpha: true, 
      antialias: true,
      powerPreference: "high-performance"
    });
    
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Convert colors to RGB
    const primaryRgb = hexToRgb(primaryColor);
    const secondaryRgb = hexToRgb(secondaryColor);
    const accentRgb = hexToRgb(accentColor);

    // Fragment shader with brand colors
    const fragmentShader = `
      uniform float uTime;
      uniform vec2 uResolution;
      uniform vec3 uPrimaryColor;
      uniform vec3 uSecondaryColor;
      uniform vec3 uAccentColor;
      uniform float uSpeed;

      // Smooth noise function
      float noise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
      }

      // Smooth interpolation
      float smoothNoise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        
        float a = noise(i);
        float b = noise(i + vec2(1.0, 0.0));
        float c = noise(i + vec2(0.0, 1.0));
        float d = noise(i + vec2(1.0, 1.0));
        
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
      }

      // Fractal noise
      float fbm(vec2 p) {
        float value = 0.0;
        float amplitude = 0.5;
        float frequency = 0.0;
        
        for (int i = 0; i < 4; i++) {
          value += amplitude * smoothNoise(p * frequency);
          frequency *= 2.0;
          amplitude *= 0.5;
        }
        
        return value;
      }

      void main() {
        vec2 uv = gl_FragCoord.xy / uResolution.xy;
        vec2 p = uv * 8.0;
        
        // Animated time
        float time = uTime * uSpeed * 0.3;
        
        // Create flowing patterns
        vec2 flow = vec2(
          fbm(p + vec2(time * 0.5, time * 0.3)),
          fbm(p + vec2(time * 0.3, time * 0.5))
        );
        
        // Combine multiple noise layers
        float n1 = fbm(p * 2.0 + flow + time * 0.2);
        float n2 = fbm(p * 1.5 - flow + time * 0.15);
        float n3 = fbm(p * 0.8 + flow * 0.5 + time * 0.1);
        
        // Create gradient based on position and noise
        float gradient = mix(
          mix(n1, n2, 0.5),
          n3,
          0.3
        );
        
        // Create radial gradient from center
        vec2 center = vec2(0.5, 0.5);
        float dist = distance(uv, center);
        float radial = 1.0 - smoothstep(0.0, 0.7, dist);
        
        // Combine patterns
        float pattern = mix(gradient, radial, 0.4);
        
        // Color mixing based on pattern
        vec3 color1 = mix(uPrimaryColor, uSecondaryColor, pattern * 0.6);
        vec3 color2 = mix(color1, uAccentColor, pattern * 0.4);
        
        // Add subtle brightness variation
        float brightness = 0.85 + pattern * 0.15;
        vec3 finalColor = color2 * brightness;
        
        // Soft edges
        float alpha = 1.0;
        float edge = smoothstep(0.0, 0.1, min(uv.x, min(uv.y, min(1.0 - uv.x, 1.0 - uv.y))));
        alpha *= edge;
        
        gl_FragColor = vec4(finalColor, alpha * 0.4);
      }
    `;

    // Vertex shader
    const vertexShader = `
      void main() {
        gl_Position = vec4(position, 1.0);
      }
    `;

    // Create shader material
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(width, height) },
        uPrimaryColor: { value: new THREE.Vector3(...primaryRgb) },
        uSecondaryColor: { value: new THREE.Vector3(...secondaryRgb) },
        uAccentColor: { value: new THREE.Vector3(...accentRgb) },
        uSpeed: { value: speed },
      },
      transparent: true,
    });

    // Create plane geometry
    const geometry = new THREE.PlaneGeometry(2, 2);
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    // Animation loop
    let time = 0;
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      time += 0.016; // ~60fps
      material.uniforms.uTime.value = time;
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      
      renderer.setSize(newWidth, newHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      material.uniforms.uResolution.value.set(newWidth, newHeight);
    };

    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      renderer.dispose();
      material.dispose();
      geometry.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [primaryColor, secondaryColor, accentColor, speed]);

  return (
    <div
      ref={containerRef}
      className={`absolute inset-0 w-full h-full ${className}`}
      style={{ zIndex: 0 }}
    />
  );
}




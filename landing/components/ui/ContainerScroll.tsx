"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import Image from "next/image";
import { cn } from "@/lib/utils";

interface ContainerScrollProps {
  titleComponent: React.ReactNode;
  children?: React.ReactNode;
  imageSrc?: string;
  imageAlt?: string;
}

export function ContainerScroll({
  titleComponent,
  children,
  imageSrc,
  imageAlt = "Hero image",
}: ContainerScrollProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
  });

  const rotateX = useTransform(scrollYProgress, [0, 1], [30, 0]);
  const rotateY = useTransform(scrollYProgress, [0, 1], [0, 0]);
  const rotate = useTransform(scrollYProgress, [0, 1], [20, 0]);
  const scale = useTransform(scrollYProgress, [0, 1], [0.8, 1]);
  const translate = useTransform(scrollYProgress, [0, 1], [0, -100]);

  return (
    <div
      className="h-[80rem] md:h-[100rem] flex items-center justify-center relative p-2 md:p-20"
      ref={containerRef}
    >
      <div
        className="py-10 md:py-40 w-full relative"
        style={{
          perspective: "1000px",
        }}
      >
        <motion.div
          style={{
            rotateX,
            rotateY,
            rotate,
            scale,
            translateY: translate,
          }}
          className="flex flex-col items-center justify-center"
        >
          <div className="w-full">{titleComponent}</div>
          {imageSrc && (
            <motion.div
              className="mt-10 md:mt-20 relative w-full max-w-6xl mx-auto"
              style={{
                rotateX,
                rotateY,
                rotate,
                scale,
              }}
            >
              <div className="relative w-full aspect-video rounded-2xl overflow-hidden shadow-2xl border border-gray-200">
                <Image
                  src={imageSrc}
                  alt={imageAlt}
                  fill
                  className="object-cover"
                  priority
                />
              </div>
            </motion.div>
          )}
          {children && (
            <motion.div
              className="mt-10 md:mt-20 w-full"
              style={{
                rotateX,
                rotateY,
                rotate,
                scale,
              }}
            >
              {children}
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  );
}


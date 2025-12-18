"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { useEffect } from "react";
import Image from "next/image";

interface MockupModalProps {
  featureId: string | null;
  feature?: {
    id: string;
    title: string;
    desc: string;
    color: string;
  } | null;
  onClose: () => void;
}

export default function MockupModal({ featureId, feature, onClose }: MockupModalProps) {
  useEffect(() => {
    if (featureId) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [featureId]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [onClose]);

  if (!featureId || !feature) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          transition={{ type: "spring", damping: 25, stiffness: 300 }}
          className="relative bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div>
              <h2 id="modal-title" className="text-2xl font-bold text-gray-900">
                {feature.title}
              </h2>
              <p className="text-gray-600 mt-1">{feature.desc}</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-600"
              aria-label="Close modal"
            >
              <X className="w-6 h-6 text-gray-600" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
            <div className="relative w-full aspect-video bg-gradient-to-br from-gray-100 to-gray-200 rounded-xl overflow-hidden mb-4">
              {`/mockups/feature-${featureId}.svg`.endsWith('.svg') ? (
                <img
                  src={`/mockups/feature-${featureId}.svg`}
                  alt={`${feature.title} mockup`}
                  className="w-full h-full object-contain"
                />
              ) : (
                <Image
                  src={`/mockups/feature-${featureId}.png`}
                  alt={`${feature.title} mockup`}
                  fill
                  className="object-contain"
                  sizes="(max-width: 768px) 100vw, 80vw"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.style.display = "none";
                  }}
                />
              )}
              {/* Fallback placeholder */}
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-tr from-gray-200 to-gray-300">
                <div className="text-center">
                  <div className={`w-16 h-16 mx-auto mb-4 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center`}>
                    <span className="text-white text-2xl">📱</span>
                  </div>
                  <p className="text-gray-600">Mockup coming soon</p>
                </div>
              </div>
            </div>

            {/* Feature details */}
            <div className="prose max-w-none">
              <h3 className="text-xl font-semibold mb-2">Key Features</h3>
              <ul className="list-disc list-inside space-y-2 text-gray-600">
                <li>Enterprise-grade security and compliance</li>
                <li>Real-time data synchronization</li>
                <li>Intuitive user interface</li>
                <li>Comprehensive analytics and reporting</li>
              </ul>
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-4 p-6 border-t border-gray-200">
            <button
              onClick={onClose}
              className="px-6 py-2 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors font-medium"
            >
              Close
            </button>
            <a
              href={`/features#${featureId}`}
              className="px-6 py-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700 transition-colors font-medium"
            >
              Learn more
            </a>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Activity, Plus, Calendar, Target } from 'lucide-react';

export default function AddMetricPopup({ isOpen, onClose, onMetricAdded }) {
  const [metricName, setMetricName] = useState('');
  const [metricValue, setMetricValue] = useState('');
  const [metricUnit, setMetricUnit] = useState('');
  const [frequency, setFrequency] = useState(1);
  const [targetValue, setTargetValue] = useState('');
  const [notes, setNotes] = useState('');

  const commonMetrics = [
    { name: 'Blood Pressure', unit: 'mmHg', example: '120/80' },
    { name: 'Blood Sugar', unit: 'mg/dL', example: '120' },
    { name: 'Weight', unit: 'kg', example: '70' },
    { name: 'Heart Rate', unit: 'bpm', example: '72' },
    { name: 'Temperature', unit: '°C', example: '36.8' },
    { name: 'Cholesterol', unit: 'mg/dL', example: '180' },
    { name: 'BMI', unit: 'kg/m²', example: '24.5' },
    { name: 'Steps', unit: 'steps', example: '8000' }
  ];

  const frequencyOptions = [
    { value: 1, label: 'Daily' },
    { value: 2, label: 'Every 2 days' },
    { value: 7, label: 'Weekly' },
    { value: 14, label: 'Every 2 weeks' },
    { value: 30, label: 'Monthly' }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!metricName.trim() || !metricValue.trim()) {
      return;
    }

    const newMetric = {
      id: Date.now(),
      name: metricName.trim(),
      value: metricValue.trim(),
      unit: metricUnit.trim(),
      frequency: parseInt(frequency),
      targetValue: targetValue.trim() || null,
      notes: notes.trim() || null,
      lastUpdated: new Date().toISOString(),
      history: [{
        value: metricValue.trim(),
        timestamp: new Date().toISOString(),
        notes: notes.trim() || null
      }]
    };

    onMetricAdded(newMetric);
    
    // Reset form
    setMetricName('');
    setMetricValue('');
    setMetricUnit('');
    setFrequency(1);
    setTargetValue('');
    setNotes('');
  };

  const handleMetricSelect = (metric) => {
    setMetricName(metric.name);
    setMetricUnit(metric.unit);
    setMetricValue('');
    setTargetValue('');
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className="bg-white rounded-2xl p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-gray-800">
                Add Health Metric
              </h2>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 bg-gray-100 hover:bg-gray-200 rounded-full flex items-center justify-center transition-colors"
            >
              <X className="w-4 h-4 text-gray-600" />
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Common Metrics Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Quick Select Common Metrics
              </label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {commonMetrics.map((metric, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => handleMetricSelect(metric)}
                    className="p-3 border border-gray-200 rounded-xl hover:border-blue-300 hover:bg-blue-50 transition-colors text-left"
                  >
                    <div className="font-medium text-gray-800 text-sm">{metric.name}</div>
                    <div className="text-xs text-gray-500">{metric.unit}</div>
                    <div className="text-xs text-gray-400">e.g., {metric.example}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Custom Metric Input */}
            <div>
              <label htmlFor="metricName" className="block text-sm font-medium text-gray-700 mb-2">
                Metric Name
              </label>
              <input
                id="metricName"
                type="text"
                value={metricName}
                onChange={(e) => setMetricName(e.target.value)}
                placeholder="e.g., Blood Pressure, Weight, Blood Sugar"
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="metricValue" className="block text-sm font-medium text-gray-700 mb-2">
                  Current Value
                </label>
                <input
                  id="metricValue"
                  type="text"
                  value={metricValue}
                  onChange={(e) => setMetricValue(e.target.value)}
                  placeholder="Enter the measurement value"
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label htmlFor="metricUnit" className="block text-sm font-medium text-gray-700 mb-2">
                  Unit
                </label>
                <input
                  id="metricUnit"
                  type="text"
                  value={metricUnit}
                  onChange={(e) => setMetricUnit(e.target.value)}
                  placeholder="e.g., mg/dL, kg, mmHg"
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="frequency" className="block text-sm font-medium text-gray-700 mb-2">
                  Reminder Frequency
                </label>
                <select
                  id="frequency"
                  value={frequency}
                  onChange={(e) => setFrequency(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {frequencyOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="targetValue" className="block text-sm font-medium text-gray-700 mb-2">
                  Target Value (Optional)
                </label>
                <input
                  id="targetValue"
                  type="text"
                  value={targetValue}
                  onChange={(e) => setTargetValue(e.target.value)}
                  placeholder="e.g., 120/80, 70 kg"
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            <div>
              <label htmlFor="notes" className="block text-sm font-medium text-gray-700 mb-2">
                Notes (Optional)
              </label>
              <textarea
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Any additional notes about this measurement..."
                rows={3}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="flex-1 py-3 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 transition-colors font-medium"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="flex-1 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all flex items-center justify-center font-medium shadow-lg transform hover:scale-[1.02] active:scale-[0.98]"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Metric
              </button>
            </div>
          </form>

          {/* Instructions */}
          <div className="mt-6 bg-blue-50 border border-blue-200 rounded-xl p-4">
            <h4 className="font-medium text-blue-800 mb-2 flex items-center">
              <Target className="w-4 h-4 mr-2" />
              How to use:
            </h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Quick select common metrics or enter a custom one</li>
              <li>• Set your current measurement value and unit</li>
              <li>• Choose how often you want to be reminded to log</li>
              <li>• Optionally set a target value to track progress</li>
              <li>• Add notes for context or special circumstances</li>
            </ul>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

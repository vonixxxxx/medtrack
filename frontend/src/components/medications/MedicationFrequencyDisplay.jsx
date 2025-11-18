 import { Pill, Clock } from 'lucide-react';

/**
 * Medication Frequency Display Component
 * Based on Confir-Med medication display format
 * Shows medication name and frequency in hours
 */

export const MedicationFrequencyDisplay = ({ medication }) => {
  const formatFrequency = (frequency) => {
    if (!frequency) return 'Not specified';
    
    // Convert frequency to hours
    const frequencyMap = {
      'daily': 24,
      'twice_daily': 12,
      'three_times_daily': 8,
      'four_times_daily': 6,
      'weekly': 168,
      'monthly': 720,
      'as_needed': null
    };
    
    const hours = frequencyMap[frequency];
    if (hours) {
      return `Every ${hours} hours`;
    }
    
    return frequency;
  };

  return (
    <div className="flex items-center gap-3 p-3 bg-white rounded-xl border border-neutral-200 hover:border-primary-300 transition-all">
      <div className="p-2 bg-primary-50 rounded-lg">
        <Pill className="w-5 h-5 text-primary-600" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-neutral-900 text-base">
          {medication.name || medication.medication_name || medication.generic_name}
        </p>
        <div className="flex items-center gap-1.5 mt-1">
          <Clock className="w-4 h-4 text-neutral-500" />
          <p className="text-sm text-neutral-600">
            {formatFrequency(medication.frequency)}
          </p>
        </div>
      </div>
    </div>
  );
};




import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import api from '../../api';

interface HbA1cAdjustmentModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface Medication {
  name: string;
  dose: number;
}

interface CalculationResult {
  MES: number;
  adjustedHbA1cPercent: number;
  adjustedHbA1cMmolMol: number;
}

export const HbA1cAdjustmentModal = ({ isOpen, onClose }: HbA1cAdjustmentModalProps) => {
  const [measuredHbA1c, setMeasuredHbA1c] = useState('');
  const [weight, setWeight] = useState('');
  const [medications, setMedications] = useState<Medication[]>([
    { name: 'metformin', dose: 0 },
    { name: 'insulin', dose: 0 },
    { name: 'glimepiride', dose: 0 },
    { name: 'glipizide', dose: 0 },
    { name: 'glyburide', dose: 0 },
    { name: 'pioglitazone', dose: 0 },
    { name: 'sitagliptin', dose: 0 },
    { name: 'saxagliptin', dose: 0 },
    { name: 'linagliptin', dose: 0 },
    { name: 'liraglutide', dose: 0 },
    { name: 'exenatide_bid', dose: 0 },
    { name: 'exenatide_qw', dose: 0 },
    { name: 'dulaglutide', dose: 0 },
    { name: 'semaglutide', dose: 0 },
    { name: 'dapagliflozin', dose: 0 },
    { name: 'canagliflozin', dose: 0 },
    { name: 'empagliflozin', dose: 0 }
  ]);
  const [result, setResult] = useState<CalculationResult | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [error, setError] = useState('');

  const handleMedicationChange = (index: number, field: 'name' | 'dose', value: string | number) => {
    const updated = [...medications];
    updated[index] = { ...updated[index], [field]: value };
    setMedications(updated);
  };

  const handleCalculate = async () => {
    if (!measuredHbA1c || !weight) {
      setError('Please enter both HbA1c and weight values');
      return;
    }

    const measuredValue = parseFloat(measuredHbA1c);
    const weightValue = parseFloat(weight);

    if (isNaN(measuredValue) || isNaN(weightValue)) {
      setError('Please enter valid numeric values');
      return;
    }

    setIsCalculating(true);
    setError('');

    try {
      // Convert medications array to object format expected by backend
      const medicationsObj = medications.reduce((acc, med) => {
        if (med.dose > 0) {
          acc[med.name] = med.dose;
        }
        return acc;
      }, {} as Record<string, number>);

      const { data } = await api.post('doctor/hba1c-adjust', {
        measuredHbA1cPercent: measuredValue,
        weightKg: weightValue,
        medications: medicationsObj
      });

      setResult(data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Calculation failed');
    } finally {
      setIsCalculating(false);
    }
  };

  const handleReset = () => {
    setMeasuredHbA1c('');
    setWeight('');
    setMedications(medications.map(med => ({ ...med, dose: 0 })));
    setResult(null);
    setError('');
  };

  const getMedicationDisplayName = (name: string) => {
    const displayNames: Record<string, string> = {
      'metformin': 'Metformin',
      'insulin': 'Insulin',
      'glimepiride': 'Glimepiride',
      'glipizide': 'Glipizide',
      'glyburide': 'Glyburide',
      'pioglitazone': 'Pioglitazone',
      'sitagliptin': 'Sitagliptin',
      'saxagliptin': 'Saxagliptin',
      'linagliptin': 'Linagliptin',
      'liraglutide': 'Liraglutide',
      'exenatide_bid': 'Exenatide (BID)',
      'exenatide_qw': 'Exenatide (QW)',
      'dulaglutide': 'Dulaglutide',
      'semaglutide': 'Semaglutide',
      'dapagliflozin': 'Dapagliflozin',
      'canagliflozin': 'Canagliflozin',
      'empagliflozin': 'Empagliflozin'
    };
    return displayNames[name] || name;
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-gray-900 rounded-3xl border border-gray-800 p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-white">HbA1c Adjustment Calculator</h2>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Section */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium text-white">Input Values</h3>
                
                {/* Basic Inputs */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Measured HbA1c (%)
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={measuredHbA1c}
                      onChange={(e) => setMeasuredHbA1c(e.target.value)}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
                      placeholder="7.5"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Weight (kg)
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={weight}
                      onChange={(e) => setWeight(e.target.value)}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-xl text-white focus:ring-2 focus:ring-white focus:border-white"
                      placeholder="70"
                    />
                  </div>
                </div>

                {/* Medications */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-3">
                    Medication Doses
                  </label>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {medications.map((med, index) => (
                      <div key={med.name} className="flex items-center gap-3">
                        <div className="flex-1 min-w-0">
                          <span className="text-sm text-gray-300 truncate">
                            {getMedicationDisplayName(med.name)}
                          </span>
                        </div>
                        <input
                          type="number"
                          step="0.1"
                          value={med.dose}
                          onChange={(e) => handleMedicationChange(index, 'dose', parseFloat(e.target.value) || 0)}
                          className="w-20 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-white text-sm focus:ring-1 focus:ring-white focus:border-white"
                          placeholder="0"
                        />
                        <span className="text-xs text-gray-400 w-8">
                          mg
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3">
                  <button
                    onClick={handleCalculate}
                    disabled={isCalculating || !measuredHbA1c || !weight}
                    className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-xl hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isCalculating ? 'Calculating...' : 'Calculate Adjustment'}
                  </button>
                  
                  <button
                    onClick={handleReset}
                    className="px-4 py-2 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition-colors"
                  >
                    Reset
                  </button>
                </div>

                {error && (
                  <div className="p-3 bg-red-900/20 border border-red-800 rounded-xl text-red-400 text-sm">
                    {error}
                  </div>
                )}
              </div>

              {/* Results Section */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium text-white">Results</h3>
                
                {result ? (
                  <div className="space-y-4">
                    <div className="bg-gray-800 rounded-xl p-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-400">
                            {result.MES.toFixed(2)}
                          </div>
                          <div className="text-sm text-gray-400">MES Score</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-400">
                            {result.adjustedHbA1cPercent.toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-400">Adjusted HbA1c (%)</div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-800 rounded-xl p-4">
                      <h4 className="text-md font-medium text-white mb-3">Detailed Results</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Measured HbA1c:</span>
                          <span className="text-white">{measuredHbA1c}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">MES Adjustment:</span>
                          <span className="text-white">+{result.MES.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Adjusted HbA1c (%):</span>
                          <span className="text-white font-medium">{result.adjustedHbA1cPercent.toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Adjusted HbA1c (mmol/mol):</span>
                          <span className="text-white font-medium">{result.adjustedHbA1cMmolMol.toFixed(1)} mmol/mol</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-blue-900/20 border border-blue-800 rounded-xl p-4">
                      <h4 className="text-sm font-medium text-blue-300 mb-2">Clinical Interpretation</h4>
                      <div className="text-xs text-blue-200 space-y-1">
                        <p>• MES (Medication Effect Score) quantifies the impact of diabetes medications</p>
                        <p>• Higher MES values indicate greater medication effect on HbA1c</p>
                        <p>• Adjusted HbA1c provides a more accurate assessment of glycemic control</p>
                        <p>• Use adjusted values for treatment decisions and patient counseling</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-800 rounded-xl p-8 text-center">
                    <div className="text-gray-400 mb-2">
                      <svg className="w-12 h-12 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <p className="text-gray-400">Enter values and click "Calculate Adjustment" to see results</p>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};



import { useState } from 'react';
import { motion } from 'framer-motion';
import api from '../../api';

export const MedicalHistoryParser = () => {
  const [medicalNotes, setMedicalNotes] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [extractedConditions, setExtractedConditions] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleParseHistory = async () => {
    if (!medicalNotes.trim()) {
      setError('Please enter medical notes to parse');
      return;
    }

    setIsProcessing(true);
    setError('');
    setSuccess('');

    try {
      const { data } = await api.post('doctor/parse-history', {
        medicalNotes: medicalNotes.trim()
      });

      setExtractedConditions(data.conditions);
      setSuccess(`Successfully extracted ${data.conditions.length} conditions`);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to parse medical history');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    setMedicalNotes('');
    setExtractedConditions([]);
    setError('');
    setSuccess('');
  };

  const handleAddToPatient = async (patientId: number) => {
    if (extractedConditions.length === 0) {
      setError('No conditions to add');
      return;
    }

    try {
      await api.post(`doctor/patients/${patientId}/conditions`, {
        conditions: extractedConditions
      });
      setSuccess(`Added ${extractedConditions.length} conditions to patient`);
      setExtractedConditions([]);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to add conditions to patient');
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900 rounded-3xl border border-gray-800 p-6"
    >
      <h3 className="text-lg font-semibold text-white mb-4">Import Medical History</h3>
      
      <div className="space-y-4">
        {/* Medical Notes Input */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Paste Medical Notes
          </label>
          <textarea
            value={medicalNotes}
            onChange={(e) => setMedicalNotes(e.target.value)}
            placeholder="Paste patient medical history, discharge notes, or clinical documentation here..."
            className="w-full h-32 px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:ring-2 focus:ring-white focus:border-white resize-none"
          />
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleParseHistory}
            disabled={isProcessing || !medicalNotes.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Processing...
              </div>
            ) : (
              'Validate & Sort with AI'
            )}
          </button>
          
          <button
            onClick={handleClear}
            className="px-4 py-2 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition-colors"
          >
            Clear
          </button>
        </div>

        {/* Status Messages */}
        {error && (
          <div className="p-3 bg-red-900/20 border border-red-800 rounded-xl text-red-400 text-sm">
            {error}
          </div>
        )}

        {success && (
          <div className="p-3 bg-green-900/20 border border-green-800 rounded-xl text-green-400 text-sm">
            {success}
          </div>
        )}

        {/* Extracted Conditions */}
        {extractedConditions.length > 0 && (
          <div className="mt-6">
            <h4 className="text-md font-medium text-white mb-3">
              Extracted Conditions ({extractedConditions.length})
            </h4>
            <div className="flex flex-wrap gap-2 mb-4">
              {extractedConditions.map((condition, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-blue-600 text-white text-sm rounded-full"
                >
                  {condition}
                </span>
              ))}
            </div>
            
            <div className="text-sm text-gray-400">
              <p>These conditions have been normalized and are ready to be added to a patient's record.</p>
              <p className="mt-1">
                <strong>Note:</strong> Select a patient from the records table to add these conditions.
              </p>
            </div>
          </div>
        )}

        {/* AI Processing Info */}
        <div className="mt-6 p-4 bg-gray-800 rounded-xl">
          <h4 className="text-sm font-medium text-white mb-2">AI Processing Information</h4>
          <div className="text-xs text-gray-400 space-y-1">
            <p>• Uses BioGPT to extract medical conditions from unstructured text</p>
            <p>• Normalizes condition names (e.g., "T2DM" → "Type 2 Diabetes Mellitus")</p>
            <p>• Removes duplicates and ensures proper medical terminology</p>
            <p>• Automatically categorizes conditions by medical specialty</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};



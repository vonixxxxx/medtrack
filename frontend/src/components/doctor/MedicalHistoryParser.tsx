import { useState } from 'react';
import { motion } from 'framer-motion';
import api from '../../api';

interface MedicalHistoryParserProps {
  selectedPatientId?: string;
  onConditionsAdded?: () => void;
}

export const MedicalHistoryParser = ({ selectedPatientId, onConditionsAdded }: MedicalHistoryParserProps) => {
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

    if (!selectedPatientId) {
      setError('Please select a patient first');
      return;
    }

    setIsProcessing(true);
    setError('');
    setSuccess('');

    try {
      // Use longer timeout for AI parsing (60 seconds)
      const { data } = await api.post('doctor/parse-history', {
        patientId: selectedPatientId,
        medicalNotes: medicalNotes.trim()
      }, {
        timeout: 60000 // 60 seconds for AI processing
      });

      if (data.success && data.parsedData) {
        const conditions = data.conditions || [];
        setExtractedConditions(conditions.map((c: any) => c.normalized || c.name || c));
        const updateCount = Object.keys(data.updates || {}).length;
        setSuccess(`✅ Successfully imported medical history! Extracted ${Object.keys(data.parsedData).length} data points and ${conditions.length} conditions. ${updateCount} fields updated in patient record.`);
        
        // Trigger refresh of patient data if callback is provided
        if (onConditionsAdded) {
          setTimeout(() => {
            onConditionsAdded();
          }, 500);
        }
      } else {
        setError(data.error || data.details || 'Failed to parse medical history');
      }
    } catch (err: any) {
      let errorMessage = err.response?.data?.details || err.response?.data?.error || err.message || 'Failed to parse medical history';
      
      // Handle timeout errors specifically
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        errorMessage = 'Request timed out. The AI model may be processing a large text. Please try with shorter notes or wait a moment and try again.';
      }
      
      const errorDetails = err.response?.data?.stack ? `\n\nDetails: ${err.response.data.stack.substring(0, 200)}...` : '';
      setError(`${errorMessage}${errorDetails}`);
      console.error('Parser error:', err.response?.data || err);
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
      whileHover={{ y: -2 }}
      className="bg-gradient-to-br from-white to-blue-50/30 rounded-2xl border border-blue-100 hover:border-blue-200 shadow-lg shadow-blue-600/5 hover:shadow-xl hover:shadow-blue-600/20 transition-all p-6"
    >
      <h3 className="text-xl font-bold text-gray-900 mb-4">Import Medical History</h3>
      
      <div className="space-y-4">
        {/* Medical Notes Input */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Paste Medical Notes
          </label>
          <textarea
            value={medicalNotes}
            onChange={(e) => setMedicalNotes(e.target.value)}
            placeholder="Paste patient medical history, discharge notes, or clinical documentation here..."
            className="w-full h-32 px-4 py-3 bg-white border-2 border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all hover:border-blue-300 resize-none"
          />
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <motion.button
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleParseHistory}
            disabled={isProcessing || !medicalNotes.trim()}
            className="px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-600/25 hover:shadow-xl hover:shadow-blue-600/40 font-semibold text-sm disabled:hover:shadow-lg disabled:hover:shadow-blue-600/25"
          >
            {isProcessing ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Processing...
              </div>
            ) : (
              'Validate & Sort with AI'
            )}
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleClear}
            className="px-4 py-2.5 bg-gray-100 text-gray-700 rounded-xl hover:bg-gray-200 transition-all border border-gray-200 font-medium text-sm"
          >
            Clear
          </motion.button>
        </div>

        {/* Status Messages */}
        {error && (
          <div className="p-3 bg-red-50 border-2 border-red-200 rounded-xl text-red-700 text-sm font-medium">
            {error}
          </div>
        )}

        {success && (
          <div className="p-3 bg-green-50 border-2 border-green-200 rounded-xl text-green-700 text-sm font-medium">
            {success}
          </div>
        )}

        {/* Extracted Conditions */}
        {extractedConditions.length > 0 && (
          <div className="mt-6">
            <h4 className="text-lg font-bold text-gray-900 mb-3">
              Extracted Conditions ({extractedConditions.length})
            </h4>
            <div className="flex flex-wrap gap-2 mb-4">
              {extractedConditions.map((condition, index) => (
                <span
                  key={index}
                  className="px-3 py-1.5 bg-blue-600 text-white text-sm rounded-xl font-medium shadow-sm"
                >
                  {condition}
                </span>
              ))}
            </div>
            
            <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-xl border border-blue-100">
              <p className="font-medium mb-1">These conditions have been normalized and are ready to be added to a patient's record.</p>
              <p>
                <strong className="text-gray-900">Note:</strong> Select a patient from the records table to add these conditions.
              </p>
            </div>
          </div>
        )}

        {/* AI Processing Info */}
        <div className="mt-6 p-4 bg-blue-50 rounded-xl border border-blue-100">
          <h4 className="text-sm font-bold text-gray-900 mb-2">AI Processing Information</h4>
          <div className="text-xs text-gray-600 space-y-1">
            <p>• Powered by Ollama (llama3.2:latest) for deterministic medical data extraction</p>
            <p>• Extracts structured data with 0/1/null boolean values (0=absent, 1=present, null=not mentioned)</p>
            <p>• Handles negations, borderline conditions, and "possible" diagnoses correctly</p>
            <p>• Maps conditions to Patient boolean columns for consistent storage</p>
            <p>• Never infers or guesses - only extracts explicitly stated information</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};



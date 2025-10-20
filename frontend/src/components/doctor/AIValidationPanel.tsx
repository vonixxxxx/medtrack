import React, { useState, useEffect } from 'react';
import { api } from '../../utils/api';

interface AuditLog {
  id: string;
  field_name: string;
  old_value: string | null;
  new_value: string;
  ai_confidence: number;
  ai_suggestion: string;
  clinician_approved: boolean;
  createdAt: string;
}

interface PatientMatch {
  id: string;
  name: string;
  email: string;
  nhsNumber: string;
  mrn: string;
  confidence: number;
}

interface AIValidationPanelProps {
  patientId?: string;
  onClose: () => void;
  onPatientSelected?: (patientId: string) => void;
}

export const AIValidationPanel: React.FC<AIValidationPanelProps> = ({ 
  patientId, 
  onClose, 
  onPatientSelected 
}) => {
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [patientMatches, setPatientMatches] = useState<PatientMatch[]>([]);
  const [extractedData, setExtractedData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(patientId || null);
  const [showPatientSelection, setShowPatientSelection] = useState(false);
  const [medicalNotes, setMedicalNotes] = useState('');
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    if (patientId) {
      fetchAuditLogs(patientId);
    }
  }, [patientId]);

  const fetchAuditLogs = async (pid: string) => {
    try {
      setLoading(true);
      const response = await api.get(`doctor/patients/${pid}/audit-logs`);
      setAuditLogs(response.data || []);
    } catch (error) {
      console.error('Error fetching audit logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const processMedicalNotes = async () => {
    if (!medicalNotes.trim()) return;

    try {
      setProcessing(true);
      const response = await api.post('doctor/intelligent-parse', {
        medicalNotes,
        hospitalCode: '123456789' // Default hospital code
      });

      if (response.data.action === 'select_patient') {
        setPatientMatches(response.data.patientMatches);
        setExtractedData(response.data.extractedData);
        setShowPatientSelection(true);
      } else {
        // Single match or new patient created
        setExtractedData(response.data.extractedData);
        if (response.data.patient?.id) {
          setSelectedPatientId(response.data.patient.id);
          await fetchAuditLogs(response.data.patient.id);
        }
      }
    } catch (error) {
      console.error('Error processing medical notes:', error);
      alert('Failed to process medical notes. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  const selectPatient = async (patientId: string) => {
    setSelectedPatientId(patientId);
    setShowPatientSelection(false);
    await fetchAuditLogs(patientId);
    onPatientSelected?.(patientId);
  };

  const approveChange = async (logId: string) => {
    try {
      await api.post(`doctor/audit-logs/${logId}/approve`);
      setAuditLogs(prev => 
        prev.map(log => 
          log.id === logId 
            ? { ...log, clinician_approved: true }
            : log
        )
      );
    } catch (error) {
      console.error('Error approving change:', error);
    }
  };

  const rejectChange = async (logId: string) => {
    try {
      await api.post(`doctor/audit-logs/${logId}/reject`);
      setAuditLogs(prev => prev.filter(log => log.id !== logId));
    } catch (error) {
      console.error('Error rejecting change:', error);
    }
  };

  const approveAllChanges = async () => {
    try {
      const pendingLogs = auditLogs.filter(log => !log.clinician_approved);
      await Promise.all(
        pendingLogs.map(log => api.post(`doctor/audit-logs/${log.id}/approve`))
      );
      setAuditLogs(prev => 
        prev.map(log => ({ ...log, clinician_approved: true }))
      );
    } catch (error) {
      console.error('Error approving all changes:', error);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-6xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">AI Data Validation</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl"
          >
            ×
          </button>
        </div>

        {/* Medical Notes Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Paste Medical Notes
          </label>
          <textarea
            value={medicalNotes}
            onChange={(e) => setMedicalNotes(e.target.value)}
            placeholder="Paste unstructured medical notes here for AI processing..."
            className="w-full h-32 bg-gray-700 text-white p-3 rounded border border-gray-600 resize-none"
          />
          <button
            onClick={processMedicalNotes}
            disabled={processing || !medicalNotes.trim()}
            className="mt-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-4 py-2 rounded"
          >
            {processing ? 'Processing...' : 'Process with AI'}
          </button>
        </div>

        {/* Patient Selection Modal */}
        {showPatientSelection && (
          <div className="mb-6 p-4 bg-gray-700 rounded-lg">
            <h3 className="text-lg font-semibold text-white mb-4">
              Select Patient ({patientMatches.length} matches found)
            </h3>
            <div className="space-y-2">
              {patientMatches.map((match) => (
                <div
                  key={match.id}
                  className="flex items-center justify-between p-3 bg-gray-600 rounded cursor-pointer hover:bg-gray-500"
                  onClick={() => selectPatient(match.id)}
                >
                  <div>
                    <p className="text-white font-medium">{match.name}</p>
                    <p className="text-gray-300 text-sm">
                      {match.email} • NHS: {match.nhsNumber} • MRN: {match.mrn}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className={`px-2 py-1 rounded text-xs ${
                      match.confidence >= 0.8 ? 'bg-green-900 text-green-300' :
                      match.confidence >= 0.6 ? 'bg-yellow-900 text-yellow-300' :
                      'bg-red-900 text-red-300'
                    }`}>
                      {Math.round(match.confidence * 100)}% match
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <button
              onClick={() => setShowPatientSelection(false)}
              className="mt-4 bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded"
            >
              Cancel
            </button>
          </div>
        )}

        {/* Extracted Data Preview */}
        {extractedData && (
          <div className="mb-6 p-4 bg-gray-700 rounded-lg">
            <h3 className="text-lg font-semibold text-white mb-4">AI Extracted Data</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-300 mb-2">Patient Information</h4>
                <p className="text-white">Name: {extractedData.Patient_Name}</p>
                <p className="text-white">Age: {extractedData.Age}</p>
                <p className="text-white">Sex: {extractedData.Sex}</p>
              </div>
              <div>
                <h4 className="font-medium text-gray-300 mb-2">Clinical Data</h4>
                <p className="text-white">Conditions: {extractedData.Conditions?.join(', ') || 'None'}</p>
                <p className="text-white">Medications: {extractedData.Medications?.length || 0} found</p>
                <p className="text-white">Lab Results: {extractedData.Labs?.length || 0} found</p>
              </div>
            </div>
          </div>
        )}

        {/* Audit Logs */}
        {selectedPatientId && (
          <div>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-white">
                Pending AI Suggestions ({auditLogs.filter(log => !log.clinician_approved).length})
              </h3>
              {auditLogs.some(log => !log.clinician_approved) && (
                <button
                  onClick={approveAllChanges}
                  className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded"
                >
                  Approve All
                </button>
              )}
            </div>

            {loading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
                <p className="text-gray-300 mt-2">Loading audit logs...</p>
              </div>
            ) : auditLogs.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-300">No pending AI suggestions for this patient.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {auditLogs.map((log) => (
                  <div
                    key={log.id}
                    className={`p-4 rounded-lg border ${
                      log.clinician_approved 
                        ? 'bg-green-900 border-green-700' 
                        : 'bg-gray-700 border-gray-600'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h4 className="font-medium text-white capitalize">
                          {log.field_name.replace(/_/g, ' ')}
                        </h4>
                        <p className="text-sm text-gray-300">
                          {log.ai_suggestion}
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`text-xs px-2 py-1 rounded ${
                          getConfidenceColor(log.ai_confidence)
                        } bg-gray-800`}>
                          {getConfidenceLabel(log.ai_confidence)} Confidence
                        </span>
                        {log.clinician_approved && (
                          <span className="text-xs px-2 py-1 rounded bg-green-800 text-green-300">
                            Approved
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                      <div>
                        <p className="text-sm text-gray-400">Current Value:</p>
                        <p className="text-white">{log.old_value || 'Not set'}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">AI Suggestion:</p>
                        <p className="text-white">{log.new_value}</p>
                      </div>
                    </div>

                    {!log.clinician_approved && (
                      <div className="flex space-x-2">
                        <button
                          onClick={() => approveChange(log.id)}
                          className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => rejectChange(log.id)}
                          className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm"
                        >
                          Reject
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

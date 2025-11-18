import React, { useState, useEffect } from 'react';
import api from '../../api';

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
  hospitalCode: string;
  onClose: () => void;
  onPatientSelected?: (patientId: string) => void;
}

export const AIValidationPanel: React.FC<AIValidationPanelProps> = ({ 
  patientId, 
  hospitalCode,
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
        hospitalCode: hospitalCode || '123456789' // Use provided hospital code
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
    } catch (error: any) {
      console.error('Error processing medical notes:', error);
      const errorMessage = error.response?.data?.error || 'Failed to process medical notes. Please try again.';
      // TODO: Replace with toast notification system
      alert(errorMessage);
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
    if (confidence >= 0.8) return 'text-medical-400';
    if (confidence >= 0.6) return 'text-warning-400';
    return 'text-error-400';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-neutral-800 rounded-lg p-6 w-full max-w-6xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">AI Data Validation</h2>
          <button
            onClick={onClose}
            className="text-neutral-400 hover:text-white text-2xl h-11 w-11 flex items-center justify-center rounded-lg hover:bg-neutral-700 transition-colors"
            aria-label="Close validation panel"
          >
            ×
          </button>
        </div>

        {/* Medical Notes Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-neutral-300 mb-2">
            Paste Medical Notes
          </label>
          <textarea
            value={medicalNotes}
            onChange={(e) => setMedicalNotes(e.target.value)}
            placeholder="Paste unstructured medical notes here for AI processing..."
            className="w-full h-32 bg-neutral-700 text-white p-3 rounded border border-neutral-600 resize-none"
          />
          <button
            onClick={processMedicalNotes}
            disabled={processing || !medicalNotes.trim()}
            className="mt-2 bg-primary-600 hover:bg-primary-700 disabled:bg-neutral-600 text-white px-4 py-2 rounded-lg h-11 min-w-[120px]"
          >
            {processing ? 'Processing...' : 'Process with AI'}
          </button>
        </div>

        {/* Patient Selection Modal */}
        {showPatientSelection && (
          <div className="mb-6 p-4 bg-neutral-700 rounded-lg">
            <h3 className="text-lg font-semibold text-white mb-4">
              Select Patient ({patientMatches.length} matches found)
            </h3>
            <div className="space-y-2">
              {patientMatches.map((match) => (
                <div
                  key={match.id}
                  className="flex items-center justify-between p-3 bg-neutral-600 rounded cursor-pointer hover:bg-neutral-500"
                  onClick={() => selectPatient(match.id)}
                >
                  <div>
                    <p className="text-white font-medium">{match.name}</p>
                    <p className="text-neutral-300 text-sm">
                      {match.email} • NHS: {match.nhsNumber} • MRN: {match.mrn}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className={`px-2 py-1 rounded text-xs ${
                      match.confidence >= 0.8 ? 'bg-medical-900 text-medical-300' :
                      match.confidence >= 0.6 ? 'bg-warning-900 text-warning-300' :
                      'bg-error-900 text-error-300'
                    }`}>
                      {Math.round(match.confidence * 100)}% match
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <button
              onClick={() => setShowPatientSelection(false)}
              className="mt-4 bg-neutral-600 hover:bg-neutral-500 text-white px-4 py-2 rounded"
            >
              Cancel
            </button>
          </div>
        )}

        {/* Extracted Data Preview */}
        {extractedData && (
          <div className="mb-6 p-4 bg-neutral-700 rounded-lg">
            <h3 className="text-lg font-semibold text-white mb-4">AI Extracted Data</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-neutral-300 mb-2">Patient Information</h4>
                <p className="text-white">Name: {extractedData.Patient_Name}</p>
                <p className="text-white">Age: {extractedData.Age}</p>
                <p className="text-white">Sex: {extractedData.Sex}</p>
              </div>
              <div>
                <h4 className="font-medium text-neutral-300 mb-2">Clinical Data</h4>
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
                  className="bg-medical-600 hover:bg-medical-700 text-white px-4 py-2 rounded-lg h-11 min-w-[120px]"
                >
                  Approve All
                </button>
              )}
            </div>

            {loading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto"></div>
                <p className="text-neutral-300 mt-2">Loading audit logs...</p>
              </div>
            ) : auditLogs.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-neutral-300">No pending AI suggestions for this patient.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {auditLogs.map((log) => (
                  <div
                    key={log.id}
                    className={`p-4 rounded-lg border ${
                      log.clinician_approved 
                        ? 'bg-medical-900 border-medical-700' 
                        : 'bg-neutral-700 border-neutral-600'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h4 className="font-medium text-white capitalize">
                          {log.field_name.replace(/_/g, ' ')}
                        </h4>
                        <p className="text-sm text-neutral-300">
                          {log.ai_suggestion}
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`text-xs px-2 py-1 rounded ${
                          getConfidenceColor(log.ai_confidence)
                        } bg-neutral-800`}>
                          {getConfidenceLabel(log.ai_confidence)} Confidence
                        </span>
                        {log.clinician_approved && (
                          <span className="text-xs px-2 py-1 rounded bg-medical-800 text-medical-300">
                            Approved
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                      <div>
                        <p className="text-sm text-neutral-400">Current Value:</p>
                        <p className="text-white">{log.old_value || 'Not set'}</p>
                      </div>
                      <div>
                        <p className="text-sm text-neutral-400">AI Suggestion:</p>
                        <p className="text-white">{log.new_value}</p>
                      </div>
                    </div>

                    {!log.clinician_approved && (
                      <div className="flex space-x-2">
                        <button
                          onClick={() => approveChange(log.id)}
                          className="bg-medical-600 hover:bg-medical-700 text-white px-3 py-2 rounded-lg text-sm h-9 min-w-[80px]"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => rejectChange(log.id)}
                          className="bg-error-600 hover:bg-error-700 text-white px-3 py-2 rounded-lg text-sm h-9 min-w-[80px]"
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

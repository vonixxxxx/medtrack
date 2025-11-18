import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FileText, X } from "lucide-react";
import { Button } from "../../ui/button";
import { createSoapNote, updateSoapNote, getSoapNote } from "../../../api";

export const SoapNoteEditor = ({ isOpen, onClose, encounterId, soapNoteId, onSuccess }) => {
  const [formData, setFormData] = useState({
    subjective: '',
    objective: '',
    assessment: '',
    plan: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (soapNoteId) {
      loadSoapNote();
    }
  }, [soapNoteId]);

  const loadSoapNote = async () => {
    try {
      const data = await getSoapNote(soapNoteId);
      setFormData({
        subjective: data.subjective || '',
        objective: data.objective || '',
        assessment: data.assessment || '',
        plan: data.plan || ''
      });
    } catch (err) {
      console.error('Error loading SOAP note:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      if (soapNoteId) {
        await updateSoapNote(soapNoteId, formData);
      } else {
        await createSoapNote({ encounterId, ...formData });
      }
      onSuccess?.();
      onClose();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to save SOAP note');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white rounded-2xl shadow-large max-w-4xl w-full max-h-[90vh] overflow-y-auto"
      >
        <div className="sticky top-0 bg-white border-b border-neutral-200 px-6 py-4 flex items-center justify-between">
          <h2 className="text-2xl font-semibold text-neutral-900">
            {soapNoteId ? 'Edit SOAP Note' : 'New SOAP Note'}
          </h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X size={20} />
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {error && (
            <div className="p-4 bg-error-50 border border-error-200 rounded-xl">
              <p className="text-error-600 text-sm font-medium">{error}</p>
            </div>
          )}

          <div>
            <label className="block text-sm font-semibold text-neutral-700 mb-2">
              Subjective (S)
            </label>
            <textarea
              value={formData.subjective}
              onChange={(e) => setFormData({ ...formData, subjective: e.target.value })}
              className="w-full min-h-[120px] rounded-md border border-neutral-200 px-4 py-3 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              placeholder="Patient's description of symptoms, history, concerns..."
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-neutral-700 mb-2">
              Objective (O)
            </label>
            <textarea
              value={formData.objective}
              onChange={(e) => setFormData({ ...formData, objective: e.target.value })}
              className="w-full min-h-[120px] rounded-md border border-neutral-200 px-4 py-3 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              placeholder="Vital signs, physical exam findings, lab results, observations..."
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-neutral-700 mb-2">
              Assessment (A)
            </label>
            <textarea
              value={formData.assessment}
              onChange={(e) => setFormData({ ...formData, assessment: e.target.value })}
              className="w-full min-h-[120px] rounded-md border border-neutral-200 px-4 py-3 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              placeholder="Clinical impression, diagnosis, differential diagnosis..."
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-neutral-700 mb-2">
              Plan (P)
            </label>
            <textarea
              value={formData.plan}
              onChange={(e) => setFormData({ ...formData, plan: e.target.value })}
              className="w-full min-h-[120px] rounded-md border border-neutral-200 px-4 py-3 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              placeholder="Treatment plan, medications, follow-up, patient education..."
            />
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-200">
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" loading={isLoading}>
              {soapNoteId ? 'Update' : 'Save'} SOAP Note
            </Button>
          </div>
        </form>
      </motion.div>
    </div>
  );
};




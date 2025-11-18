import { useState } from "react";
import { motion } from "framer-motion";
import { X } from "lucide-react";
import { Button } from "../../ui/button";
import { Input } from "../../ui/input";
import { createEncounter } from "../../../api";

export const EncounterForm = ({ isOpen, onClose, patientId, onSuccess }) => {
  const [formData, setFormData] = useState({
    patientId: patientId || '',
    providerId: '',
    encounterDate: new Date().toISOString().split('T')[0],
    encounterTime: '',
    encounterType: 'office',
    reason: '',
    status: 'finished',
    priority: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      await createEncounter(formData);
      onSuccess?.();
      onClose();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to create encounter');
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
        className="bg-white rounded-2xl shadow-large max-w-2xl w-full max-h-[90vh] overflow-y-auto"
      >
        <div className="sticky top-0 bg-white border-b border-neutral-200 px-6 py-4 flex items-center justify-between">
          <h2 className="text-2xl font-semibold text-neutral-900">New Encounter</h2>
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

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Encounter Date *
              </label>
              <Input
                type="date"
                value={formData.encounterDate}
                onChange={(e) => setFormData({ ...formData, encounterDate: e.target.value })}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Encounter Time
              </label>
              <Input
                type="time"
                value={formData.encounterTime}
                onChange={(e) => setFormData({ ...formData, encounterTime: e.target.value })}
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Encounter Type
              </label>
              <select
                value={formData.encounterType}
                onChange={(e) => setFormData({ ...formData, encounterType: e.target.value })}
                className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              >
                <option value="office">Office Visit</option>
                <option value="inpatient">Inpatient</option>
                <option value="outpatient">Outpatient</option>
                <option value="emergency">Emergency</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Status
              </label>
              <select
                value={formData.status}
                onChange={(e) => setFormData({ ...formData, status: e.target.value })}
                className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              >
                <option value="planned">Planned</option>
                <option value="arrived">Arrived</option>
                <option value="triaged">Triaged</option>
                <option value="in-progress">In Progress</option>
                <option value="finished">Finished</option>
                <option value="cancelled">Cancelled</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-neutral-700 mb-2">
              Reason for Visit
            </label>
            <Input
              value={formData.reason}
              onChange={(e) => setFormData({ ...formData, reason: e.target.value })}
              placeholder="Chief complaint or reason for visit"
            />
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-200">
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" loading={isLoading}>
              Create Encounter
            </Button>
          </div>
        </form>
      </motion.div>
    </div>
  );
};




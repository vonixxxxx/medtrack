import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { X } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { createAllergy, updateAllergy } from "../../api";

export const AllergyForm = ({ isOpen, onClose, allergy, patientId, onSuccess }) => {
  const [formData, setFormData] = useState({
    patientId: patientId || '',
    allergen: '',
    allergenType: '',
    reaction: '',
    severity: '',
    onsetDate: '',
    status: 'active',
    notes: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (allergy) {
      setFormData({
        patientId: allergy.patientId || patientId || '',
        allergen: allergy.allergen || '',
        allergenType: allergy.allergenType || '',
        reaction: allergy.reaction || '',
        severity: allergy.severity || '',
        onsetDate: allergy.onsetDate ? new Date(allergy.onsetDate).toISOString().split('T')[0] : '',
        status: allergy.status || 'active',
        notes: allergy.notes || ''
      });
    }
  }, [allergy, patientId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      if (allergy) {
        await updateAllergy(allergy.id, formData);
      } else {
        await createAllergy(formData);
      }
      onSuccess?.();
      onClose();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to save allergy');
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
          <h2 className="text-2xl font-semibold text-neutral-900">
            {allergy ? 'Edit Allergy' : 'Add Allergy'}
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
              Allergen *
            </label>
            <Input
              value={formData.allergen}
              onChange={(e) => setFormData({ ...formData, allergen: e.target.value })}
              placeholder="e.g., Penicillin, Peanuts"
              required
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Allergen Type
              </label>
              <select
                value={formData.allergenType}
                onChange={(e) => setFormData({ ...formData, allergenType: e.target.value })}
                className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              >
                <option value="">Select type</option>
                <option value="drug">Drug</option>
                <option value="food">Food</option>
                <option value="environmental">Environmental</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Severity
              </label>
              <select
                value={formData.severity}
                onChange={(e) => setFormData({ ...formData, severity: e.target.value })}
                className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              >
                <option value="">Select severity</option>
                <option value="mild">Mild</option>
                <option value="moderate">Moderate</option>
                <option value="severe">Severe</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Onset Date
              </label>
              <Input
                type="date"
                value={formData.onsetDate}
                onChange={(e) => setFormData({ ...formData, onsetDate: e.target.value })}
              />
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
                <option value="active">Active</option>
                <option value="resolved">Resolved</option>
                <option value="inactive">Inactive</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-neutral-700 mb-2">
              Reaction
            </label>
            <Input
              value={formData.reaction}
              onChange={(e) => setFormData({ ...formData, reaction: e.target.value })}
              placeholder="e.g., Hives, Difficulty breathing"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-neutral-700 mb-2">
              Notes
            </label>
            <textarea
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              className="w-full min-h-[100px] rounded-md border border-neutral-200 px-4 py-3 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              placeholder="Additional notes about the allergy..."
            />
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-200">
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" loading={isLoading}>
              {allergy ? 'Update' : 'Add'} Allergy
            </Button>
          </div>
        </form>
      </motion.div>
    </div>
  );
};




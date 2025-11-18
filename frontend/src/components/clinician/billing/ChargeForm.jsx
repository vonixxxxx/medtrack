import { useState } from "react";
import { motion } from "framer-motion";
import { X } from "lucide-react";
import { Button } from "../../ui/button";
import { Input } from "../../ui/input";
import { createCharge } from "../../../api";

export const ChargeForm = ({ isOpen, onClose, patientId, encounterId, onSuccess }) => {
  const [formData, setFormData] = useState({
    patientId: patientId || '',
    encounterId: encounterId || '',
    code: '',
    codeType: 'CPT',
    description: '',
    units: 1,
    fee: '',
    dateOfService: new Date().toISOString().split('T')[0],
    providerId: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      await createCharge({
        ...formData,
        units: parseFloat(formData.units) || 1,
        fee: parseFloat(formData.fee) || 0
      });
      onSuccess?.();
      onClose();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to create charge');
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
          <h2 className="text-2xl font-semibold text-neutral-900">Add Charge</h2>
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
                Code (CPT/HCPCS) *
              </label>
              <Input
                value={formData.code}
                onChange={(e) => setFormData({ ...formData, code: e.target.value })}
                placeholder="e.g., 99213"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Code Type
              </label>
              <select
                value={formData.codeType}
                onChange={(e) => setFormData({ ...formData, codeType: e.target.value })}
                className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              >
                <option value="CPT">CPT</option>
                <option value="HCPCS">HCPCS</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Description
              </label>
              <Input
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="Service description"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Date of Service *
              </label>
              <Input
                type="date"
                value={formData.dateOfService}
                onChange={(e) => setFormData({ ...formData, dateOfService: e.target.value })}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Units
              </label>
              <Input
                type="number"
                value={formData.units}
                onChange={(e) => setFormData({ ...formData, units: e.target.value })}
                min="1"
                step="1"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Fee ($) *
              </label>
              <Input
                type="number"
                value={formData.fee}
                onChange={(e) => setFormData({ ...formData, fee: e.target.value })}
                placeholder="0.00"
                step="0.01"
                min="0"
                required
              />
            </div>
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-200">
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" loading={isLoading}>
              Add Charge
            </Button>
          </div>
        </form>
      </motion.div>
    </div>
  );
};




import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { X } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { createPrescription, updatePrescription } from "../../api";

export const PrescriptionForm = ({ isOpen, onClose, prescription, patientId, encounterId, onSuccess }) => {
  const [formData, setFormData] = useState({
    patientId: patientId || '',
    encounterId: encounterId || '',
    medicationName: '',
    ndcCode: '',
    rxnormCode: '',
    dosage: '',
    unit: '',
    route: '',
    frequency: '',
    quantity: '',
    refills: 0,
    startDate: new Date().toISOString().split('T')[0],
    endDate: '',
    status: 'active',
    instructions: '',
    pharmacyId: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (prescription) {
      setFormData({
        patientId: prescription.patientId || patientId || '',
        encounterId: prescription.encounterId || encounterId || '',
        medicationName: prescription.medicationName || '',
        ndcCode: prescription.ndcCode || '',
        rxnormCode: prescription.rxnormCode || '',
        dosage: prescription.dosage || '',
        unit: prescription.unit || '',
        route: prescription.route || '',
        frequency: prescription.frequency || '',
        quantity: prescription.quantity?.toString() || '',
        refills: prescription.refills || 0,
        startDate: prescription.startDate ? new Date(prescription.startDate).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
        endDate: prescription.endDate ? new Date(prescription.endDate).toISOString().split('T')[0] : '',
        status: prescription.status || 'active',
        instructions: prescription.instructions || '',
        pharmacyId: prescription.pharmacyId || ''
      });
    }
  }, [prescription, patientId, encounterId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const submitData = {
        ...formData,
        quantity: formData.quantity ? parseFloat(formData.quantity) : null,
        refills: parseInt(formData.refills) || 0
      };
      
      if (prescription) {
        await updatePrescription(prescription.id, submitData);
      } else {
        await createPrescription(submitData);
      }
      onSuccess?.();
      onClose();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to save prescription');
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
            {prescription ? 'Edit Prescription' : 'New Prescription'}
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
              Medication Name *
            </label>
            <Input
              value={formData.medicationName}
              onChange={(e) => setFormData({ ...formData, medicationName: e.target.value })}
              placeholder="e.g., Metformin 500mg"
              required
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Dosage
              </label>
              <Input
                value={formData.dosage}
                onChange={(e) => setFormData({ ...formData, dosage: e.target.value })}
                placeholder="e.g., 500"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Unit
              </label>
              <Input
                value={formData.unit}
                onChange={(e) => setFormData({ ...formData, unit: e.target.value })}
                placeholder="e.g., mg, mL"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Route
              </label>
              <select
                value={formData.route}
                onChange={(e) => setFormData({ ...formData, route: e.target.value })}
                className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              >
                <option value="">Select route</option>
                <option value="oral">Oral</option>
                <option value="topical">Topical</option>
                <option value="injection">Injection</option>
                <option value="inhalation">Inhalation</option>
                <option value="nasal">Nasal</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Frequency
              </label>
              <Input
                value={formData.frequency}
                onChange={(e) => setFormData({ ...formData, frequency: e.target.value })}
                placeholder="e.g., Once daily, Twice daily"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Quantity
              </label>
              <Input
                type="number"
                value={formData.quantity}
                onChange={(e) => setFormData({ ...formData, quantity: e.target.value })}
                placeholder="30"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Refills
              </label>
              <Input
                type="number"
                value={formData.refills}
                onChange={(e) => setFormData({ ...formData, refills: parseInt(e.target.value) || 0 })}
                min="0"
                placeholder="0"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Start Date *
              </label>
              <Input
                type="date"
                value={formData.startDate}
                onChange={(e) => setFormData({ ...formData, startDate: e.target.value })}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                End Date
              </label>
              <Input
                type="date"
                value={formData.endDate}
                onChange={(e) => setFormData({ ...formData, endDate: e.target.value })}
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
                <option value="filled">Filled</option>
                <option value="cancelled">Cancelled</option>
                <option value="expired">Expired</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-neutral-700 mb-2">
              Instructions
            </label>
            <textarea
              value={formData.instructions}
              onChange={(e) => setFormData({ ...formData, instructions: e.target.value })}
              className="w-full min-h-[100px] rounded-md border border-neutral-200 px-4 py-3 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              placeholder="e.g., Take with food, Take at bedtime"
            />
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-200">
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" loading={isLoading}>
              {prescription ? 'Update' : 'Create'} Prescription
            </Button>
          </div>
        </form>
      </motion.div>
    </div>
  );
};




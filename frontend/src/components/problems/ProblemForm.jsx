import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { X } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { createProblem, updateProblem } from "../../api";

export const ProblemForm = ({ isOpen, onClose, problem, patientId, onSuccess }) => {
  const [formData, setFormData] = useState({
    patientId: patientId || '',
    title: '',
    code: '',
    codeType: 'ICD10',
    beginDate: '',
    endDate: '',
    status: 'active',
    severity: '',
    notes: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (problem) {
      setFormData({
        patientId: problem.patientId || patientId || '',
        title: problem.title || '',
        code: problem.code || '',
        codeType: problem.codeType || 'ICD10',
        beginDate: problem.beginDate ? new Date(problem.beginDate).toISOString().split('T')[0] : '',
        endDate: problem.endDate ? new Date(problem.endDate).toISOString().split('T')[0] : '',
        status: problem.status || 'active',
        severity: problem.severity || '',
        notes: problem.notes || ''
      });
    }
  }, [problem, patientId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      if (problem) {
        await updateProblem(problem.id, formData);
      } else {
        await createProblem(formData);
      }
      onSuccess?.();
      onClose();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to save problem');
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
            {problem ? 'Edit Problem' : 'Add Problem'}
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
              Problem Title *
            </label>
            <Input
              value={formData.title}
              onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              placeholder="e.g., Type 2 Diabetes"
              required
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                ICD-10 Code
              </label>
              <Input
                value={formData.code}
                onChange={(e) => setFormData({ ...formData, code: e.target.value })}
                placeholder="E11.9"
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

            <div>
              <label className="block text-sm font-semibold text-neutral-700 mb-2">
                Begin Date
              </label>
              <Input
                type="date"
                value={formData.beginDate}
                onChange={(e) => setFormData({ ...formData, beginDate: e.target.value })}
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
          </div>

          <div>
            <label className="block text-sm font-semibold text-neutral-700 mb-2">
              Notes
            </label>
            <textarea
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              className="w-full min-h-[100px] rounded-md border border-neutral-200 px-4 py-3 text-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500"
              placeholder="Additional notes about the problem..."
            />
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-200">
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" loading={isLoading}>
              {problem ? 'Update' : 'Add'} Problem
            </Button>
          </div>
        </form>
      </motion.div>
    </div>
  );
};




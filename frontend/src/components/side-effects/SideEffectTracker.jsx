import { useState, useEffect } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { AlertCircle, Plus, X, Calendar } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Input } from '../ui/input';
import { LoadingSkeleton } from '../dashboard/LoadingSkeleton';
import { EmptyState } from '../dashboard/EmptyState';
import { getSideEffects, createSideEffect, updateSideEffect, deleteSideEffect } from '../../api';

export const SideEffectTracker = ({ medicationId, patientId, medicationName, medications }) => {
  const prefersReducedMotion = useReducedMotion();
  const [sideEffects, setSideEffects] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formData, setFormData] = useState({
    symptom: '',
    severity: 'mild',
    onsetDate: new Date().toISOString().split('T')[0],
    resolvedDate: '',
    notes: ''
  });

  useEffect(() => {
    loadSideEffects();
  }, [medicationId, patientId]);

  const loadSideEffects = async () => {
    try {
      setIsLoading(true);
      const params = {};
      if (medicationId) params.medicationId = medicationId;
      if (patientId) params.patientId = patientId;
      
      const data = await getSideEffects(params);
      // Ensure data is always an array
      const sideEffectsArray = Array.isArray(data) ? data : [];
      setSideEffects(sideEffectsArray);
    } catch (error) {
      console.error('Error loading side effects:', error);
      setSideEffects([]); // Set empty array on error
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editingId) {
        await updateSideEffect(editingId, {
          ...formData,
          resolvedDate: formData.resolvedDate || null
        });
      } else {
        await createSideEffect({
          medicationId,
          ...formData,
          resolvedDate: formData.resolvedDate || null
        });
      }
      setIsFormOpen(false);
      setEditingId(null);
      setFormData({
        symptom: '',
        severity: 'mild',
        onsetDate: new Date().toISOString().split('T')[0],
        resolvedDate: '',
        notes: ''
      });
      loadSideEffects();
    } catch (error) {
      console.error('Error saving side effect:', error);
    }
  };

  const handleEdit = (sideEffect) => {
    setEditingId(sideEffect.id);
    setFormData({
      symptom: sideEffect.symptom,
      severity: sideEffect.severity || 'mild',
      onsetDate: sideEffect.onsetDate ? new Date(sideEffect.onsetDate).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
      resolvedDate: sideEffect.resolvedDate ? new Date(sideEffect.resolvedDate).toISOString().split('T')[0] : '',
      notes: sideEffect.notes || ''
    });
    setIsFormOpen(true);
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this side effect?')) return;
    try {
      await deleteSideEffect(id);
      loadSideEffects();
    } catch (error) {
      console.error('Error deleting side effect:', error);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'severe':
        return 'error';
      case 'moderate':
        return 'warning';
      case 'mild':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <DashboardCard
      title={medicationName ? `Side Effects - ${medicationName}` : 'Side Effects'}
      icon={<AlertCircle size={20} />}
      variant="patient"
      action={
        <Button
          onClick={() => {
            setIsFormOpen(true);
            setEditingId(null);
            setFormData({
              symptom: '',
              severity: 'mild',
              onsetDate: new Date().toISOString().split('T')[0],
              resolvedDate: '',
              notes: ''
            });
          }}
          variant="primary"
          size="sm"
        >
          <Plus size={16} className="mr-1.5" />
          Add
        </Button>
      }
    >
      {isLoading ? (
        <LoadingSkeleton variant="list" count={3} />
      ) : (
        <>
          {/* Medication Selector (if multiple medications) */}
          {!medicationId && medications && medications.length > 0 && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-neutral-700 mb-2">
                Select Medication
              </label>
              <select
                value={medicationId || ''}
                onChange={(e) => {
                  const med = medications.find(m => m.id === e.target.value);
                  if (med) {
                    // This would need to be handled by parent component
                  }
                }}
                className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="">All Medications</option>
                {medications.map(med => (
                  <option key={med.id} value={med.id}>
                    {med.name || med.medication_name || med.generic_name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Side Effects List */}
          {sideEffects.length === 0 ? (
            <EmptyState
              icon={AlertCircle}
              title="No side effects recorded"
              description="Track any side effects you experience with your medications"
              action={{
                label: "Add Side Effect",
                onClick: () => setIsFormOpen(true)
              }}
            />
          ) : (
            <div className="space-y-3">
              {Array.isArray(sideEffects) && sideEffects.map((effect, index) => (
                <motion.div
                  key={effect.id}
                  initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="p-4 bg-white rounded-2xl border border-neutral-200 hover:border-primary-300 hover:shadow-medium transition-all"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-semibold text-neutral-900">{effect.symptom}</h4>
                        <Badge variant={getSeverityColor(effect.severity)}>
                          {effect.severity || 'Unknown'}
                        </Badge>
                        {effect.resolvedDate && (
                          <Badge variant="success">Resolved</Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-4 text-xs text-neutral-600 mb-2">
                        <div className="flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          <span>Onset: {new Date(effect.onsetDate).toLocaleDateString()}</span>
                        </div>
                        {effect.resolvedDate && (
                          <div className="flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            <span>Resolved: {new Date(effect.resolvedDate).toLocaleDateString()}</span>
                          </div>
                        )}
                      </div>
                      {effect.notes && (
                        <p className="text-sm text-neutral-700">{effect.notes}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        onClick={() => handleEdit(effect)}
                        variant="ghost"
                        size="sm"
                      >
                        Edit
                      </Button>
                      <Button
                        onClick={() => handleDelete(effect.id)}
                        variant="ghost"
                        size="sm"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          {/* Add/Edit Form */}
          {isFormOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-6 p-6 bg-neutral-50 rounded-2xl border border-neutral-200"
            >
              <h4 className="font-semibold text-neutral-900 mb-4">
                {editingId ? 'Edit Side Effect' : 'Add Side Effect'}
              </h4>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Symptom *
                  </label>
                  <Input
                    value={formData.symptom}
                    onChange={(e) => setFormData({ ...formData, symptom: e.target.value })}
                    placeholder="e.g., Nausea, Headache, Dizziness"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Severity *
                  </label>
                  <select
                    value={formData.severity}
                    onChange={(e) => setFormData({ ...formData, severity: e.target.value })}
                    className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus:outline-none focus:ring-2 focus:ring-primary-500"
                    required
                  >
                    <option value="mild">Mild</option>
                    <option value="moderate">Moderate</option>
                    <option value="severe">Severe</option>
                  </select>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-2">
                      Onset Date *
                    </label>
                    <Input
                      type="date"
                      value={formData.onsetDate}
                      onChange={(e) => setFormData({ ...formData, onsetDate: e.target.value })}
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-2">
                      Resolved Date
                    </label>
                    <Input
                      type="date"
                      value={formData.resolvedDate}
                      onChange={(e) => setFormData({ ...formData, resolvedDate: e.target.value })}
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Notes
                  </label>
                  <textarea
                    value={formData.notes}
                    onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                    className="w-full h-24 rounded-md border border-neutral-200 px-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-primary-500"
                    placeholder="Additional details about the side effect..."
                  />
                </div>
                <div className="flex items-center gap-3">
                  <Button type="submit" variant="primary" size="md">
                    {editingId ? 'Update' : 'Add'} Side Effect
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="md"
                    onClick={() => {
                      setIsFormOpen(false);
                      setEditingId(null);
                    }}
                  >
                    Cancel
                  </Button>
                </div>
              </form>
            </motion.div>
          )}
        </>
      )}
    </DashboardCard>
  );
};


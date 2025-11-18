import { useState, useEffect } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Package, AlertTriangle, CheckCircle, Plus, Edit2 } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Input } from '../ui/input';
import { LoadingSkeleton } from '../dashboard/LoadingSkeleton';
import { EmptyState } from '../dashboard/EmptyState';
import api from '../../api';

export const MedicationStockTracker = ({ medications = [], onUpdate }) => {
  const prefersReducedMotion = useReducedMotion();
  const [isLoading, setIsLoading] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formData, setFormData] = useState({
    stockQuantity: '',
    stockUnit: 'pills',
    lowStockThreshold: ''
  });

  const medicationsWithStock = medications.filter(med => 
    med.stockQuantity !== null && med.stockQuantity !== undefined
  );

  const lowStockMedications = medicationsWithStock.filter(med => {
    if (!med.stockQuantity || !med.lowStockThreshold) return false;
    return med.stockQuantity <= med.lowStockThreshold;
  });

  const outOfStockMedications = medicationsWithStock.filter(med => 
    med.outOfStock || (med.stockQuantity !== null && med.stockQuantity <= 0)
  );

  const handleUpdateStock = async (medicationId) => {
    try {
      setIsLoading(true);
      await api.put(`/meds/user/${medicationId}`, {
        stockQuantity: parseFloat(formData.stockQuantity) || null,
        stockUnit: formData.stockUnit,
        lowStockThreshold: parseFloat(formData.lowStockThreshold) || null,
        outOfStock: parseFloat(formData.stockQuantity) <= 0
      });
      setEditingId(null);
      setFormData({ stockQuantity: '', stockUnit: 'pills', lowStockThreshold: '' });
      if (onUpdate) onUpdate();
    } catch (error) {
      console.error('Error updating stock:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleEdit = (medication) => {
    setEditingId(medication.id);
    setFormData({
      stockQuantity: medication.stockQuantity?.toString() || '',
      stockUnit: medication.stockUnit || 'pills',
      lowStockThreshold: medication.lowStockThreshold?.toString() || ''
    });
  };

  const getStockStatus = (medication) => {
    if (medication.outOfStock || (medication.stockQuantity !== null && medication.stockQuantity <= 0)) {
      return { status: 'out', color: 'error', label: 'Out of Stock' };
    }
    if (medication.lowStockThreshold && medication.stockQuantity <= medication.lowStockThreshold) {
      return { status: 'low', color: 'warning', label: 'Low Stock' };
    }
    return { status: 'ok', color: 'success', label: 'In Stock' };
  };

  return (
    <DashboardCard
      title="Medication Stock Tracker"
      icon={<Package size={20} />}
      variant="patient"
    >
      {isLoading ? (
        <LoadingSkeleton variant="list" count={3} />
      ) : (
        <div className="space-y-6">
          {/* Alerts */}
          {outOfStockMedications.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: prefersReducedMotion ? 0 : -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-error-50 border-2 border-error-300 rounded-xl"
            >
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-error-600" />
                <h4 className="font-semibold text-error-900">
                  Out of Stock ({outOfStockMedications.length})
                </h4>
              </div>
              <p className="text-sm text-error-700">
                {outOfStockMedications.map(m => m.name || m.medication_name).join(', ')}
              </p>
            </motion.div>
          )}

          {lowStockMedications.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: prefersReducedMotion ? 0 : -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-warning-50 border-2 border-warning-300 rounded-xl"
            >
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-warning-600" />
                <h4 className="font-semibold text-warning-900">
                  Low Stock ({lowStockMedications.length})
                </h4>
              </div>
              <p className="text-sm text-warning-700">
                {lowStockMedications.map(m => m.name || m.medication_name).join(', ')}
              </p>
            </motion.div>
          )}

          {/* Stock List */}
          {medicationsWithStock.length === 0 ? (
            <EmptyState
              icon={Package}
              title="No stock tracking set up"
              description="Start tracking your medication inventory to get low stock alerts"
            />
          ) : (
            <div className="space-y-3">
              {medicationsWithStock.map((medication, index) => {
                const stockStatus = getStockStatus(medication);
                const isEditing = editingId === medication.id;

                return (
                  <motion.div
                    key={medication.id}
                    initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="p-4 bg-white rounded-2xl border border-neutral-200 hover:border-primary-300 hover:shadow-medium transition-all"
                  >
                    {isEditing ? (
                      <div className="space-y-3">
                        <div>
                          <label className="block text-xs font-medium text-neutral-700 mb-1">
                            Current Stock
                          </label>
                          <div className="flex items-center gap-2">
                            <Input
                              type="number"
                              value={formData.stockQuantity}
                              onChange={(e) => setFormData({ ...formData, stockQuantity: e.target.value })}
                              placeholder="0"
                              className="flex-1"
                            />
                            <select
                              value={formData.stockUnit}
                              onChange={(e) => setFormData({ ...formData, stockUnit: e.target.value })}
                              className="h-11 rounded-md border border-neutral-200 px-3 text-sm"
                            >
                              <option value="pills">Pills</option>
                              <option value="tablets">Tablets</option>
                              <option value="capsules">Capsules</option>
                              <option value="ml">mL</option>
                              <option value="mg">mg</option>
                            </select>
                          </div>
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-neutral-700 mb-1">
                            Low Stock Alert At
                          </label>
                          <Input
                            type="number"
                            value={formData.lowStockThreshold}
                            onChange={(e) => setFormData({ ...formData, lowStockThreshold: e.target.value })}
                            placeholder="e.g., 10"
                          />
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            onClick={() => handleUpdateStock(medication.id)}
                            variant="primary"
                            size="sm"
                          >
                            Save
                          </Button>
                          <Button
                            onClick={() => {
                              setEditingId(null);
                              setFormData({ stockQuantity: '', stockUnit: 'pills', lowStockThreshold: '' });
                            }}
                            variant="ghost"
                            size="sm"
                          >
                            Cancel
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h4 className="font-semibold text-neutral-900">
                              {medication.name || medication.medication_name}
                            </h4>
                            <Badge variant={stockStatus.color}>
                              {stockStatus.label}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-neutral-600">
                            <span>
                              Stock: {medication.stockQuantity} {medication.stockUnit || 'pills'}
                            </span>
                            {medication.lowStockThreshold && (
                              <span>
                                Alert at: {medication.lowStockThreshold} {medication.stockUnit || 'pills'}
                              </span>
                            )}
                          </div>
                        </div>
                        <Button
                          onClick={() => handleEdit(medication)}
                          variant="ghost"
                          size="sm"
                        >
                          <Edit2 className="w-4 h-4" />
                        </Button>
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          )}

          {/* Add Stock Tracking */}
          {medications.filter(m => !m.stockQuantity && m.stockQuantity !== 0).length > 0 && (
            <div className="pt-4 border-t border-neutral-200">
              <p className="text-sm text-neutral-600 mb-3">
                Add stock tracking to medications:
              </p>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {medications
                  .filter(m => !m.stockQuantity && m.stockQuantity !== 0)
                  .map((medication) => (
                    <button
                      key={medication.id}
                      onClick={() => handleEdit(medication)}
                      className="w-full text-left p-2 rounded-lg hover:bg-neutral-50 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <Plus className="w-4 h-4 text-neutral-400" />
                        <span className="text-sm text-neutral-700">
                          {medication.name || medication.medication_name}
                        </span>
                      </div>
                    </button>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </DashboardCard>
  );
};




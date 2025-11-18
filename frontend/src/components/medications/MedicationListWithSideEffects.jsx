import { useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Pill, Info, AlertTriangle } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '../ui/dialog';
import { LoadingSkeleton } from '../dashboard/LoadingSkeleton';
import { EmptyState } from '../dashboard/EmptyState';
import api from '../../api';

/**
 * Medication List with Side Effects
 * Based on Confir-Med patient page medication list
 * Shows medications with ability to view side effects
 */

export const MedicationListWithSideEffects = ({ medications = [], patientId }) => {
  const prefersReducedMotion = useReducedMotion();
  const [selectedMedication, setSelectedMedication] = useState(null);
  const [sideEffects, setSideEffects] = useState(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleViewSideEffects = async (medication) => {
    setSelectedMedication(medication);
    setIsDialogOpen(true);
    setIsLoading(true);
    setSideEffects(null);

    try {
      // Get monopharmacy side effects
      const response = await api.get('/mono_se', {
        params: {
          drug_name: medication.name || medication.medication_name || medication.generic_name
        }
      });

      const effects = response.data?.side_effects || [];
      setSideEffects(effects);
    } catch (error) {
      console.error('Error fetching side effects:', error);
      setSideEffects([]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatFrequency = (frequency) => {
    if (!frequency) return 'Not specified';
    
    const frequencyMap = {
      'daily': 'Once daily',
      'twice_daily': 'Twice daily',
      'three_times_daily': 'Three times daily',
      'four_times_daily': 'Four times daily',
      'weekly': 'Once weekly',
      'monthly': 'Once monthly',
      'as_needed': 'As needed'
    };
    
    return frequencyMap[frequency] || frequency;
  };

  return (
    <>
      <DashboardCard
        title="Medications"
        icon={<Pill size={20} />}
        variant="patient"
      >
        {medications.length === 0 ? (
          <EmptyState
            icon={Pill}
            title="No medications"
            description="Add medications to track side effects and interactions"
          />
        ) : (
          <div className="space-y-3">
            {medications.map((medication, index) => (
              <motion.div
                key={medication.id || index}
                initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="flex items-center gap-3 p-4 bg-white rounded-2xl border border-neutral-200 hover:border-primary-300 hover:shadow-medium transition-all"
              >
                <Button
                  onClick={() => handleViewSideEffects(medication)}
                  variant="ghost"
                  size="icon"
                  className="flex-shrink-0"
                >
                  <Info className="w-5 h-5 text-primary-600" />
                </Button>
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-neutral-900 text-base">
                    {medication.name || medication.medication_name || medication.generic_name}
                  </p>
                  <p className="text-sm text-neutral-600">
                    {formatFrequency(medication.frequency)}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </DashboardCard>

      {/* Side Effects Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Side Effects for {selectedMedication?.name || selectedMedication?.medication_name || selectedMedication?.generic_name}
            </DialogTitle>
            <DialogDescription>
              {isLoading ? (
                <div className="flex items-center gap-2 py-4">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600" />
                  <span>Loading side effects...</span>
                </div>
              ) : sideEffects && sideEffects.length > 0 ? (
                <div className="space-y-2 py-4">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="w-5 h-5 text-warning-600 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-neutral-900 mb-2">
                        Common side effects:
                      </p>
                      <ul className="list-disc list-inside space-y-1 text-sm text-neutral-700">
                        {sideEffects.slice(0, 10).map((effect, idx) => (
                          <li key={idx}>{effect}</li>
                        ))}
                      </ul>
                      {sideEffects.length > 10 && (
                        <p className="text-xs text-neutral-500 mt-2">
                          Showing first 10 of {sideEffects.length} side effects
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-4">
                  <p className="text-sm text-neutral-600">
                    No side effects found for this medication.
                  </p>
                </div>
              )}
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </>
  );
};




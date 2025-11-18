import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, XCircle, Info } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { LoadingSkeleton } from '../dashboard/LoadingSkeleton';
import { checkDrugInteractions, getMedicationInteractions } from '../../api';
import { useReducedMotion } from 'framer-motion';

export const DrugInteractionChecker = ({ medications = [], onMedicationSelect }) => {
  const prefersReducedMotion = useReducedMotion();
  const [selectedMedications, setSelectedMedications] = useState([]);
  const [interactions, setInteractions] = useState([]);
  const [isChecking, setIsChecking] = useState(false);
  const [error, setError] = useState(null);

  const handleMedicationToggle = (medicationId) => {
    setSelectedMedications(prev => {
      if (prev.includes(medicationId)) {
        return prev.filter(id => id !== medicationId);
      } else {
        return [...prev, medicationId];
      }
    });
  };

  const checkInteractions = async () => {
    if (selectedMedications.length < 2) {
      setError('Please select at least 2 medications to check for interactions');
      return;
    }

    setIsChecking(true);
    setError(null);

    try {
      const result = await checkDrugInteractions({
        medicationIds: selectedMedications
      });
      const interactions = result.interactions || [];
      setInteractions(interactions);
      
      // Show summary if interactions found
      if (interactions.length > 0) {
        const severeCount = interactions.filter(i => i.severity === 'severe').length;
        const moderateCount = interactions.filter(i => i.severity === 'moderate').length;
        console.log(`Found ${interactions.length} interactions: ${severeCount} severe, ${moderateCount} moderate`);
      }
    } catch (err) {
      console.error('Error checking interactions:', err);
      setError('Failed to check drug interactions. Please try again.');
    } finally {
      setIsChecking(false);
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

  const getSeverityIcon = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'severe':
        return <XCircle className="w-5 h-5" />;
      case 'moderate':
        return <AlertTriangle className="w-5 h-5" />;
      case 'mild':
        return <Info className="w-5 h-5" />;
      default:
        return <CheckCircle className="w-5 h-5" />;
    }
  };

  return (
    <DashboardCard
      title="Drug Interaction Checker"
      icon={<AlertTriangle size={20} />}
      variant="patient"
    >
      <div className="space-y-6">
        {/* Medication Selection */}
        <div>
          <h4 className="text-sm font-semibold text-neutral-900 mb-3">
            Select Medications to Check
          </h4>
          {medications.length === 0 ? (
            <p className="text-sm text-neutral-600">No medications available to check</p>
          ) : (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {medications.map((med) => (
                <motion.label
                  key={med.id}
                  initial={{ opacity: 0, x: prefersReducedMotion ? 0 : -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-center gap-3 p-3 rounded-xl border border-neutral-200 hover:border-primary-300 hover:bg-primary-50/50 cursor-pointer transition-all"
                >
                  <input
                    type="checkbox"
                    checked={selectedMedications.includes(med.id)}
                    onChange={() => handleMedicationToggle(med.id)}
                    className="w-4 h-4 text-primary-600 rounded border-neutral-300 focus:ring-primary-500"
                  />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-neutral-900">
                      {med.name || med.medication_name || med.generic_name}
                    </p>
                    {med.dosage && (
                      <p className="text-xs text-neutral-600">{med.dosage}</p>
                    )}
                  </div>
                </motion.label>
              ))}
            </div>
          )}
        </div>

        {/* Check Button */}
        <Button
          onClick={checkInteractions}
          disabled={selectedMedications.length < 2 || isChecking}
          variant="primary"
          size="md"
          className="w-full"
        >
          {isChecking ? 'Checking...' : `Check Interactions (${selectedMedications.length} selected)`}
        </Button>

        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: prefersReducedMotion ? 0 : -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 bg-error-50 border border-error-200 rounded-xl"
          >
            <p className="text-sm text-error-700">{error}</p>
          </motion.div>
        )}

        {/* Interactions Results */}
        {interactions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <h4 className="text-sm font-semibold text-neutral-900">
              Interaction Results ({interactions.length})
            </h4>
            {interactions.map((interaction, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 rounded-2xl border-2 ${
                  interaction.severity === 'severe'
                    ? 'bg-error-50 border-error-300'
                    : interaction.severity === 'moderate'
                    ? 'bg-warning-50 border-warning-300'
                    : 'bg-info-50 border-info-300'
                }`}
              >
                <div className="flex items-start gap-3 mb-3">
                  <div className={`text-${getSeverityColor(interaction.severity)}-600 flex-shrink-0 mt-0.5`}>
                    {getSeverityIcon(interaction.severity)}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h5 className="font-semibold text-neutral-900">
                        {interaction.medication1} + {interaction.medication2}
                      </h5>
                      <Badge variant={getSeverityColor(interaction.severity)}>
                        {interaction.severity || interaction.interactionType}
                      </Badge>
                    </div>
                    <p className="text-sm text-neutral-700 mb-2">{interaction.description}</p>
                    {interaction.clinicalSignificance && (
                      <div className="mt-2 p-2 bg-white/50 rounded-lg">
                        <p className="text-xs font-medium text-neutral-900 mb-1">
                          Clinical Significance:
                        </p>
                        <p className="text-xs text-neutral-700">
                          {interaction.clinicalSignificance}
                        </p>
                      </div>
                    )}
                    {interaction.management && (
                      <div className="mt-2 p-2 bg-white/50 rounded-lg">
                        <p className="text-xs font-medium text-neutral-900 mb-1">
                          Management:
                        </p>
                        <p className="text-xs text-neutral-700">{interaction.management}</p>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}

        {/* No Interactions */}
        {interactions.length === 0 && !isChecking && !error && selectedMedications.length >= 2 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="p-4 bg-medical-50 border border-medical-200 rounded-xl text-center"
          >
            <CheckCircle className="w-8 h-8 text-medical-600 mx-auto mb-2" />
            <p className="text-sm font-medium text-medical-900">
              No known interactions found
            </p>
            <p className="text-xs text-medical-700 mt-1">
              Always consult your healthcare provider before making medication changes
            </p>
          </motion.div>
        )}
      </div>
    </DashboardCard>
  );
};

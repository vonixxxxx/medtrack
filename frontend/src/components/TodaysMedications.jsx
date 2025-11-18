import { useState, useEffect } from "react";
import { Pill, Plus, CheckCircle, Info } from "lucide-react";
import { getMonopharmacySideEffects } from "../api";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "./ui/dialog";
import DashboardCard from "./DashboardCard";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { LoadingSkeleton } from "./dashboard/LoadingSkeleton";
import { EmptyState } from "./dashboard/EmptyState";
import { motion, useReducedMotion } from "framer-motion";
import api from "../api";
import { getMedicationsWithWarnings } from "../api";

export const TodaysMedications = ({ onAddMedication, refreshTrigger, onRefresh }) => {
  const prefersReducedMotion = useReducedMotion();
  const [medications, setMedications] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedMedication, setSelectedMedication] = useState(null);
  const [sideEffects, setSideEffects] = useState(null);
  const [isSideEffectsDialogOpen, setIsSideEffectsDialogOpen] = useState(false);
  const [isLoadingSideEffects, setIsLoadingSideEffects] = useState(false);

  // Fetch medications from API with interaction warnings
  const fetchMedications = async () => {
    try {
      setIsLoading(true);
      // Get user ID
      const userStr = localStorage.getItem('user');
      const user = userStr ? JSON.parse(userStr) : null;
      const userId = user?.id;
      
      if (!userId) {
        console.error('No user ID found');
        setMedications([]);
        setError('Please log in to view medications');
        return;
      }
      
      // Try to get medications with warnings first
      try {
        const warningsData = await getMedicationsWithWarnings({ userId });
        if (warningsData && warningsData.medications && Array.isArray(warningsData.medications)) {
          // Filter out test medications
          const filteredMeds = warningsData.medications.filter(med => {
            const name = (med.medication_name || med.name || med.generic_name || '').toLowerCase();
            return !name.includes('final test') && 
                   !name.includes('test2') && 
                   !name.includes('test medication');
          });
          console.log('Got medications with warnings:', filteredMeds.length);
          setMedications(filteredMeds);
          if (warningsData.hasInteractions) {
            console.log('Interaction warnings:', warningsData.medicationWarnings);
          }
          setError(null);
          return;
        }
      } catch (warnErr) {
        console.log('Warnings endpoint failed, using regular endpoint:', warnErr.message);
      }
      
      // Fallback to regular endpoint
      console.log('Fetching medications for user:', userId);
      const response = await api.get('meds/user', {
        params: { userId }
      });
      
      console.log('Medications response:', response.data);
      const medications = response.data?.medications || response.data || [];
      // Filter out test medications
      const filteredMeds = medications.filter(med => {
        const name = (med.medication_name || med.name || med.generic_name || '').toLowerCase();
        return !name.includes('final test') && 
               !name.includes('test2') && 
               !name.includes('test medication');
      });
      console.log('Setting medications:', Array.isArray(filteredMeds) ? filteredMeds.length : 'not an array');
      setMedications(Array.isArray(filteredMeds) ? filteredMeds : []);
      setError(null);
    } catch (err) {
      console.error('Error fetching medications:', err);
      setError('Failed to load medications');
      // Fallback to empty array
      setMedications([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch medications on mount and when refreshTrigger changes
  useEffect(() => {
    fetchMedications();
  }, [refreshTrigger]);

  // Format time for display
  const formatTime = (timeString) => {
    if (!timeString) return 'Not scheduled';
    try {
      const time = new Date(timeString);
      return time.toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
      });
    } catch {
      return timeString;
    }
  };

  // Format strength to avoid duplicate units
  const formatStrength = (strength, unit) => {
    if (!strength) return 'Not specified';
    
    // If strength already includes unit, don't add it again
    if (unit && strength.includes(unit)) {
      return strength;
    }
    
    // If strength is just a number, add the unit
    if (unit && /^\d+$/.test(strength)) {
      return `${strength}${unit}`;
    }
    
    return strength;
  };

  // Format frequency for display
  const formatFrequency = (frequency, customFrequency) => {
    if (frequency === 'custom' && customFrequency) {
      return customFrequency;
    }
    
    const frequencyMap = {
      'daily': 'Once daily',
      'twice_daily': 'Twice daily',
      'three_times_daily': 'Three times daily',
      'four_times_daily': 'Four times daily',
      'weekly': 'Once weekly',
      'monthly': 'Once monthly',
      'as_needed': 'As needed'
    };
    
    return frequencyMap[frequency] || frequency || 'Not specified';
  };

  // Handle marking medication as taken
  const handleMarkTaken = async (medicationId) => {
    try {
      await api.post('meds/user/log-dose', {
        medicationId,
        takenAt: new Date().toISOString()
      });
      
      // Update local state
      setMedications(prev => 
        prev.map(med => 
          med.id === medicationId 
            ? { ...med, taken: true, takenAt: new Date().toISOString() }
            : med
        )
      );
      
      // Trigger refresh of other components
      if (onRefresh) {
        onRefresh();
      }
    } catch (error) {
      console.error('Error marking medication as taken:', error);
    }
  };

  // Check if medication is due today
  const isDueToday = (medication) => {
    if (!medication.start_date) return true; // If no start date, assume due today
    
    const today = new Date().toISOString().split('T')[0];
    const startDate = medication.start_date.split('T')[0];
    
    return today >= startDate;
  };

  // Get next dose time
  const getNextDoseTime = (medication) => {
    if (medication.taken) return 'Taken';
    
    const frequency = medication.frequency;
    const now = new Date();
    
    switch (frequency) {
      case 'daily':
        return '8:00 AM';
      case 'twice_daily':
        return now.getHours() < 12 ? '8:00 AM' : '8:00 PM';
      case 'three_times_daily':
        if (now.getHours() < 8) return '8:00 AM';
        if (now.getHours() < 14) return '2:00 PM';
        return '8:00 PM';
      case 'weekly':
        return 'Weekly';
      case 'as_needed':
        return 'As needed';
      default:
        return '8:00 AM';
    }
  };

  return (
    <>
    <DashboardCard
      title="Today's Medications"
      icon={<Pill size={20} />}
      variant="patient"
      action={
        <Button
          onClick={onAddMedication}
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
      ) : error ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center py-12"
        >
          <div className="mb-4 p-3 bg-error-50 rounded-2xl inline-block">
            <p className="text-error-600 font-medium text-sm">{error}</p>
          </div>
          <Button
            onClick={fetchMedications}
            variant="secondary"
            size="sm"
          >
            Try again
          </Button>
        </motion.div>
      ) : medications.length === 0 ? (
        <EmptyState
          icon={Pill}
          title="No medications added yet"
          description="Start tracking your medications to stay on top of your health journey"
          action={{
            label: "Add Medication",
            onClick: onAddMedication,
          }}
        />
      ) : (
        <div className="space-y-3">
          {medications.map((med, index) => (
            <motion.div
              key={med.id || index}
              initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.2,
                delay: index * 0.03,
                ease: [0.16, 1, 0.3, 1],
              }}
              whileHover={prefersReducedMotion ? {} : { y: -2 }}
              className="group flex items-center justify-between p-4 bg-white rounded-2xl border border-neutral-200 hover:border-primary-300 hover:shadow-medium transition-all duration-200"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <p className="font-semibold text-neutral-900 text-base truncate">
                    {med.medication_name || med.name || med.generic_name || 'Unknown Medication'}
                  </p>
                  {med.drug_class && (
                    <Badge className="text-xs bg-primary-50 text-primary-700 border-0 font-medium">
                      {med.drug_class}
                    </Badge>
                  )}
                  <Button
                    onClick={async () => {
                      setSelectedMedication(med);
                      setIsSideEffectsDialogOpen(true);
                      setIsLoadingSideEffects(true);
                      setSideEffects(null);
                      
                      try {
                        const response = await getMonopharmacySideEffects(
                          med.medication_name || med.name || med.generic_name
                        );
                        setSideEffects(response.side_effects || []);
                      } catch (error) {
                        console.error('Error fetching side effects:', error);
                        setSideEffects([]);
                      } finally {
                        setIsLoadingSideEffects(false);
                      }
                    }}
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0 flex-shrink-0"
                  >
                    <Info className="w-4 h-4 text-primary-600" />
                  </Button>
                </div>
                <p className="text-sm text-neutral-600 mb-1 font-medium">
                  {formatStrength(med.strength, med.unit)} • {getNextDoseTime(med)}
                </p>
                <p className="text-xs text-neutral-500">
                  {formatFrequency(med.frequency, med.customFrequency)}
                </p>
                {med.special_instructions && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-2 text-xs text-primary-700 bg-primary-50 px-3 py-1.5 rounded-lg border border-primary-100"
                  >
                    <span className="font-medium">Note:</span> {med.special_instructions}
                  </motion.div>
                )}
              </div>
              <div className="flex items-center gap-2 ml-4 flex-shrink-0">
                {!med.taken ? (
                  <Button
                    onClick={() => handleMarkTaken(med.id)}
                    variant="success"
                    size="sm"
                  >
                    <CheckCircle className="w-4 h-4 mr-1.5" />
                    Mark Taken
                  </Button>
                ) : (
                  <div className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-medical-50 text-medical-700 border border-medical-200">
                    <CheckCircle className="w-4 h-4" />
                    Taken
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </DashboardCard>

    {/* Side Effects Dialog */}
    <Dialog open={isSideEffectsDialogOpen} onOpenChange={setIsSideEffectsDialogOpen}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            Side Effects for {selectedMedication?.medication_name || selectedMedication?.name || selectedMedication?.generic_name}
          </DialogTitle>
          <DialogDescription>
            {isLoadingSideEffects ? (
              <div className="flex items-center gap-2 py-4">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600" />
                <span>Loading side effects...</span>
              </div>
            ) : sideEffects && sideEffects.length > 0 ? (
              <div className="space-y-2 py-4">
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
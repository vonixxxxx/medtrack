import { useState, useEffect } from "react";
import { Pill, Plus, Loader2, CheckCircle } from "lucide-react";
import { MedTrackCard } from "./MedTrackCard";
import { NeonButton } from "./NeonButton";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { motion } from "framer-motion";
import api from "../api";

export const TodaysMedications = ({ onAddMedication, refreshTrigger, onRefresh }) => {
  const [medications, setMedications] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch medications from API
  const fetchMedications = async () => {
    try {
      setIsLoading(true);
      const response = await api.get('meds/user');
      setMedications(response.data.medications || []);
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
    <MedTrackCard>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Pill className="w-5 h-5 text-foreground" />
          <h3 className="text-lg font-semibold">Today's Medications</h3>
        </div>
        <NeonButton onClick={onAddMedication} size="sm">
          Add Medication
        </NeonButton>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
        </div>
      ) : error ? (
        <div className="text-center py-8">
          <p className="text-muted-foreground">{error}</p>
          <button 
            onClick={fetchMedications}
            className="mt-2 text-sm text-blue-600 hover:text-blue-700"
          >
            Try again
          </button>
        </div>
      ) : medications.length === 0 ? (
        <div className="text-center py-8">
          <Pill className="w-12 h-12 text-muted-foreground/50 mx-auto mb-3" />
          <p className="text-muted-foreground">No medications added yet</p>
          <p className="text-sm text-muted-foreground/70">Click "Add" to get started</p>
        </div>
      ) : (
        <div className="space-y-3">
          {medications.map((med, index) => (
            <motion.div
              key={med.id || index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-3.5 bg-secondary/50 rounded-lg border border-border hover:bg-secondary transition-all"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <p className="font-medium text-foreground text-lg">
                    {med.medication_name || med.name || med.generic_name || 'Unknown Medication'}
                  </p>
                  {med.drug_class && (
                    <Badge variant="secondary" className="text-xs">
                      {med.drug_class}
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-muted-foreground mb-1">
                  {formatStrength(med.strength, med.unit)} • {getNextDoseTime(med)}
                </p>
                <p className="text-xs text-muted-foreground/70">
                  {formatFrequency(med.frequency, med.customFrequency)}
                </p>
                {med.special_instructions && (
                  <p className="text-xs text-blue-400/80 mt-1 italic">
                    💡 {med.special_instructions}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-2">
                {!med.taken ? (
                  <Button
                    onClick={() => handleMarkTaken(med.id)}
                    size="sm"
                    className="px-3 py-1 h-8 text-xs bg-green-600 hover:bg-green-700 text-white shadow-md hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Mark Taken
                  </Button>
                ) : (
                  <div className="flex items-center gap-1 px-3 py-1 rounded-md text-xs font-medium bg-gray-500/20 text-gray-600 border border-gray-500/30">
                    <CheckCircle className="w-3 h-3" />
                    Taken
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </MedTrackCard>
  );
};
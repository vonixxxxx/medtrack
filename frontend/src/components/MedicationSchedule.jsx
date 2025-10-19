import { useState, useEffect } from "react";
import { Calendar, Loader2, Clock } from "lucide-react";
import { MedTrackCard } from "./MedTrackCard";
import { Badge } from "./ui/badge";
import { motion } from "framer-motion";
import api from "../api";

export const MedicationSchedule = ({ refreshTrigger }) => {
  const [schedule, setSchedule] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch medication schedule from API
  const fetchSchedule = async () => {
    try {
      setIsLoading(true);
      const response = await api.get('meds/user');
      const medications = response.data.medications || [];
      
      // Transform medications into schedule format
      const scheduleData = medications.map(med => ({
        id: med.id,
        name: med.medication_name || med.name || med.generic_name || 'Unknown Medication',
        frequency: formatFrequency(med.frequency, med.customFrequency),
        next: formatNextDose(med),
        strength: formatStrength(med.strength, med.unit),
        drugClass: med.drug_class,
        taken: med.taken || false,
        special_instructions: med.special_instructions
      }));
      
      setSchedule(scheduleData);
      setError(null);
    } catch (err) {
      console.error('Error fetching schedule:', err);
      setError('Failed to load schedule');
      setSchedule([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch schedule on mount and when refreshTrigger changes
  useEffect(() => {
    fetchSchedule();
  }, [refreshTrigger]);

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

  // Format next dose time
  const formatNextDose = (medication) => {
    if (!medication) return 'Not scheduled';
    
    const frequency = medication.frequency;
    const now = new Date();
    
    // If medication is taken today, show next dose
    if (medication.taken) {
      switch (frequency) {
        case 'daily':
          return 'Tomorrow, 8:00 AM';
        case 'twice_daily':
          return now.getHours() < 12 ? 'Today, 8:00 PM' : 'Tomorrow, 8:00 AM';
        case 'three_times_daily':
          if (now.getHours() < 8) return 'Today, 8:00 AM';
          if (now.getHours() < 14) return 'Today, 2:00 PM';
          if (now.getHours() < 20) return 'Today, 8:00 PM';
          return 'Tomorrow, 8:00 AM';
        case 'weekly':
          return 'Next week';
        case 'as_needed':
          return 'As needed';
        default:
          return 'Tomorrow, 8:00 AM';
      }
    }
    
    // If not taken today, show today's dose
    switch (frequency) {
      case 'daily':
        return 'Today, 8:00 AM';
      case 'twice_daily':
        return now.getHours() < 12 ? 'Today, 8:00 AM' : 'Today, 8:00 PM';
      case 'three_times_daily':
        if (now.getHours() < 8) return 'Today, 8:00 AM';
        if (now.getHours() < 14) return 'Today, 2:00 PM';
        return 'Today, 8:00 PM';
      case 'weekly':
        return 'This week';
      case 'as_needed':
        return 'As needed';
      default:
        return 'Today, 8:00 AM';
    }
  };

  return (
    <MedTrackCard>
      <div className="flex items-center gap-2 mb-4">
        <Calendar className="w-5 h-5 text-foreground" />
        <h3 className="text-lg font-semibold">Medication Schedule</h3>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
        </div>
      ) : error ? (
        <div className="text-center py-8">
          <p className="text-muted-foreground">{error}</p>
          <button 
            onClick={fetchSchedule}
            className="mt-2 text-sm text-blue-600 hover:text-blue-700"
          >
            Try again
          </button>
        </div>
      ) : schedule.length === 0 ? (
        <div className="text-center py-8">
          <Calendar className="w-12 h-12 text-muted-foreground/50 mx-auto mb-3" />
          <p className="text-muted-foreground">No medications scheduled</p>
          <p className="text-sm text-muted-foreground/70">Add medications to see your schedule</p>
        </div>
      ) : (
        <div className="space-y-3">
          {schedule.map((item, index) => (
            <motion.div
              key={item.id || index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="p-3.5 bg-secondary/50 rounded-lg border border-border hover:bg-secondary transition-all"
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-medium text-foreground text-lg">{item.name}</p>
                    {item.drugClass && (
                      <Badge variant="secondary" className="text-xs">
                        {item.drugClass}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground mb-1">
                    {item.frequency} • {item.strength}
                  </p>
                  {item.special_instructions && (
                    <p className="text-xs text-blue-400/80 mt-1 italic">
                      💡 {item.special_instructions}
                    </p>
                  )}
                </div>
                <div className="text-right">
                  <div className="flex items-center gap-1 text-sm text-foreground">
                    <Clock className="w-4 h-4" />
                    <span>Next: {item.next}</span>
                  </div>
                  {item.taken && (
                    <div className="text-xs text-green-600 mt-1">
                      ✓ Taken today
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </MedTrackCard>
  );
};
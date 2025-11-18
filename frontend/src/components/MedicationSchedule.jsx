import { useState, useEffect } from "react";
import { Calendar, Clock } from "lucide-react";
import DashboardCard from "./DashboardCard";
import { Badge } from "./ui/badge";
import { LoadingSkeleton } from "./dashboard/LoadingSkeleton";
import { EmptyState } from "./dashboard/EmptyState";
import { motion, useReducedMotion } from "framer-motion";
import api from "../api";

export const MedicationSchedule = ({ refreshTrigger }) => {
  const prefersReducedMotion = useReducedMotion();
  const [schedule, setSchedule] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch medication schedule from API
  const fetchSchedule = async () => {
    try {
      setIsLoading(true);
      const response = await api.get('meds/user');
      const medications = response.data.medications || [];
      
      // Filter out test medications
      const filteredMeds = medications.filter(med => {
        const name = (med.medication_name || med.name || med.generic_name || '').toLowerCase();
        return !name.includes('final test') && 
               !name.includes('test2') && 
               !name.includes('test medication');
      });
      
      // Transform medications into schedule format
      const scheduleData = filteredMeds.map(med => ({
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
    <DashboardCard
      title="Medication Schedule"
      icon={<Calendar size={20} />}
      variant="patient"
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
          <button 
            onClick={fetchSchedule}
            className="text-sm text-primary-600 hover:text-primary-700 font-semibold transition-colors"
          >
            Try again
          </button>
        </motion.div>
      ) : schedule.length === 0 ? (
        <EmptyState
          icon={Calendar}
          title="No medications scheduled"
          description="Add medications to see your schedule and upcoming doses"
        />
      ) : (
        <div className="space-y-3">
          {schedule.map((item, index) => (
            <motion.div
              key={item.id || index}
              initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.2,
                delay: index * 0.03,
                ease: [0.16, 1, 0.3, 1],
              }}
              whileHover={prefersReducedMotion ? {} : { y: -2 }}
              className="group p-4 bg-white rounded-2xl border border-neutral-200 hover:border-primary-300 hover:shadow-medium transition-all duration-200"
            >
              <div className="flex justify-between items-start gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <p className="font-semibold text-neutral-900 text-base truncate">{item.name}</p>
                    {item.drugClass && (
                      <Badge className="text-xs bg-primary-50 text-primary-700 border-0 font-medium">
                        {item.drugClass}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-neutral-600 mb-1 font-medium">
                    {item.frequency} • {item.strength}
                  </p>
                  {item.special_instructions && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="mt-2 text-xs text-primary-700 bg-primary-50 px-3 py-1.5 rounded-lg border border-primary-100"
                    >
                      <span className="font-medium">Note:</span> {item.special_instructions}
                    </motion.div>
                  )}
                </div>
                <div className="text-right flex-shrink-0">
                  <div className="flex items-center gap-1.5 text-sm text-neutral-700 font-semibold mb-1">
                    <Clock className="w-4 h-4 text-neutral-500" />
                    <span>{item.next}</span>
                  </div>
                  {item.taken && (
                    <div className="text-xs text-medical-700 font-semibold bg-medical-50 px-2 py-1 rounded-lg border border-medical-200 inline-block">
                      ✓ Taken today
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </DashboardCard>
  );
};
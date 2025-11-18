import { useState, useEffect } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { FileText, Plus, Calendar, User, Clock } from "lucide-react";
import DashboardCard from "../../DashboardCard";
import { Button } from "../../ui/button";
import { Badge } from "../../ui/badge";
import { LoadingSkeleton } from "../../dashboard/LoadingSkeleton";
import { EmptyState } from "../../dashboard/EmptyState";
import { getEncounters } from "../../../api";

export const EncounterList = ({ patientId, onAddEncounter, refreshTrigger }) => {
  const prefersReducedMotion = useReducedMotion();
  const [encounters, setEncounters] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchEncounters();
  }, [patientId, refreshTrigger]);

  const fetchEncounters = async () => {
    try {
      setIsLoading(true);
      const params = patientId ? { patientId } : {};
      const data = await getEncounters(params);
      setEncounters(data || []);
      setError(null);
    } catch (err) {
      console.error('Error fetching encounters:', err);
      setError('Failed to load encounters');
      setEncounters([]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const formatTime = (timeString) => {
    if (!timeString) return '';
    const time = new Date(`1970-01-01T${timeString}`);
    return time.toLocaleTimeString('en-US', { 
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  const getStatusColor = (status) => {
    const colors = {
      planned: 'bg-primary-50 text-primary-700',
      arrived: 'bg-medical-50 text-medical-700',
      triaged: 'bg-warning-50 text-warning-700',
      'in-progress': 'bg-primary-100 text-primary-800',
      finished: 'bg-medical-50 text-medical-700',
      cancelled: 'bg-error-50 text-error-700'
    };
    return colors[status] || 'bg-neutral-50 text-neutral-700';
  };

  return (
    <DashboardCard
      title="Encounters"
      icon={<FileText size={20} />}
      variant="clinician"
      action={
        onAddEncounter && (
          <Button onClick={onAddEncounter} variant="primary" size="sm">
            <Plus size={16} className="mr-1.5" />
            New Encounter
          </Button>
        )
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
          <Button onClick={fetchEncounters} variant="secondary" size="sm">
            Try again
          </Button>
        </motion.div>
      ) : encounters.length === 0 ? (
        <EmptyState
          icon={FileText}
          title="No encounters"
          description="Create an encounter to document a patient visit"
          action={onAddEncounter ? {
            label: "New Encounter",
            onClick: onAddEncounter
          } : null}
        />
      ) : (
        <div className="space-y-3">
          {encounters.map((encounter, index) => (
            <motion.div
              key={encounter.id || index}
              initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.2,
                delay: index * 0.03,
                ease: [0.16, 1, 0.3, 1]
              }}
              whileHover={prefersReducedMotion ? {} : { y: -2 }}
              className="group p-4 bg-white rounded-2xl border border-neutral-200 hover:border-primary-300 hover:shadow-medium transition-all duration-200 cursor-pointer"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h4 className="font-semibold text-neutral-900 text-base">
                      {encounter.encounterType || 'Encounter'}
                    </h4>
                    <Badge className={`text-xs ${getStatusColor(encounter.status)} border-0 font-medium`}>
                      {encounter.status?.replace('-', ' ')}
                    </Badge>
                  </div>
                  
                  <div className="space-y-1.5 text-sm text-neutral-600">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-neutral-400" />
                      <span className="font-medium">{formatDate(encounter.encounterDate)}</span>
                      {encounter.encounterTime && (
                        <>
                          <Clock className="w-4 h-4 text-neutral-400 ml-2" />
                          <span>{formatTime(encounter.encounterTime)}</span>
                        </>
                      )}
                    </div>
                    {encounter.patient?.user?.name && (
                      <div className="flex items-center gap-2">
                        <User className="w-4 h-4 text-neutral-400" />
                        <span>{encounter.patient.user.name}</span>
                      </div>
                    )}
                    {encounter.reason && (
                      <p className="text-neutral-500 mt-2">{encounter.reason}</p>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </DashboardCard>
  );
};




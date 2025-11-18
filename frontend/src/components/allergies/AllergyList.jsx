import { useState, useEffect } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, Plus } from "lucide-react";
import DashboardCard from "../DashboardCard";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { LoadingSkeleton } from "../dashboard/LoadingSkeleton";
import { EmptyState } from "../dashboard/EmptyState";
import { getAllergies } from "../../api";

export const AllergyList = ({ patientId, onAddAllergy, refreshTrigger, readOnly = false }) => {
  const prefersReducedMotion = useReducedMotion();
  const [allergies, setAllergies] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAllergies();
  }, [patientId, refreshTrigger]);

  const fetchAllergies = async () => {
    try {
      setIsLoading(true);
      const params = patientId ? { patientId } : {};
      const data = await getAllergies(params);
      setAllergies(data || []);
      setError(null);
    } catch (err) {
      console.error('Error fetching allergies:', err);
      setError('Failed to load allergies');
      setAllergies([]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return null;
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const getSeverityColor = (severity) => {
    const colors = {
      mild: 'bg-warning-50 text-warning-700',
      moderate: 'bg-warning-100 text-warning-800',
      severe: 'bg-error-50 text-error-700'
    };
    return colors[severity] || 'bg-neutral-50 text-neutral-700';
  };

  return (
    <DashboardCard
      title="Allergies"
      icon={<AlertTriangle size={20} />}
      variant={readOnly ? "patient" : "clinician"}
      action={
        !readOnly && onAddAllergy && (
          <Button onClick={onAddAllergy} variant="primary" size="sm">
            <Plus size={16} className="mr-1.5" />
            Add Allergy
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
          <Button onClick={fetchAllergies} variant="secondary" size="sm">
            Try again
          </Button>
        </motion.div>
      ) : allergies.length === 0 ? (
        <EmptyState
          icon={AlertTriangle}
          title="No allergies recorded"
          description={readOnly ? "Your allergy list is empty" : "Add allergies to track patient reactions"}
          action={!readOnly && onAddAllergy ? {
            label: "Add Allergy",
            onClick: onAddAllergy
          } : null}
        />
      ) : (
        <div className="space-y-3">
          {allergies.map((allergy, index) => (
            <motion.div
              key={allergy.id || index}
              initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.2,
                delay: index * 0.03,
                ease: [0.16, 1, 0.3, 1]
              }}
              whileHover={prefersReducedMotion ? {} : { y: -2 }}
              className="group p-4 bg-white rounded-2xl border border-neutral-200 hover:border-primary-300 hover:shadow-medium transition-all duration-200"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h4 className="font-semibold text-neutral-900 text-base">
                      {allergy.allergen}
                    </h4>
                    {allergy.allergenType && (
                      <Badge variant="outline" className="text-xs">
                        {allergy.allergenType}
                      </Badge>
                    )}
                    {allergy.severity && (
                      <Badge className={`text-xs ${getSeverityColor(allergy.severity)} border-0 font-medium`}>
                        {allergy.severity}
                      </Badge>
                    )}
                    {allergy.status === 'active' && (
                      <Badge className="text-xs bg-error-50 text-error-700 border-0 font-medium">
                        Active
                      </Badge>
                    )}
                  </div>
                  
                  <div className="space-y-1 text-sm text-neutral-600">
                    {allergy.reaction && (
                      <p><span className="font-medium">Reaction:</span> {allergy.reaction}</p>
                    )}
                    {allergy.onsetDate && (
                      <p>Onset: {formatDate(allergy.onsetDate)}</p>
                    )}
                    {allergy.notes && (
                      <p className="text-neutral-500 mt-2">{allergy.notes}</p>
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




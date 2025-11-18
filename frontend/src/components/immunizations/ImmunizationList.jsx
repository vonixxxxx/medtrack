import { useState, useEffect } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Syringe, Plus, Calendar } from "lucide-react";
import DashboardCard from "../DashboardCard";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { LoadingSkeleton } from "../dashboard/LoadingSkeleton";
import { EmptyState } from "../dashboard/EmptyState";
import { getImmunizations } from "../../api";

export const ImmunizationList = ({ patientId, onAddImmunization, refreshTrigger, readOnly = false }) => {
  const prefersReducedMotion = useReducedMotion();
  const [immunizations, setImmunizations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchImmunizations();
  }, [patientId, refreshTrigger]);

  const fetchImmunizations = async () => {
    try {
      setIsLoading(true);
      const params = patientId ? { patientId } : {};
      const data = await getImmunizations(params);
      // Ensure data is always an array
      const immunizationsArray = Array.isArray(data) ? data : [];
      setImmunizations(immunizationsArray);
      setError(null);
    } catch (err) {
      console.error('Error fetching immunizations:', err);
      setError('Failed to load immunizations');
      setImmunizations([]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return null;
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  return (
    <DashboardCard
      title="Immunizations"
      icon={<Syringe size={20} />}
      variant={readOnly ? "patient" : "clinician"}
      action={
        !readOnly && onAddImmunization && (
          <Button onClick={onAddImmunization} variant="primary" size="sm">
            <Plus size={16} className="mr-1.5" />
            Add Immunization
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
          <Button onClick={fetchImmunizations} variant="secondary" size="sm">
            Try again
          </Button>
        </motion.div>
      ) : (Array.isArray(immunizations) ? immunizations : []).length === 0 ? (
        <EmptyState
          icon={Syringe}
          title="No immunizations recorded"
          description={readOnly ? "Your immunization history is empty" : "Add immunizations to track patient vaccines"}
          action={!readOnly && onAddImmunization ? {
            label: "Add Immunization",
            onClick: onAddImmunization
          } : null}
        />
      ) : (
        <div className="space-y-3">
          {(Array.isArray(immunizations) ? immunizations : []).map((immunization, index) => (
            <motion.div
              key={immunization.id || index}
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
                      {immunization.vaccineName}
                    </h4>
                    {immunization.vaccineCode && (
                      <Badge variant="outline" className="text-xs">
                        {immunization.vaccineCode}
                      </Badge>
                    )}
                  </div>
                  
                  <div className="space-y-1 text-sm text-neutral-600">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-neutral-400" />
                      <span>Administered: {formatDate(immunization.administrationDate)}</span>
                    </div>
                    {immunization.route && (
                      <p>Route: <span className="font-medium">{immunization.route}</span></p>
                    )}
                    {immunization.site && (
                      <p>Site: <span className="font-medium">{immunization.site}</span></p>
                    )}
                    {immunization.dose && (
                      <p>Dose: <span className="font-medium">{immunization.dose}</span></p>
                    )}
                    {immunization.manufacturer && (
                      <p>Manufacturer: <span className="font-medium">{immunization.manufacturer}</span></p>
                    )}
                    {immunization.lotNumber && (
                      <p>Lot #: <span className="font-medium">{immunization.lotNumber}</span></p>
                    )}
                    {immunization.provider && (
                      <p>Provider: <span className="font-medium">{immunization.provider}</span></p>
                    )}
                    {immunization.notes && (
                      <p className="text-neutral-500 mt-2">{immunization.notes}</p>
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


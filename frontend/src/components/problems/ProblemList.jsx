import { useState, useEffect } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { AlertCircle, Plus, CheckCircle, XCircle } from "lucide-react";
import DashboardCard from "../DashboardCard";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { LoadingSkeleton } from "../dashboard/LoadingSkeleton";
import { EmptyState } from "../dashboard/EmptyState";
import { getProblems } from "../../api";

export const ProblemList = ({ patientId, onAddProblem, refreshTrigger, readOnly = false }) => {
  const prefersReducedMotion = useReducedMotion();
  const [problems, setProblems] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchProblems();
  }, [patientId, refreshTrigger]);

  const fetchProblems = async () => {
    try {
      setIsLoading(true);
      const params = patientId ? { patientId } : {};
      const data = await getProblems(params);
      setProblems(data || []);
      setError(null);
    } catch (err) {
      console.error('Error fetching problems:', err);
      setError('Failed to load problems');
      setProblems([]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return null;
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active':
        return <AlertCircle className="w-4 h-4 text-error-500" />;
      case 'resolved':
        return <CheckCircle className="w-4 h-4 text-medical-500" />;
      case 'inactive':
        return <XCircle className="w-4 h-4 text-neutral-400" />;
      default:
        return <AlertCircle className="w-4 h-4 text-neutral-400" />;
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      active: 'bg-error-50 text-error-700',
      resolved: 'bg-medical-50 text-medical-700',
      inactive: 'bg-neutral-100 text-neutral-600'
    };
    return colors[status] || 'bg-neutral-50 text-neutral-700';
  };

  return (
    <DashboardCard
      title="Problem List"
      icon={<AlertCircle size={20} />}
      variant={readOnly ? "patient" : "clinician"}
      action={
        !readOnly && onAddProblem && (
          <Button onClick={onAddProblem} variant="primary" size="sm">
            <Plus size={16} className="mr-1.5" />
            Add Problem
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
          <Button onClick={fetchProblems} variant="secondary" size="sm">
            Try again
          </Button>
        </motion.div>
      ) : problems.length === 0 ? (
        <EmptyState
          icon={AlertCircle}
          title="No problems recorded"
          description={readOnly ? "Your problem list is empty" : "Add problems to track patient conditions"}
          action={!readOnly && onAddProblem ? {
            label: "Add Problem",
            onClick: onAddProblem
          } : null}
        />
      ) : (
        <div className="space-y-3">
          {problems.map((problem, index) => (
            <motion.div
              key={problem.id || index}
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
                    {getStatusIcon(problem.status)}
                    <h4 className="font-semibold text-neutral-900 text-base">
                      {problem.title}
                    </h4>
                    <Badge className={`text-xs ${getStatusColor(problem.status)} border-0 font-medium`}>
                      {problem.status}
                    </Badge>
                    {problem.code && (
                      <Badge variant="outline" className="text-xs">
                        {problem.code}
                      </Badge>
                    )}
                  </div>
                  
                  <div className="space-y-1 text-sm text-neutral-600">
                    {problem.beginDate && (
                      <p>Started: {formatDate(problem.beginDate)}</p>
                    )}
                    {problem.endDate && (
                      <p>Resolved: {formatDate(problem.endDate)}</p>
                    )}
                    {problem.severity && (
                      <p>Severity: <span className="font-medium">{problem.severity}</span></p>
                    )}
                    {problem.notes && (
                      <p className="text-neutral-500 mt-2">{problem.notes}</p>
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




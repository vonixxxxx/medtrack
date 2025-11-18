import { useState, useEffect } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { DollarSign, Plus, FileText } from "lucide-react";
import DashboardCard from "../../DashboardCard";
import { Button } from "../../ui/button";
import { Badge } from "../../ui/badge";
import { LoadingSkeleton } from "../../dashboard/LoadingSkeleton";
import { EmptyState } from "../../dashboard/EmptyState";
import { getCharges, createCharge } from "../../../api";

export const ChargeCapture = ({ patientId, encounterId, onAddCharge, refreshTrigger }) => {
  const prefersReducedMotion = useReducedMotion();
  const [charges, setCharges] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchCharges();
  }, [patientId, encounterId, refreshTrigger]);

  const fetchCharges = async () => {
    try {
      setIsLoading(true);
      const params = {};
      if (patientId) params.patientId = patientId;
      if (encounterId) params.encounterId = encounterId;
      const data = await getCharges(params);
      setCharges(data || []);
      setError(null);
    } catch (err) {
      console.error('Error fetching charges:', err);
      setError('Failed to load charges');
      setCharges([]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const getStatusColor = (status) => {
    const colors = {
      pending: 'bg-warning-50 text-warning-700',
      billed: 'bg-primary-50 text-primary-700',
      paid: 'bg-medical-50 text-medical-700',
      denied: 'bg-error-50 text-error-700',
      cancelled: 'bg-neutral-100 text-neutral-600'
    };
    return colors[status] || 'bg-neutral-50 text-neutral-700';
  };

  return (
    <DashboardCard
      title="Charges"
      icon={<DollarSign size={20} />}
      variant="clinician"
      action={
        onAddCharge && (
          <Button onClick={onAddCharge} variant="primary" size="sm">
            <Plus size={16} className="mr-1.5" />
            Add Charge
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
          <Button onClick={fetchCharges} variant="secondary" size="sm">
            Try again
          </Button>
        </motion.div>
      ) : charges.length === 0 ? (
        <EmptyState
          icon={DollarSign}
          title="No charges"
          description="Add charges to capture billing information"
          action={onAddCharge ? {
            label: "Add Charge",
            onClick: onAddCharge
          } : null}
        />
      ) : (
        <div className="space-y-3">
          {charges.map((charge, index) => (
            <motion.div
              key={charge.id || index}
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
                      {charge.description || charge.code}
                    </h4>
                    <Badge className={`text-xs ${getStatusColor(charge.status)} border-0 font-medium`}>
                      {charge.status}
                    </Badge>
                  </div>
                  
                  <div className="space-y-1 text-sm text-neutral-600">
                    <p><span className="font-medium">Code:</span> {charge.code} ({charge.codeType})</p>
                    <p><span className="font-medium">Date:</span> {formatDate(charge.dateOfService)}</p>
                    {charge.units > 1 && (
                      <p><span className="font-medium">Units:</span> {charge.units}</p>
                    )}
                  </div>
                </div>
                
                <div className="text-right">
                  <p className="text-xl font-bold text-neutral-900">
                    {formatCurrency(charge.fee * (charge.units || 1))}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </DashboardCard>
  );
};




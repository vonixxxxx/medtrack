import { useState, useEffect } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Calendar as CalendarIcon, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';
import { LoadingSkeleton } from '../dashboard/LoadingSkeleton';
import api from '../../api';

export const MedicationHistoryCalendar = ({ medicationId, patientId, days = 10 }) => {
  const prefersReducedMotion = useReducedMotion();
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadHistory();
  }, [medicationId, patientId, days]);

  const loadHistory = async () => {
    try {
      setIsLoading(true);
      // Get adherence logs for the last N days
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - days);

      const params = {
        medicationId,
        patientId,
        startDate: startDate.toISOString().split('T')[0],
        endDate: endDate.toISOString().split('T')[0]
      };

      const data = await api.get('/adherence', { params });
      const logs = data.data?.logs || [];

      // Transform logs into calendar format
      const calendarData = [];
      for (let i = days - 1; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        const dateKey = date.toISOString().split('T')[0];
        
        const dayLog = logs.find(log => {
          const logDate = new Date(log.date).toISOString().split('T')[0];
          return logDate === dateKey;
        });

        calendarData.push({
          date: dateKey,
          status: dayLog?.status || 'none',
          taken: dayLog?.status === 'taken',
          missed: dayLog?.status === 'missed',
          skipped: dayLog?.status === 'skipped',
          delayed: dayLog?.status === 'delayed'
        });
      }

      setHistory(calendarData);
      setError(null);
    } catch (err) {
      console.error('Error loading medication history:', err);
      setError('Failed to load medication history');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const formatValue = (status) => {
    switch (status) {
      case 'taken':
        return 'Medication Taken';
      case 'missed':
        return 'Medication NOT Taken';
      case 'skipped':
        return 'Medication Skipped';
      case 'delayed':
        return 'Medication Delayed';
      default:
        return 'No Data';
    }
  };

  const getColorClass = (status) => {
    switch (status) {
      case 'taken':
        return 'bg-medical-500';
      case 'missed':
        return 'bg-error-500';
      case 'skipped':
        return 'bg-warning-500';
      case 'delayed':
        return 'bg-info-500';
      default:
        return 'bg-neutral-200';
    }
  };

  const getIcon = (status) => {
    switch (status) {
      case 'taken':
        return <CheckCircle className="w-3 h-3 text-white" />;
      case 'missed':
        return <XCircle className="w-3 h-3 text-white" />;
      case 'skipped':
        return <AlertCircle className="w-3 h-3 text-white" />;
      default:
        return null;
    }
  };

  if (isLoading) {
    return (
      <DashboardCard
        title={`Medication History (Last ${days} Days)`}
        icon={<CalendarIcon size={20} />}
        variant="patient"
      >
        <LoadingSkeleton variant="card" />
      </DashboardCard>
    );
  }

  if (error) {
    return (
      <DashboardCard
        title={`Medication History (Last ${days} Days)`}
        icon={<CalendarIcon size={20} />}
        variant="patient"
      >
        <div className="text-center py-8">
          <p className="text-sm text-error-600">{error}</p>
        </div>
      </DashboardCard>
    );
  }

  return (
    <DashboardCard
      title={`Medication History (Last ${days} Days)`}
      icon={<CalendarIcon size={20} />}
      variant="patient"
    >
      <div className="space-y-4">
        <div className="grid grid-cols-10 gap-2">
          <TooltipProvider>
            {history.map((day, index) => (
              <Tooltip key={day.date} delayDuration={50}>
                <TooltipTrigger asChild>
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.02 }}
                    className={`w-6 h-6 rounded-sm shadow-sm flex items-center justify-center transition-all ${
                      getColorClass(day.status)
                    } ${day.status !== 'none' ? 'cursor-pointer hover:scale-110' : ''}`}
                  >
                    {getIcon(day.status)}
                  </motion.div>
                </TooltipTrigger>
                <TooltipContent
                  className={day.status === 'taken' ? 'bg-medical-600' : day.status === 'missed' ? 'bg-error-600' : 'bg-neutral-600'}
                >
                  <div className="text-xs">
                    <p className="font-semibold">{formatDate(day.date)}</p>
                    <p>{formatValue(day.status)}</p>
                  </div>
                </TooltipContent>
              </Tooltip>
            ))}
          </TooltipProvider>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-4 text-xs pt-2 border-t border-neutral-200">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-sm bg-medical-500" />
            <span className="text-neutral-600">Taken</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-sm bg-error-500" />
            <span className="text-neutral-600">Missed</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-sm bg-warning-500" />
            <span className="text-neutral-600">Skipped</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-sm bg-neutral-200" />
            <span className="text-neutral-600">No Data</span>
          </div>
        </div>
      </div>
    </DashboardCard>
  );
};



